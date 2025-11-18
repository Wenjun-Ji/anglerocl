"""Modules for creating adversarial object patch."""

import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adv_patch_gen.utils.median_pool import MedianPool2d


class PatchTransformer(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches, randomly adjusting brightness and
    contrast, adding random amount of noise, and rotating randomly. Resizes patches according to as size based on the
    batch of labels, and pads them to the dimension of an image.
    """

    def __init__(
        self,
        t_size_frac: Union[float, Tuple[float, float]] = 0.3,
        mul_gau_mean: Union[float, Tuple[float, float]] = (0.5, 0.8),
        mul_gau_std: Union[float, Tuple[float, float]] = 0.1,
        x_off_loc: Tuple[float, float] = [-0.25, 0.25],
        y_off_loc: Tuple[float, float] = [-0.25, 0.25],
        dev: torch.device = torch.device("cuda:0"),
    ):
        super(PatchTransformer, self).__init__()
        # convert to duplicated lists/tuples to unpack and send to np.random.uniform
        self.t_size_frac = [t_size_frac, t_size_frac] if isinstance(t_size_frac, float) else t_size_frac
        self.m_gau_mean = [mul_gau_mean, mul_gau_mean] if isinstance(mul_gau_mean, float) else mul_gau_mean
        self.m_gau_std = [mul_gau_std, mul_gau_std] if isinstance(mul_gau_std, float) else mul_gau_std
        assert (
            len(self.t_size_frac) == 2 and len(self.m_gau_mean) == 2 and len(self.m_gau_std) == 2
        ), "Range must have 2 values"
        self.x_off_loc = x_off_loc
        self.y_off_loc = y_off_loc
        self.dev = dev
        self.min_contrast = 0.8
        self.max_contrast = 1.2
        self.min_brightness = -0.1
        self.max_brightness = 0.1
        self.noise_factor = 0.10
        self.minangle = -20 / 180 * math.pi
        self.maxangle = 20 / 180 * math.pi
        self.medianpooler = MedianPool2d(kernel_size=7, same=True)

        self.tensor = torch.FloatTensor if "cpu" in str(dev) else torch.cuda.FloatTensor

    def forward(
        self, adv_patch, lab_batch, model_in_sz, use_mul_add_gau=True, do_transforms=True, do_rotate=True, rand_loc=True
    ):
        # add gaussian noise to reduce contrast with a stohastic process
        p_c, p_h, p_w = adv_patch.shape
        if use_mul_add_gau:
            mul_gau = torch.normal(
                np.random.uniform(*self.m_gau_mean),
                np.random.uniform(*self.m_gau_std),
                (p_c, p_h, p_w),
                device=self.dev,
            )
            add_gau = torch.normal(0, 0.001, (p_c, p_h, p_w), device=self.dev)
            adv_patch = adv_patch * mul_gau + add_gau
        adv_patch = self.medianpooler(adv_patch.unsqueeze(0))
        m_h, m_w = model_in_sz
        # Determine size of padding
        pad = (m_w - adv_patch.size(-1)) / 2
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0)
        adv_batch = adv_patch.expand(
            lab_batch.size(0), lab_batch.size(1), -1, -1, -1
        )  # [bsize, max_bbox_labels, pchannel, pheight, pwidth]
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        # Contrast, brightness and noise transforms
        if do_transforms:
            # Create random contrast tensor
            contrast = self.tensor(batch_size).uniform_(self.min_contrast, self.max_contrast)
            contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            contrast = contrast.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))

            # Create random brightness tensor
            brightness = self.tensor(batch_size).uniform_(self.min_brightness, self.max_brightness)
            brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            brightness = brightness.expand(-1, -1, adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1))
            
            # Create random noise tensor
            noise = self.tensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

            # Apply contrast/brightness/noise, clamp
            adv_batch = adv_batch * contrast + brightness + noise

            adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)

        # Where the label class_id is 1 we don't want a patch (padding) --> fill mask with zero's
        cls_ids = lab_batch[..., 0].unsqueeze(-1)  # equiv to torch.narrow(lab_batch, 2, 0, 1)
        cls_mask = cls_ids.expand(-1, -1, p_c)
        cls_mask = cls_mask.unsqueeze(-1)
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(3))
        cls_mask = cls_mask.unsqueeze(-1)
        # [bsize, max_bbox_labels, pchannel, pheight, pwidth]
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(4))
        msk_batch = self.tensor(cls_mask.size()).fill_(1)

        # Pad patch and mask to image dimensions
        patch_pad = nn.ConstantPad2d((int(pad + 0.5), int(pad), int(pad + 0.5), int(pad)), 0)
        adv_batch = patch_pad(adv_batch)
        msk_batch = patch_pad(msk_batch)

        # Rotation and rescaling transforms
        anglesize = lab_batch.size(0) * lab_batch.size(1)
        if do_rotate:
            angle = self.tensor(anglesize).uniform_(self.minangle, self.maxangle)
        else:
            angle = self.tensor(anglesize).fill_(0)

        # Resizes and rotates
        current_patch_size = adv_patch.size(-1)
        lab_batch_scaled = self.tensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * m_w
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * m_w
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * m_w
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * m_w
        tsize = np.random.uniform(*self.t_size_frac)
        target_size = torch.sqrt(
            ((lab_batch_scaled[:, :, 3].mul(tsize)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(tsize)) ** 2)
        )

        target_x = lab_batch[:, :, 1].view(np.prod(batch_size))
        target_y = lab_batch[:, :, 2].view(np.prod(batch_size))
        targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size))
        targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size))
        if rand_loc:
            off_x = targetoff_x * (self.tensor(targetoff_x.size()).uniform_(*self.x_off_loc))
            target_x = target_x + off_x
            off_y = targetoff_y * (self.tensor(targetoff_y.size()).uniform_(*self.x_off_loc))
            target_y = target_y + off_y
        scale = target_size / current_patch_size
        scale = scale.view(anglesize)

        s = adv_batch.size()
        adv_batch = adv_batch.view(s[0] * s[1], s[2], s[3], s[4])
        msk_batch = msk_batch.view(s[0] * s[1], s[2], s[3], s[4])

        tx = (-target_x + 0.5) * 2
        ty = (-target_y + 0.5) * 2
        sin = torch.sin(angle)
        cos = torch.cos(angle)

        # Theta = rotation/rescale matrix
        # Theta = input batch of affine matrices with shape (N×2×3) for 2D or (N×3×4) for 3D
        theta = self.tensor(anglesize, 2, 3).fill_(0)
        theta[:, 0, 0] = cos / scale
        theta[:, 0, 1] = sin / scale
        theta[:, 0, 2] = tx * cos / scale + ty * sin / scale
        theta[:, 1, 0] = -sin / scale
        theta[:, 1, 1] = cos / scale
        theta[:, 1, 2] = -tx * sin / scale + ty * cos / scale

        grid = F.affine_grid(theta, adv_batch.shape)
        adv_batch_t = F.grid_sample(adv_batch, grid)
        msk_batch_t = F.grid_sample(msk_batch, grid)

        adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4])
        msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4])

        adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)

        return adv_batch_t * msk_batch_t


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.
    The patch (adv_batch) has the same size as the image, just is zero everywhere there isn't a patch.
    If patch_alpha == 1 (default), just overwrite the background image values with the patch values.
    Else, blend the patch with the image
    See: https://learnopencv.com/alpha-blending-using-opencv-cpp-python/
         https://stackoverflow.com/questions/49737541/merge-two-images-with-alpha-channel/49738078
        I = \alpha F + (1 - \alpha) B
            F = foregraound (patch, or adv_batch)
            B = background (image, or img_batch)
    """

    def __init__(self, patch_alpha: float = 1):
        super(PatchApplier, self).__init__()
        self.patch_alpha = patch_alpha

    def forward(self, img_batch, adv_batch):
        advs = torch.unbind(adv_batch, 1)
        for adv in advs:
            # replace image values with patch values
            if self.patch_alpha == 1:
                img_batch = torch.where((adv == 0), img_batch, adv)
            # alpha blend
            else:
                # get combo of image and adv
                alpha_blend = self.patch_alpha * adv + (1.0 - self.patch_alpha) * img_batch
                # apply alpha blend where the patch is non-zero
                img_batch = torch.where((adv == 0), img_batch, alpha_blend)

        return img_batch
    

import torch
import kornia
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.transform import warp_perspective


class PerspectiveViewGenerator:
    """
    透视视图生成器，用于从2D图像生成多角度视图
    适用于训练角度鲁棒性补丁
    """
    
    def __init__(self, dev, image_size=512, fov=60.0, distance=5.0):
        """
        初始化透视视图生成器
        
        参数:
            dev: 运算设备 (CPU或CUDA)
            image_size: 输出图像尺寸
            fov: 视场角度 (degrees)
            distance: 相机与图像的默认距离
        """
        self.dev = dev
        self.image_size = image_size
        self.fov = fov
        self.default_distance = distance
    
    def __call__(self, image, longitudes=None, latitudes=None, distance=None):
        """
        为给定图像生成多个视角的渲染图
        
        参数:
            image: 输入图像张量, 形状为 [B, C, H, W]
            longitudes: 经度角度列表，范围[-90, 90]度，控制水平视角
            latitudes: 纬度角度列表，范围[-90, 90]度，控制垂直视角
            distance: 相机距离，如果为None则使用默认值
            
        返回:
            rendered_views: 渲染后的视图 [V, B, C, H, W] 其中V是视角数
            longitude_angles: 对应的经度角度 [V]
            latitude_angles: 对应的纬度角度 [V]
        """
        # 确保图像是4D张量 [B, C, H, W]
        assert image.dim() == 4, f"图像张量应为4D [B,C,H,W]，但当前是{image.dim()}D"
        
        batch_size, channels, height, width = image.shape
        
        # 使用默认值如果参数未指定
        if longitudes is None:
            longitudes = [-30, 0, 30]  # 默认三个水平视角
        if latitudes is None:
            latitudes = [0]  # 默认只有水平视角
        if distance is None:
            distance = self.default_distance
            
        # 计算视角总数
        num_views = len(longitudes) * len(latitudes)
        
        # 准备存储所有视角的张量
        all_views = []
        all_longitudes = []
        all_latitudes = []
        
        for lat in latitudes:
            for lon in longitudes:
                # 为每个角度创建视图
                view = self.create_perspective_view(
                    image, 
                    longitude=lon, 
                    latitude=lat, 
                    distance=distance,
                    output_size=(self.image_size, self.image_size)
                )
                # view形状: [B, C, H, W]
                all_views.append(view)
                all_longitudes.append(lon)
                all_latitudes.append(lat)
        
        # 堆叠所有视图 [V, B, C, H, W]
        rendered_views = torch.stack(all_views, dim=0)
        longitude_angles = torch.tensor(all_longitudes, device=self.dev)
        latitude_angles = torch.tensor(all_latitudes, device=self.dev)
        
        return rendered_views, longitude_angles, latitude_angles
    
    def create_perspective_view(self, image, longitude, latitude, distance, output_size=None):
        """
        创建给定2D图像的透视视图
        
        参数:
            image: 输入图像张量, 形状为 [B, C, H, W]
            longitude: 经度，范围[-90, 90]度，控制水平方向的视角
            latitude: 纬度，范围[-90, 90]度，控制垂直方向的视角
            distance: 相机与图像的距离
            output_size: 输出图像尺寸, 默认使用原图尺寸
        
        返回:
            透视变换后的图像 [B, C, H, W]
        """
        batch_size, _, height, width = image.shape
        
        # 如果没有指定输出尺寸，使用原图尺寸
        if output_size is None:
            output_size = (height, width)
        
        # 将角度转换为弧度
        longitude_rad = np.radians(longitude)
        latitude_rad = np.radians(latitude)
        
        # 图像中心点在3D空间的坐标
        center_3d = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.dev)
        
        # 相机在3D空间的坐标 (假设图像位于yz平面)
        # 使用球坐标系: x = d*cos(lat)*cos(lon), y = d*cos(lat)*sin(lon), z = d*sin(lat)
        camera_position = torch.tensor([
            distance * np.cos(latitude_rad) * np.cos(longitude_rad),
            distance * np.cos(latitude_rad) * np.sin(longitude_rad),
            distance * np.sin(latitude_rad)
        ], dtype=torch.float32, device=self.dev)
        
        # 相机的正上方向（一般选择z轴正方向）
        up_vector = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.dev)
        
        # 相机的方向向量 (从相机指向图像中心点)
        direction_vector = center_3d - camera_position
        direction_vector = direction_vector / torch.norm(direction_vector)
        
        # 相机的右向量 (cross product of direction and up)
        right_vector = torch.cross(direction_vector, up_vector)
        right_vector = right_vector / torch.norm(right_vector)
        
        # 相机的真正的上向量 (cross product of right and direction)
        true_up_vector = torch.cross(right_vector, direction_vector)
        true_up_vector = true_up_vector / torch.norm(true_up_vector)
        
        # 构建视图矩阵
        view_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.dev)
        view_matrix[0, :3] = right_vector
        view_matrix[1, :3] = true_up_vector
        view_matrix[2, :3] = -direction_vector
        view_matrix[0, 3] = -torch.dot(right_vector, camera_position)
        view_matrix[1, 3] = -torch.dot(true_up_vector, camera_position)
        view_matrix[2, 3] = torch.dot(direction_vector, camera_position)
        view_matrix[3, 3] = 1.0
        
        # 构建投影矩阵 (透视投影)
        aspect_ratio = width / height
        near_plane = 0.1
        far_plane = distance * 2
        
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2)
        
        projection_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.dev)
        projection_matrix[0, 0] = f / aspect_ratio
        projection_matrix[1, 1] = f
        projection_matrix[2, 2] = (far_plane + near_plane) / (near_plane - far_plane)
        projection_matrix[2, 3] = (2 * far_plane * near_plane) / (near_plane - far_plane)
        projection_matrix[3, 2] = -1.0
        
        # 组合投影和视图矩阵
        combined_matrix = torch.matmul(projection_matrix, view_matrix)
        
        # 定义图像在yz平面上的四个角点 (3D坐标)
        half_width = width / 2
        half_height = height / 2
        
        corners_3d = torch.tensor([
            [0, -half_width, -half_height, 1],  # 左上
            [0, half_width, -half_height, 1],   # 右上
            [0, half_width, half_height, 1],    # 右下
            [0, -half_width, half_height, 1]    # 左下
        ], dtype=torch.float32, device=self.dev)
        
        # 通过视图和投影矩阵转换角点
        corners_projected = torch.matmul(combined_matrix, corners_3d.t()).t()
        
        # 透视除法
        corners_projected = corners_projected[:, :3] / corners_projected[:, 3:4]
        
        # 转换到图像坐标
        corners_2d = torch.zeros(4, 2, dtype=torch.float32, device=self.dev)
        corners_2d[:, 0] = (corners_projected[:, 0] + 1) * output_size[1] / 2  # x坐标
        corners_2d[:, 1] = (corners_projected[:, 1] + 1) * output_size[0] / 2  # y坐标
        
        # 原始图像角点
        original_corners = torch.tensor([
            [0, 0],                  # 左上
            [width - 1, 0],          # 右上
            [width - 1, height - 1], # 右下
            [0, height - 1]          # 左下
        ], dtype=torch.float32, device=self.dev)
        
        # 获取透视变换矩阵
        perspective_matrix = kornia.geometry.transform.get_perspective_transform(
            original_corners.unsqueeze(0).repeat(batch_size, 1, 1),
            corners_2d.unsqueeze(0).repeat(batch_size, 1, 1)
        )
        
        # 应用透视变换
        output = kornia.geometry.transform.warp_perspective(
            image, perspective_matrix, output_size,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        return output
    
    def render_on_backgrounds(self, patch, backgrounds, longitudes=None, latitudes=None, distance=None):
        """
        在多个背景上渲染补丁的多个透视视图，只对补丁进行透视变换，背景保持不变
        
        参数:
            patch: 输入补丁张量, 形状为 [C, H, W]
            backgrounds: 背景图像张量, 形状为 [B, C, H, W]
            longitudes: 经度角度列表，范围[-90, 90]度，控制水平视角
            latitudes: 纬度角度列表，范围[-90, 90]度，控制垂直视角
            distance: 相机距离，如果为None则使用默认值
                
        返回:
            rendered_views: 渲染后的视图 [V, B, C, H, W] 其中V是视角数
            longitude_angles: 对应的经度角度 [V]
            latitude_angles: 对应的纬度角度 [V]
        """
        # 确保patch是3D张量 [C, H, W]
        if patch.dim() == 4 and patch.size(0) == 1:
            patch = patch.squeeze(0)
        
        assert patch.dim() == 3, f"补丁张量应为3D [C,H,W]，但当前是{patch.dim()}D"
        
        # 获取形状信息
        batch_size, bg_channels, bg_height, bg_width = backgrounds.shape
        patch_channels, patch_height, patch_width = patch.shape
        
        # 确保通道数匹配
        assert bg_channels == patch_channels, "背景和补丁的通道数必须匹配"
        
        # 使用默认值如果参数未指定
        if longitudes is None:
            longitudes = [-30, 0, 30]  # 默认三个水平视角
        if latitudes is None:
            latitudes = [0]  # 默认只有水平视角
        if distance is None:
            distance = self.default_distance
            
        # 计算视角总数
        num_views = len(longitudes) * len(latitudes)
        
        # 准备存储所有视角的张量
        all_views = []
        all_longitudes = []
        all_latitudes = []
        
        # 创建mask用于指示patch区域（值为1）和背景区域（值为0）
        patch_mask = torch.zeros(1, 1, patch_height, patch_width, device=self.dev)
        patch_mask.fill_(1.0)  # 全1的mask表示补丁区域
        
        # 将patch扩展到[1, C, H, W]
        patch_4d = patch.unsqueeze(0)
        
        # 对每个角度进行处理
        for lat in latitudes:
            for lon in longitudes:
                # 将角度转换为弧度
                longitude_rad = np.radians(lon)
                latitude_rad = np.radians(lat)
                
                # 图像中心点在3D空间的坐标
                center_3d = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.dev)
                
                # 相机在3D空间的坐标 (假设图像位于yz平面)
                # 使用球坐标系: x = d*cos(lat)*cos(lon), y = d*cos(lat)*sin(lon), z = d*sin(lat)
                camera_position = torch.tensor([
                    distance * np.cos(latitude_rad) * np.cos(longitude_rad),
                    distance * np.cos(latitude_rad) * np.sin(longitude_rad),
                    distance * np.sin(latitude_rad)
                ], dtype=torch.float32, device=self.dev)
                
                # 相机的正上方向（一般选择z轴正方向）
                up_vector = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.dev)
                
                # 相机的方向向量 (从相机指向图像中心点)
                direction_vector = center_3d - camera_position
                direction_vector = direction_vector / torch.norm(direction_vector)
                
                # 相机的右向量 (cross product of direction and up)
                right_vector = torch.cross(direction_vector, up_vector)
                right_vector = right_vector / torch.norm(right_vector)
                
                # 相机的真正的上向量 (cross product of right and direction)
                true_up_vector = torch.cross(right_vector, direction_vector)
                true_up_vector = true_up_vector / torch.norm(true_up_vector)
                
                # 构建视图矩阵
                view_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.dev)
                view_matrix[0, :3] = right_vector
                view_matrix[1, :3] = true_up_vector
                view_matrix[2, :3] = -direction_vector
                view_matrix[0, 3] = -torch.dot(right_vector, camera_position)
                view_matrix[1, 3] = -torch.dot(true_up_vector, camera_position)
                view_matrix[2, 3] = torch.dot(direction_vector, camera_position)
                view_matrix[3, 3] = 1.0
                
                # 构建投影矩阵 (透视投影)
                aspect_ratio = patch_width / patch_height
                near_plane = 0.1
                far_plane = distance * 2
                
                fov_rad = np.radians(self.fov)
                f = 1.0 / np.tan(fov_rad / 2)
                
                projection_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.dev)
                projection_matrix[0, 0] = f / aspect_ratio
                projection_matrix[1, 1] = f
                projection_matrix[2, 2] = (far_plane + near_plane) / (near_plane - far_plane)
                projection_matrix[2, 3] = (2 * far_plane * near_plane) / (near_plane - far_plane)
                projection_matrix[3, 2] = -1.0
                
                # 组合投影和视图矩阵
                combined_matrix = torch.matmul(projection_matrix, view_matrix)
                
                # 定义补丁在yz平面上的四个角点 (3D坐标)
                half_width = patch_width / 2
                half_height = patch_height / 2
                
                corners_3d = torch.tensor([
                    [0, -half_width, -half_height, 1],  # 左上
                    [0, half_width, -half_height, 1],   # 右上
                    [0, half_width, half_height, 1],    # 右下
                    [0, -half_width, half_height, 1]    # 左下
                ], dtype=torch.float32, device=self.dev)
                
                # 通过视图和投影矩阵转换角点
                corners_projected = torch.matmul(combined_matrix, corners_3d.t()).t()
                
                # 透视除法
                corners_projected = corners_projected[:, :3] / corners_projected[:, 3:4]
                
                # 转换到输出尺寸的图像坐标
                corners_2d = torch.zeros(4, 2, dtype=torch.float32, device=self.dev)
                corners_2d[:, 0] = (corners_projected[:, 0] + 1) * patch_width / 2  # x坐标
                corners_2d[:, 1] = (corners_projected[:, 1] + 1) * patch_height / 2  # y坐标
                
                # 原始补丁角点
                original_corners = torch.tensor([
                    [0, 0],                  # 左上
                    [patch_width-1, 0],          # 右上
                    [patch_width-1, patch_height-1], # 右下
                    [0, patch_height-1]          # 左下
                ], dtype=torch.float32, device=self.dev)
                
                # 获取透视变换矩阵
                perspective_matrix = kornia.geometry.transform.get_perspective_transform(
                    original_corners.unsqueeze(0),
                    corners_2d.unsqueeze(0)
                )
                
                # 对补丁应用透视变换
                warped_patch = kornia.geometry.transform.warp_perspective(
                    patch_4d, 
                    perspective_matrix, 
                    (patch_height, patch_width),
                    mode='bilinear', 
                    padding_mode='zeros', 
                    align_corners=True
                )
                
                # 对mask应用同样的透视变换，以获取变换后的区域
                warped_mask = kornia.geometry.transform.warp_perspective(
                    patch_mask, 
                    perspective_matrix, 
                    (patch_height, patch_width),
                    mode='bilinear', 
                    padding_mode='zeros', 
                    align_corners=True
                )
                
                # 对每个背景应用变换后的补丁
                batch_views = []
                for b in range(batch_size):
                    # 创建当前背景的副本
                    current_bg = backgrounds[b:b+1].clone()
                    
                    # 计算补丁在背景中的位置（居中）
                    y_offset = (bg_height - patch_height) // 2
                    x_offset = (bg_width - patch_width) // 2
                    
                    # 提取背景上对应的区域
                    bg_region = torch.zeros_like(warped_patch)
                    bg_region[:, :, :, :] = current_bg[:, :, y_offset:y_offset+patch_height, x_offset:x_offset+patch_width]
                    
                    # 使用mask合成：背景 * (1-mask) + 补丁 * mask
                    composited_region = bg_region * (1 - warped_mask) + warped_patch * warped_mask
                    
                    # 将合成区域放回背景
                    result = current_bg.clone()
                    result[:, :, y_offset:y_offset+patch_height, x_offset:x_offset+patch_width] = composited_region
                    
                    batch_views.append(result)
                
                # 堆叠当前角度的所有背景视图
                batch_views_tensor = torch.cat(batch_views, dim=0)  # [B, C, H, W]
                all_views.append(batch_views_tensor)
                all_longitudes.append(lon)
                all_latitudes.append(lat)
        
        # 堆叠所有视图 [V, B, C, H, W]
        rendered_views = torch.stack(all_views, dim=0)
        longitude_angles = torch.tensor(all_longitudes, device=self.dev)
        latitude_angles = torch.tensor(all_latitudes, device=self.dev)
        
        return rendered_views, longitude_angles, latitude_angles
    
    
# class PerspectiveViewGenerator:
#     """
#     透视视图生成器，用于从2D图像生成多角度视图
#     适用于训练角度鲁棒性补丁
#     """
    
#     def __init__(self, dev, image_size=512, fov=60.0, distance=5.0):
#         """
#         初始化透视视图生成器
        
#         参数:
#             dev: 运算设备 (CPU或CUDA)
#             image_size: 输出图像尺寸
#             fov: 视场角度 (degrees)
#             distance: 相机与图像的默认距离
#         """
#         self.dev = dev
#         self.image_size = image_size
#         self.fov = fov
#         self.default_distance = distance
    
#     def __call__(self, image, longitudes=None, latitudes=None, distance=None):
#         """
#         为给定图像生成多个视角的渲染图
        
#         参数:
#             image: 输入图像张量, 形状为 [B, C, H, W]
#             longitudes: 经度角度列表，范围[-90, 90]度，控制水平视角
#             latitudes: 纬度角度列表，范围[-90, 90]度，控制垂直视角
#             distance: 相机距离，如果为None则使用默认值
            
#         返回:
#             rendered_views: 渲染后的视图 [V, B, C, H, W] 其中V是视角数
#             longitude_angles: 对应的经度角度 [V]
#             latitude_angles: 对应的纬度角度 [V]
#         """
#         # 确保图像是4D张量 [B, C, H, W]
#         assert image.dim() == 4, f"图像张量应为4D [B,C,H,W]，但当前是{image.dim()}D"
        
#         batch_size, channels, height, width = image.shape
        
#         # 使用默认值如果参数未指定
#         if longitudes is None:
#             longitudes = [-30, 0, 30]  # 默认三个水平视角
#         if latitudes is None:
#             latitudes = [0]  # 默认只有水平视角
#         if distance is None:
#             distance = self.default_distance
            
#         # 计算视角总数
#         num_views = len(longitudes) * len(latitudes)
        
#         # 准备存储所有视角的张量
#         all_views = []
#         all_longitudes = []
#         all_latitudes = []
        
#         for lat in latitudes:
#             for lon in longitudes:
#                 # 为每个角度创建视图
#                 view = self.create_perspective_view(
#                     image, 
#                     longitude=lon, 
#                     latitude=lat, 
#                     distance=distance,
#                     output_size=(self.image_size, self.image_size)
#                 )
#                 # view形状: [B, C, H, W]
#                 all_views.append(view)
#                 all_longitudes.append(lon)
#                 all_latitudes.append(lat)
        
#         # 堆叠所有视图 [V, B, C, H, W]
#         rendered_views = torch.stack(all_views, dim=0)
#         longitude_angles = torch.tensor(all_longitudes, device=self.dev)
#         latitude_angles = torch.tensor(all_latitudes, device=self.dev)
        
#         return rendered_views, longitude_angles, latitude_angles

#     def render_on_backgrounds(self, patch, backgrounds, longitudes=None, latitudes=None, distance=None):
#         """
#         在多个背景上渲染补丁的多个透视视图
        
#         参数:
#             patch: 输入补丁张量, 形状为 [C, H, W]
#             backgrounds: 背景图像张量, 形状为 [B, C, H, W]
#             longitudes: 经度角度列表，范围[-90, 90]度，控制水平视角
#             latitudes: 纬度角度列表，范围[-90, 90]度，控制垂直视角
#             distance: 相机距离，如果为None则使用默认值
                
#         返回:
#             rendered_views: 渲染后的视图 [V, B, C, H, W] 其中V是视角数
#             longitude_angles: 对应的经度角度 [V]
#             latitude_angles: 对应的纬度角度 [V]
#         """
#         # 确保patch是3D张量 [C, H, W]
#         if patch.dim() == 4 and patch.size(0) == 1:
#             patch = patch.squeeze(0)
        
#         assert patch.dim() == 3, f"补丁张量应为3D [C,H,W]，但当前是{patch.dim()}D"
        
#         batch_size, channels, height, width = backgrounds.shape
        
#         # 使用默认值如果参数未指定
#         if longitudes is None:
#             longitudes = [-30, 0, 30]  # 默认三个水平视角
#         if latitudes is None:
#             latitudes = [0]  # 默认只有水平视角
#         if distance is None:
#             distance = self.default_distance
            
#         # 计算视角总数
#         num_views = len(longitudes) * len(latitudes)
        
#         # 准备存储所有视角的张量
#         all_views = []
#         all_longitudes = []
#         all_latitudes = []
        
#         for lat in latitudes:
#             for lon in longitudes:
#                 # 为每个角度创建视图
#                 view = self.create_perspective_view_with_background(
#                     patch, 
#                     backgrounds, 
#                     longitude=lon, 
#                     latitude=lat, 
#                     distance=distance
#                 )
#                 # view形状: [B, C, H, W]
#                 all_views.append(view)
#                 all_longitudes.append(lon)
#                 all_latitudes.append(lat)
        
#         # 堆叠所有视图 [V, B, C, H, W]
#         rendered_views = torch.stack(all_views, dim=0)
#         longitude_angles = torch.tensor(all_longitudes, device=self.dev)
#         latitude_angles = torch.tensor(all_latitudes, device=self.dev)
        
#         return rendered_views, longitude_angles, latitude_angles
    
#     def create_perspective_view(self, image, longitude, latitude, distance, output_size=None):
#         """
#         创建给定2D图像的透视视图
        
#         参数:
#             image: 输入图像张量, 形状为 [B, C, H, W]
#             longitude: 经度，范围[-90, 90]度，控制水平方向的视角
#             latitude: 纬度，范围[-90, 90]度，控制垂直方向的视角
#             distance: 相机与图像的距离
#             output_size: 输出图像尺寸, 默认使用原图尺寸
        
#         返回:
#             透视变换后的图像 [B, C, H, W]
#         """
#         batch_size, _, height, width = image.shape
        
#         # 如果没有指定输出尺寸，使用原图尺寸
#         if output_size is None:
#             output_size = (height, width)
        
#         # 将角度转换为弧度
#         longitude_rad = np.radians(longitude)
#         latitude_rad = np.radians(latitude)
        
#         # 图像中心点在3D空间的坐标
#         center_3d = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.dev)
        
#         # 相机在3D空间的坐标 (假设图像位于yz平面)
#         # 使用球坐标系: x = d*cos(lat)*cos(lon), y = d*cos(lat)*sin(lon), z = d*sin(lat)
#         camera_position = torch.tensor([
#             distance * np.cos(latitude_rad) * np.cos(longitude_rad),
#             distance * np.cos(latitude_rad) * np.sin(longitude_rad),
#             distance * np.sin(latitude_rad)
#         ], dtype=torch.float32, device=self.dev)
        
#         # 相机的正上方向（一般选择z轴正方向）
#         up_vector = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.dev)
        
#         # 相机的方向向量 (从相机指向图像中心点)
#         direction_vector = center_3d - camera_position
#         direction_vector = direction_vector / torch.norm(direction_vector)
        
#         # 相机的右向量 (cross product of direction and up)
#         right_vector = torch.cross(direction_vector, up_vector)
#         right_vector = right_vector / torch.norm(right_vector)
        
#         # 相机的真正的上向量 (cross product of right and direction)
#         true_up_vector = torch.cross(right_vector, direction_vector)
#         true_up_vector = true_up_vector / torch.norm(true_up_vector)
        
#         # 构建视图矩阵
#         view_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.dev)
#         view_matrix[0, :3] = right_vector
#         view_matrix[1, :3] = true_up_vector
#         view_matrix[2, :3] = -direction_vector
#         view_matrix[0, 3] = -torch.dot(right_vector, camera_position)
#         view_matrix[1, 3] = -torch.dot(true_up_vector, camera_position)
#         view_matrix[2, 3] = torch.dot(direction_vector, camera_position)
#         view_matrix[3, 3] = 1.0
        
#         # 构建投影矩阵 (透视投影)
#         aspect_ratio = width / height
#         near_plane = 0.1
#         far_plane = distance * 2
        
#         fov_rad = np.radians(self.fov)
#         f = 1.0 / np.tan(fov_rad / 2)
        
#         projection_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.dev)
#         projection_matrix[0, 0] = f / aspect_ratio
#         projection_matrix[1, 1] = f
#         projection_matrix[2, 2] = (far_plane + near_plane) / (near_plane - far_plane)
#         projection_matrix[2, 3] = (2 * far_plane * near_plane) / (near_plane - far_plane)
#         projection_matrix[3, 2] = -1.0
        
#         # 组合投影和视图矩阵
#         combined_matrix = torch.matmul(projection_matrix, view_matrix)
        
#         # 定义图像在yz平面上的四个角点 (3D坐标)
#         half_width = width / 2
#         half_height = height / 2
        
#         corners_3d = torch.tensor([
#             [0, -half_width, -half_height, 1],  # 左上
#             [0, half_width, -half_height, 1],   # 右上
#             [0, half_width, half_height, 1],    # 右下
#             [0, -half_width, half_height, 1]    # 左下
#         ], dtype=torch.float32, device=self.dev)
        
#         # 通过视图和投影矩阵转换角点
#         corners_projected = torch.matmul(combined_matrix, corners_3d.t()).t()
        
#         # 透视除法
#         corners_projected = corners_projected[:, :3] / corners_projected[:, 3:4]
        
#         # 转换到图像坐标
#         corners_2d = torch.zeros(4, 2, dtype=torch.float32, device=self.dev)
#         corners_2d[:, 0] = (corners_projected[:, 0] + 1) * output_size[1] / 2  # x坐标
#         corners_2d[:, 1] = (corners_projected[:, 1] + 1) * output_size[0] / 2  # y坐标
        
#         # 原始图像角点
#         original_corners = torch.tensor([
#             [0, 0],                  # 左上
#             [width - 1, 0],          # 右上
#             [width - 1, height - 1], # 右下
#             [0, height - 1]          # 左下
#         ], dtype=torch.float32, device=self.dev)
        
#         # 获取透视变换矩阵
#         perspective_matrix = self.get_perspective_transform(
#             original_corners.unsqueeze(0).repeat(batch_size, 1, 1),
#             corners_2d.unsqueeze(0).repeat(batch_size, 1, 1)
#         )
        
#         # 应用透视变换
#         output = self.warp_perspective(
#             image, perspective_matrix, output_size,
#             mode='bilinear', padding_mode='zeros', align_corners=True
#         )
        
#         return output
    
#     def create_perspective_view_with_background(self, patch, background, longitude, latitude, distance, output_size=None):
#         """
#         创建给定补丁的透视视图，并直接将其放置在背景图像上
        
#         参数:
#             patch: 输入补丁张量, 形状为 [C, H, W]
#             background: 背景图像张量, 形状为 [B, C, H, W]
#             longitude: 经度，范围[-90, 90]度，控制水平方向的视角
#             latitude: 纬度，范围[-90, 90]度，控制垂直方向的视角
#             distance: 相机与图像的距离
#             output_size: 输出图像尺寸, 默认使用背景图像尺寸
        
#         返回:
#             合成后的图像 [B, C, H, W]
#         """
#         # 确保patch是3D张量 [C, H, W]
#         if patch.dim() == 4 and patch.size(0) == 1:
#             patch = patch.squeeze(0)
        
#         assert patch.dim() == 3, f"补丁张量应为3D [C,H,W]，但当前是{patch.dim()}D"
        
#         # 获取背景形状信息
#         batch_size, bg_channels, bg_height, bg_width = background.shape
#         patch_channels, patch_height, patch_width = patch.shape
        
#         # 确保通道数匹配
#         assert bg_channels == patch_channels, "背景和补丁的通道数必须匹配"
        
#         # 如果没有指定输出尺寸，使用背景尺寸
#         if output_size is None:
#             output_size = (bg_height, bg_width)
        
#         # 将角度转换为弧度
#         longitude_rad = np.radians(longitude)
#         latitude_rad = np.radians(latitude)
        
#         # 图像中心点在3D空间的坐标
#         center_3d = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.dev)
        
#         # 相机在3D空间的坐标 (假设图像位于yz平面)
#         # 使用球坐标系: x = d*cos(lat)*cos(lon), y = d*cos(lat)*sin(lon), z = d*sin(lat)
#         camera_position = torch.tensor([
#             distance * np.cos(latitude_rad) * np.cos(longitude_rad),
#             distance * np.cos(latitude_rad) * np.sin(longitude_rad),
#             distance * np.sin(latitude_rad)
#         ], dtype=torch.float32, device=self.dev)
        
#         # 相机的正上方向（一般选择z轴正方向）
#         up_vector = torch.tensor([0, 0, 1], dtype=torch.float32, device=self.dev)
        
#         # 相机的方向向量 (从相机指向图像中心点)
#         direction_vector = center_3d - camera_position
#         direction_vector = direction_vector / torch.norm(direction_vector)
        
#         # 相机的右向量 (cross product of direction and up)
#         right_vector = torch.cross(direction_vector, up_vector)
#         right_vector = right_vector / torch.norm(right_vector)
        
#         # 相机的真正的上向量 (cross product of right and direction)
#         true_up_vector = torch.cross(right_vector, direction_vector)
#         true_up_vector = true_up_vector / torch.norm(true_up_vector)
        
#         # 构建视图矩阵
#         view_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.dev)
#         view_matrix[0, :3] = right_vector
#         view_matrix[1, :3] = true_up_vector
#         view_matrix[2, :3] = -direction_vector
#         view_matrix[0, 3] = -torch.dot(right_vector, camera_position)
#         view_matrix[1, 3] = -torch.dot(true_up_vector, camera_position)
#         view_matrix[2, 3] = torch.dot(direction_vector, camera_position)
#         view_matrix[3, 3] = 1.0
        
#         # 构建投影矩阵 (透视投影)
#         aspect_ratio = patch_width / patch_height
#         near_plane = 0.1
#         far_plane = distance * 2
        
#         fov_rad = np.radians(self.fov)
#         f = 1.0 / np.tan(fov_rad / 2)
        
#         projection_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.dev)
#         projection_matrix[0, 0] = f / aspect_ratio
#         projection_matrix[1, 1] = f
#         projection_matrix[2, 2] = (far_plane + near_plane) / (near_plane - far_plane)
#         projection_matrix[2, 3] = (2 * far_plane * near_plane) / (near_plane - far_plane)
#         projection_matrix[3, 2] = -1.0
        
#         # 组合投影和视图矩阵
#         combined_matrix = torch.matmul(projection_matrix, view_matrix)
        
#         # 定义补丁在yz平面上的四个角点 (3D坐标)
#         half_width = patch_width / 2
#         half_height = patch_height / 2
        
#         corners_3d = torch.tensor([
#             [0, -half_width, -half_height, 1],  # 左上
#             [0, half_width, -half_height, 1],   # 右上
#             [0, half_width, half_height, 1],    # 右下
#             [0, -half_width, half_height, 1]    # 左下
#         ], dtype=torch.float32, device=self.dev)
        
#         # 通过视图和投影矩阵转换角点
#         corners_projected = torch.matmul(combined_matrix, corners_3d.t()).t()
        
#         # 透视除法
#         corners_projected = corners_projected[:, :3] / corners_projected[:, 3:4]
        
#         # 转换到输出尺寸的图像坐标
#         corners_2d = torch.zeros(4, 2, dtype=torch.float32, device=self.dev)
#         corners_2d[:, 0] = (corners_projected[:, 0] + 1) * output_size[1] / 2  # x坐标
#         corners_2d[:, 1] = (corners_projected[:, 1] + 1) * output_size[0] / 2  # y坐标
        
#         # 计算补丁应该放置在背景上的位置
#         # 我们假设补丁应该放在背景中央
#         y_offset = (bg_height - patch_height) // 2
#         x_offset = (bg_width - patch_width) // 2
        
#         # 补丁在背景上的角点
#         patch_corners = torch.tensor([
#             [x_offset, y_offset],                       # 左上
#             [x_offset + patch_width - 1, y_offset],     # 右上
#             [x_offset + patch_width - 1, y_offset + patch_height - 1],  # 右下
#             [x_offset, y_offset + patch_height - 1]      # 左下
#         ], dtype=torch.float32, device=self.dev)
        
#         # 获取透视变换矩阵
#         perspective_matrix = self.get_perspective_transform(
#             patch_corners.unsqueeze(0).repeat(batch_size, 1, 1),
#             corners_2d.unsqueeze(0).repeat(batch_size, 1, 1)
#         )
        
#         # 创建补丁掩码 - 一个全1的矩阵，形状与补丁相同
#         # 扩展为与背景相同的形状
#         patch_mask = torch.zeros(batch_size, 1, bg_height, bg_width, device=self.dev)
#         for b in range(batch_size):
#             patch_mask[b, :, y_offset:y_offset+patch_height, x_offset:x_offset+patch_width] = 1.0
        
#         # 将补丁放置在背景上，作为输入图像
#         input_images = background.clone()
#         for b in range(batch_size):
#             input_images[b, :, y_offset:y_offset+patch_height, x_offset:x_offset+patch_width] = patch
        
#         # 应用透视变换到图像
#         warped_images = self.warp_perspective(
#             input_images, perspective_matrix, output_size,
#             mode='bilinear', padding_mode='zeros', align_corners=True
#         )
        
#         # 应用透视变换到掩码
#         warped_masks = self.warp_perspective(
#             patch_mask, perspective_matrix, output_size,
#             mode='bilinear', padding_mode='zeros', align_corners=True
#         )
        
#         # 将变换后的补丁与背景合成
#         # 我们只在掩码区域使用变换后的图像，其余使用原始背景
#         composite_images = background * (1.0 - warped_masks) + warped_images * warped_masks
        
#         return composite_images

#     def get_perspective_transform(self, src, dst):
#         """
#         计算透视变换矩阵
        
#         参数:
#             src: 源点，形状为 [B, 4, 2]
#             dst: 目标点，形状为 [B, 4, 2]
            
#         返回:
#             透视变换矩阵，形状为 [B, 3, 3]
#         """
#         # 验证输入形状
#         if not (src.shape[1:] == (4, 2) and dst.shape[1:] == (4, 2)):
#             raise ValueError(f"源点和目标点必须为形状[B, 4, 2]的张量，但得到src: {src.shape}, dst: {dst.shape}")
        
#         batch_size = src.shape[0]
#         device = src.device
#         dtype = src.dtype
        
#         # 为每批创建变换矩阵
#         M = torch.zeros((batch_size, 3, 3), device=device, dtype=dtype)
        
#         for b in range(batch_size):
#             # 获取当前批的源点和目标点
#             src_pts = src[b]
#             dst_pts = dst[b]
            
#             # 构建系数矩阵
#             A = torch.zeros((8, 8), device=device, dtype=dtype)
#             b_vec = torch.zeros(8, device=device, dtype=dtype)
            
#             for i in range(4):
#                 x, y = src_pts[i]
#                 u, v = dst_pts[i]
                
#                 A[i*2, 0] = x
#                 A[i*2, 1] = y
#                 A[i*2, 2] = 1
#                 A[i*2, 6] = -x * u
#                 A[i*2, 7] = -y * u
#                 b_vec[i*2] = u
                
#                 A[i*2+1, 3] = x
#                 A[i*2+1, 4] = y
#                 A[i*2+1, 5] = 1
#                 A[i*2+1, 6] = -x * v
#                 A[i*2+1, 7] = -y * v
#                 b_vec[i*2+1] = v
            
#             # 求解线性系统 Ax = b
#             try:
#                 x = torch.linalg.solve(A, b_vec)
                
#                 # 填充变换矩阵
#                 M[b, 0, 0] = x[0]
#                 M[b, 0, 1] = x[1]
#                 M[b, 0, 2] = x[2]
#                 M[b, 1, 0] = x[3]
#                 M[b, 1, 1] = x[4]
#                 M[b, 1, 2] = x[5]
#                 M[b, 2, 0] = x[6]
#                 M[b, 2, 1] = x[7]
#                 M[b, 2, 2] = 1.0
#             except:
#                 # 如果求解失败，使用单位矩阵
#                 M[b] = torch.eye(3, device=device, dtype=dtype)
#                 logger.warning(f"计算透视变换矩阵失败，使用单位矩阵")
                
#         return M

#     def warp_perspective(self, src, M, dsize, mode='bilinear', padding_mode='zeros', align_corners=True):
#         """
#         应用透视变换到图像
        
#         参数:
#             src: 输入图像，形状为 [B, C, H, W]
#             M: 变换矩阵，形状为 [B, 3, 3]
#             dsize: 输出尺寸 (height, width)
#             mode: 插值模式，'bilinear'或'nearest'
#             padding_mode: 填充模式，'zeros'、'border'或'reflection'
#             align_corners: 是否对齐角点
            
#         返回:
#             变换后的图像，形状为 [B, C, H, W]
#         """
#         # 使用torch.nn.functional的grid_sample功能实现
#         batch_size, _, src_height, src_width = src.shape
#         dst_height, dst_width = dsize
        
#         # 创建归一化网格坐标
#         y = torch.arange(0, dst_height, device=src.device, dtype=src.dtype)
#         x = torch.arange(0, dst_width, device=src.device, dtype=src.dtype)
        
#         if align_corners:
#             y = 2 * y / (dst_height - 1) - 1 if dst_height > 1 else y
#             x = 2 * x / (dst_width - 1) - 1 if dst_width > 1 else x
#         else:
#             y = 2 * (y + 0.5) / dst_height - 1
#             x = 2 * (x + 0.5) / dst_width - 1
        
#         # 创建网格
#         grid_y, grid_x = torch.meshgrid(y, x)
#         grid = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1)
#         grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [B, H, W, 3]
        
#         # 应用逆变换
#         # M的形状为 [B, 3, 3]
#         # grid的形状为 [B, H, W, 3]
#         # M_inv @ grid 的形状为 [B, H, W, 3]
#         M_inv = torch.inverse(M)
#         transformed = torch.matmul(grid.view(batch_size, -1, 3), M_inv.transpose(1, 2))
#         transformed = transformed.view(batch_size, dst_height, dst_width, 3)
        
#         # 透视除法
#         transformed = transformed[..., :2] / transformed[..., 2:3].clamp(min=1e-6)
        
#         # 转换到归一化坐标 [-1, 1]
#         if align_corners:
#             transformed[..., 0] = 2 * transformed[..., 0] / (src_width - 1) - 1 if src_width > 1 else transformed[..., 0]
#             transformed[..., 1] = 2 * transformed[..., 1] / (src_height - 1) - 1 if src_height > 1 else transformed[..., 1]
#         else:
#             transformed[..., 0] = 2 * transformed[..., 0] / src_width - 1
#             transformed[..., 1] = 2 * transformed[..., 1] / src_height - 1
        
#         # 执行采样
#         return F.grid_sample(src, transformed, mode=mode, padding_mode=padding_mode, align_corners=align_corners)