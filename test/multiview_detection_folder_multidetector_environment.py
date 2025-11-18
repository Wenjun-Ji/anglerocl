#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Detector Multiview Detection Script with Environment Compositing and AASR Analysis

This script:
1. Takes a folder containing texture images as input
2. For each texture image in the folder:
   a. Renders images from different angles using PerspectiveViewGenerator
   b. Composites the rendered images onto a specified background at the center
   c. Detects specified targets using selected detector (YOLOv3/YOLOv5/YOLOv10/Faster R-CNN/DETR)
   d. Saves detection confidence scores to separate CSV files
   e. Calculates AASR metrics (Angular Attack Success Rate)
   f. Creates confidence-angle curve plots
   g. Optionally saves all rendered images
3. Aggregates results for all textures
"""

import sys
import os
# Get absolute path of project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add project root to Python path
sys.path.insert(0, project_root)

import csv
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import math
from tqdm import tqdm
import glob
import time
import concurrent.futures
import torch.nn.functional as F
from utils.general import non_max_suppression

# MMDetection imports
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules

# Ultralytics imports  
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Warning: Ultralytics not available. YOLOv10 will not work.")

# Import custom PerspectiveViewGenerator
from adv_patch_gen.utils.patch import PerspectiveViewGenerator

from PIL import Image, ImageEnhance

# Import AASR calculation function
try:
    from metrics import calculate_AASR
    AASR_AVAILABLE = True
except ImportError:
    AASR_AVAILABLE = False
    print("Warning: metrics.py not found. AASR calculation will be disabled.")

class MMDetectionDetector:
    def __init__(self, config_file, checkpoint_file, device='cuda:0', target_class_id=11):
        """
        Initialize MMDetection detector
        
        Args:
            config_file: Path to model configuration file
            checkpoint_file: Path to model weights file
            device: Device to run on
            target_class_id: Target detection class ID
        """
        self.model = init_detector(config_file, checkpoint_file, device=device)
        self.device = device
        self.target_class_id = target_class_id
        
    def detect_stop_sign(self, image):
        """
        Detect target object in a single image
        
        Args:
            image: numpy array
            
        Returns:
            float: Maximum confidence score
        """
        # Call directly without additional preprocessing
        result = inference_detector(self.model, image)
        
        pred_instances = result.pred_instances
        target_mask = pred_instances.labels == self.target_class_id
        if target_mask.any():
            return float(pred_instances.scores[target_mask].max().item())
        return 0.0
    
    def batch_detect_stop_sign(self, images):
        """
        Batch detect multiple images using MMDetection's batch processing
        
        Args:
            images: List of numpy arrays
            
        Returns:
            list: List of confidence scores
        """
        # Pass list directly and let inference_detector handle batch processing
        results = inference_detector(self.model, images)
        
        confidences = []
        for result in results:
            pred_instances = result.pred_instances
            target_mask = pred_instances.labels == self.target_class_id
            if target_mask.any():
                confidences.append(float(pred_instances.scores[target_mask].max().item()))
            else:
                confidences.append(0.0)
        
        return confidences

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-detector Environment Multiview Detection with AASR')
    parser.add_argument('--texture-folder', 
                      type=str, 
                      default="/zhaotingfeng/jiwenjun/anglerocl/data/stable_diffusion_1.5/ndda/select/", 
                      help='Path to folder containing texture images')
    parser.add_argument('--texture-extensions', 
                      type=str, 
                      default="png,jpg,jpeg", 
                      help='Comma-separated list of image extensions to process')
    parser.add_argument('--background-image', 
                      type=str, 
                      default="/zhaotingfeng/jiwenjun/anglerocl/environment/back.jpg", 
                      help='Path to the background image')
    parser.add_argument('--background-size', 
                      type=int, 
                      nargs=2, 
                      default=[1600, 900], 
                      help='Background size (width, height)')
    parser.add_argument('--detector', 
                      type=str, 
                      choices=['yolov3', 'yolov5', 'yolov10', 'faster_rcnn', 'detr'],
                      default='yolov5',
                      help='Detector to use')
    parser.add_argument('--target-class-id',
                      type=int,
                      default=11,
                      help='Target class ID to detect (COCO format, default: 11 for stop sign)')
    parser.add_argument('--yolov5-model', 
                      type=str, 
                      default="/zhaotingfeng/jiwenjun/anglerocl/checkpoints/yolov5n.pt", 
                      help='Path to YOLOv5 weights (only for yolov5)')
    parser.add_argument('--yolov10-model',
                      type=str,
                      default='/zhaotingfeng/jiwenjun/anglerocl/checkpoints/yolov10n.pt',
                      help='YOLOv10 model variant')
    parser.add_argument('--mmdet-config',
                      type=str,
                      help='MMDetection config file path')
    parser.add_argument('--mmdet-checkpoint',
                      type=str,
                      help='MMDetection checkpoint file path')
    parser.add_argument('--output-dir', 
                      type=str, 
                      default="/zhaotingfeng/jiwenjun/anglerocl/multiview_output", 
                      help='Base directory to save outputs')
    parser.add_argument('--save-renders', 
                      action='store_true',
                      default=False, 
                      help='Whether to save all rendered images')
    parser.add_argument('--azim-step', 
                      type=int, 
                      default=1, 
                      help='Step size for azimuth angle (degrees)')
    parser.add_argument('--renderer-fov',
                      type=float,
                      default=60.0,
                      help='Field of view for the perspective renderer')
    parser.add_argument('--renderer-distance',
                      type=float,
                      default=1500,
                      help='Distance from camera to the image plane')
    parser.add_argument('--resolution',
                      type=int,
                      default=512,
                      help='Resolution of the rendered images')
    parser.add_argument('--num-workers', 
                      type=int, 
                      default=1, 
                      help='Number of parallel workers for processing files')
    parser.add_argument('--summary-output', 
                      type=str, 
                      default="summary_results.csv", 
                      help='Filename for summary results across all textures')
    parser.add_argument('--batch-size',
                      type=int,
                      default=178,
                      help='Batch size for MMDetection models')
    
    # AASR related parameters
    parser.add_argument('--enable-aasr',
                      action='store_true',
                      default=True,
                      help='Enable AASR calculation (default: True)')
    parser.add_argument('--aasr-thresholds',
                      type=str,
                      default="0.25,0.5,0.75",
                      help='AASR confidence thresholds, comma-separated (default: 0.25,0.5,0.75)')
    parser.add_argument('--aasr-min-angle',
                      type=float,
                      default=-89.0,
                      help='Minimum angle for AASR calculation (default: -89.0)')
    parser.add_argument('--aasr-max-angle',
                      type=float,
                      default=89.0,
                      help='Maximum angle for AASR calculation (default: 89.0)')
    
    return parser.parse_args()

def find_texture_files(folder_path, extensions):
    """Find all texture files in the given folder with specified extensions"""
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path {folder_path} does not exist")
    
    ext_list = [e if e.startswith('.') else f'.{e}' for e in extensions.split(',')]
    
    texture_files = []
    for ext in ext_list:
        texture_files.extend(glob.glob(os.path.join(folder_path, f'*{ext}')))
    
    texture_files.sort()
    
    if not texture_files:
        raise ValueError(f"No files with extensions {extensions} found in {folder_path}")
    
    return texture_files

def setup_output_dirs(texture_path, output_dir):  
    """Set up output directories based on input texture path"""
    parts = texture_path.rstrip('/').split('/')
    
    if len(parts) >= 4:
        dir_path = os.path.join(parts[-4], parts[-3], parts[-2])
    elif len(parts) == 3:
        dir_path = os.path.join(parts[-3], parts[-2])
    elif len(parts) == 2:
        dir_path = parts[-2]
    else:
        dir_path = ""
    
    filename = os.path.splitext(os.path.basename(texture_path))[0]
    
    output_path = os.path.join(output_dir, dir_path, filename)
    render_path = os.path.join(output_path, 'render')
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(render_path, exist_ok=True)
    
    return output_path, render_path

def load_texture_image(image_path, size=512, device="cuda"):
    """Load image and convert to tensor"""
    if image_path is None or not os.path.exists(image_path):
        raise ValueError(f"Image path {image_path} not found or not provided.")
    
    img = Image.open(image_path)
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_np = np.array(img) / 255.0
    texture = torch.tensor(img_np, dtype=torch.float32, device=device)
    texture = texture.permute(2, 0, 1).unsqueeze(0)
    
    return texture

def load_background_image(image_path, size=(1280, 720), device="cuda"):
    """
    Load background image and convert to tensor
    """
    if image_path is None:
        # If no background image provided, create solid color background (e.g., green grass)
        bg = np.ones((size[1], size[0], 3), dtype=np.float32)
        bg[:, :, 0] *= 0.2  # R
        bg[:, :, 1] *= 0.6  # G
        bg[:, :, 2] *= 0.3  # B
    else:
        # Read image
        bg = Image.open(image_path)
        
        # Resize
        bg = bg.resize((size[0], size[1]), Image.Resampling.LANCZOS)
        
        # Ensure RGB mode
        if bg.mode != 'RGB':
            bg = bg.convert('RGB')
        
        # Convert to NumPy array and normalize to [0,1]
        bg = np.array(bg) / 255.0
    
    # Convert to PyTorch tensor
    bg_tensor = torch.tensor(bg, dtype=torch.float32, device=device)
    
    # Adjust shape to [B,C,H,W]
    bg_tensor = bg_tensor.permute(2, 0, 1).unsqueeze(0)
    
    return bg_tensor

def composite_image(rendered_view, background, center_position=None):
    """
    Composite rendered view onto background image
    
    Args:
        rendered_view: Tensor of shape [C,H,W] representing rendered view
        background: Tensor of shape [C,H,W] representing background image
        center_position: Optional (x,y) tuple specifying placement position, defaults to background center
    
    Returns:
        Composited image tensor of shape [C,H,W]
    """
    # Get dimensions
    _, bg_h, bg_w = background.shape
    _, rv_h, rv_w = rendered_view.shape
    
    # If center position not specified, use background center
    if center_position is None:
        center_x = bg_w // 2
        center_y = bg_h // 2
    else:
        center_x, center_y = center_position
    
    # Calculate top-left corner coordinates for placement
    start_x = center_x - rv_w // 2
    start_y = center_y - rv_h // 2
    
    # Create copy of composite image
    composite = background.clone()
    
    # Ensure coordinates are within valid range
    if start_x < 0:
        start_x = 0
    if start_y < 0:
        start_y = 0
    if start_x + rv_w > bg_w:
        rv_w = bg_w - start_x
    if start_y + rv_h > bg_h:
        rv_h = bg_h - start_y
    
    # Create mask for rendered view (non-black pixels)
    # Average three channels to create intensity mask
    intensity = rendered_view.mean(dim=0)
    mask = (intensity > 0.0).float()
    mask = mask.unsqueeze(0).expand_as(rendered_view)
    
    # Place rendered view at corresponding position on background image
    for c in range(3):  # For each channel
        composite[c, start_y:start_y+rv_h, start_x:start_x+rv_w] = (
            mask[c, :rv_h, :rv_w] * rendered_view[c, :rv_h, :rv_w] + 
            (1 - mask[c, :rv_h, :rv_w]) * composite[c, start_y:start_y+rv_h, start_x:start_x+rv_w]
        )
    
    return composite

def save_rendered_image(rgb_image, save_path, title="", add_title=False):
    """Save rendered image to disk, optionally with title and border"""
    # If input is PyTorch tensor, convert to NumPy array
    if isinstance(rgb_image, torch.Tensor):
        rgb_image = rgb_image.detach().cpu().numpy()
        # If shape is [C,H,W], convert to [H,W,C]
        if rgb_image.shape[0] == 3:
            rgb_image = rgb_image.transpose(1, 2, 0)
    
    add_title = False
    if add_title:
        # Create image with title
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image)
        plt.title(title)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        # Save image directly without title and border
        # Ensure values are in 0-255 range
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)
        
        # Use PIL to save image directly
        img = Image.fromarray(rgb_image)
        img.save(save_path)

def load_detector(args, device):
    """Load detector based on arguments"""
    
    if args.detector == 'yolov5':
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=args.yolov5_model)
        model.classes = [args.target_class_id]  # Use specified target class
        model.to(device)
        return model
    
    elif args.detector == 'yolov10':
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics not installed. Please install it to use YOLOv10.")
        model = YOLO(args.yolov10_model)
        return model
    
    elif args.detector in ['yolov3', 'faster_rcnn', 'detr']:
        register_all_modules()  # Register MMDetection modules
        
        # Configuration file mapping
        config_map = {
            'yolov3': '/zhaotingfeng/jiwenjun/anglerocl/checkpoints/yolov3_d53_320_273e_coco.py',
            'faster_rcnn': '/zhaotingfeng/jiwenjun/anglerocl/checkpoints/faster-rcnn_r50_fpn_1x_coco.py',
            'detr': '/zhaotingfeng/jiwenjun/anglerocl/checkpoints/detr_r50_8xb2-150e_coco.py'
        }
        
        checkpoint_map = {
            'yolov3': '/zhaotingfeng/jiwenjun/anglerocl/checkpoints/yolov3_d53_320_273e_coco-421362b6.pth',
            'faster_rcnn': '/zhaotingfeng/jiwenjun/anglerocl/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
            'detr': '/zhaotingfeng/jiwenjun/anglerocl/checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'
        }
        
        config_file = args.mmdet_config or config_map[args.detector]
        checkpoint_file = args.mmdet_checkpoint or checkpoint_map[args.detector]
        
        return MMDetectionDetector(config_file, checkpoint_file, device, args.target_class_id)
    
    else:
        raise ValueError(f"Unknown detector: {args.detector}")

def convert_to_numpy(image):
    """Convert various image formats to numpy array"""
    if isinstance(image, torch.Tensor):
        if image.min() < 0 or image.max() > 1:
            image = (image + 1) / 2
        image = image.clamp(0, 1)
        
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.dim() == 3:
            image = image.permute(1, 2, 0)
        
        image_np = (image.detach().cpu().numpy() * 255).astype(np.uint8)
    elif isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image_np = (image * 255).astype(np.uint8)
        else:
            image_np = image.astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    return image_np

def detect_stop_sign_unified(model, image, detector_type, target_class_id):
    """Unified detection interface"""
    
    if detector_type == 'yolov5':
        # YOLOv5 uses PIL images
        if isinstance(image, torch.Tensor) or isinstance(image, np.ndarray):
            image_np = convert_to_numpy(image)
            img = Image.fromarray(image_np)
        else:
            img = image
                                          
        results = model(img)
        detections = results.pandas().xyxy[0]
        target_detections = detections[detections['class'] == target_class_id]
        return float(target_detections['confidence'].max()) if not target_detections.empty else 0.0
    
    elif detector_type == 'yolov10':
        # YOLOv10 uses PIL images
        if isinstance(image, torch.Tensor) or isinstance(image, np.ndarray):
            image_np = convert_to_numpy(image)
            img = Image.fromarray(image_np)
        else:
            img = image
        
        results = model(img)
        detections = results[0].boxes
        target_mask = detections.cls == target_class_id  # Use specified target class
        if target_mask.any():
            target_confs = detections.conf[target_mask]
            return float(target_confs.max().item())
        return 0.0
    
    elif detector_type in ['yolov3', 'faster_rcnn', 'detr']:
        # MMDetection uses numpy arrays
        image_np = convert_to_numpy(image)
        return model.detect_stop_sign(image_np)
    
    else:
        raise ValueError(f"Unknown detector type: {detector_type}")

def batch_detect_stop_sign_unified(model, images, detector_type, batch_size=128, target_class_id=11):
    """Unified interface for batch detection"""
        
    if detector_type in ['yolov3', 'faster_rcnn', 'detr']:
        # MMDetection batch processing
        images_np = [convert_to_numpy(img) for img in images]  
        # If batch_size is less than number of images, process in batches
        if len(images_np) <= batch_size:
            return model.batch_detect_stop_sign(images_np)
        else:
            # Process large batches in smaller chunks
            confidences = []
            for i in range(0, len(images_np), batch_size):
                batch = images_np[i:i+batch_size]
                batch_confs = model.batch_detect_stop_sign(batch)
                confidences.extend(batch_confs)
            return confidences
    else:
        # Other detectors still process individually
        confidences = []
        for image in images:
            conf = detect_stop_sign_unified(model, image, detector_type, target_class_id)
            confidences.append(conf)
        return confidences

def extract_prompt_from_path(path):
    """Extract prompt from directory path"""
    dir_name = os.path.basename(os.path.normpath(path))
    
    processed_text = ""
    i = 0
    while i < len(dir_name):
        if i + 1 < len(dir_name) and dir_name[i] == '_' and dir_name[i+1] == '_':
            processed_text += ' "'
            i += 2
        elif dir_name[i] == '_':
            processed_text += ' '
            i += 1
        else:
            processed_text += dir_name[i]
            i += 1
    
    if processed_text.count('"') % 2 == 1:
        processed_text += '"'
    
    return processed_text

def plot_confidence_curve(angles, confidences, plot_path, title=None):
    """Create and save a plot of angle vs. confidence"""
    if confidences:
        print(f"Confidence stats - Min: {min(confidences):.4f}, Max: {max(confidences):.4f}, Average: {sum(confidences)/len(confidences):.4f}")
    else:
        print("Warning: No confidence values to plot!")
        return
        
    plt.figure(figsize=(12, 6))
    plt.plot(angles, confidences, 'b-', linewidth=2)
    plt.plot(angles, confidences, 'ro', markersize=3)
    
    plt.xlabel('Azimuth Angle (degrees)')
    plt.ylabel('Detection Confidence')
    plt_title = 'Target Detection Confidence vs. Viewing Angle'
    if title:
        plt_title += f" - {title}"
    plt.title(plt_title)
    plt.grid(True)
    plt.xlim(-90, 90)
    plt.ylim(0, 1.05)
    
    plt.axvline(x=0, color='g', linestyle='--', label='Direct View (0°)')
    
    plt.axhline(y=0.25, color='r', linestyle=':', alpha=0.7, label='0.25 Confidence')
    plt.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='0.50 Confidence')
    plt.axhline(y=0.75, color='g', linestyle=':', alpha=0.7, label='0.75 Confidence')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Confidence curve saved to {plot_path}")

def calculate_aasr_for_texture(angles, confidences, thresholds, output_dir=None):
    """
    Calculate AASR metrics for a single texture, save as CSV format
    
    Parameters:
        angles: List of angles
        confidences: List of confidences
        thresholds: List of thresholds
        output_dir: Output directory (optional)
        
    Returns:
        Dictionary of AASR results
    """
    if not AASR_AVAILABLE:
        return None
    
    aasr_results = {}
    
    try:
        # Calculate AASR for each threshold
        for threshold in thresholds:
            aasr_results[f"AASR_{threshold}"] = calculate_AASR(angles, confidences, threshold)
        
        # Calculate average AASR
        avg_aasr = sum(aasr_results[f"AASR_{t}"]["AASR"] for t in thresholds) / len(thresholds)
        aasr_results["AASR_average"] = float(avg_aasr)
        
        # If output directory provided, save results as CSV
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. Save summary AASR results to CSV
            metrics_path = os.path.join(output_dir, 'aasr_metrics.csv')
            with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
                csv_writer = csv.writer(f)
                
                # Write header
                csv_writer.writerow(['Metric', 'Value', 'Details'])
                
                # Write average AASR
                csv_writer.writerow(['Average_AASR', f"{aasr_results['AASR_average']:.4f}", ''])
                
                # Blank line separator
                csv_writer.writerow([])
                
                # Write detailed AASR info for each threshold
                csv_writer.writerow(['Threshold', 'AASR (%)', 'Successful_Attacks', 'Total_Angles'])
                
                for threshold in thresholds:
                    result = aasr_results[f"AASR_{threshold}"]
                    csv_writer.writerow([
                        threshold,
                        f"{result['AASR']:.4f}",
                        result['successful_attacks'],
                        result['total_angles']
                    ])
            
            # 2. Save detailed angle-level attack information (optional)
            # Use attack_details from first threshold as example
            if thresholds:
                first_threshold = thresholds[0]
                attack_details = aasr_results[f"AASR_{first_threshold}"].get('attack_details', [])
                
                if attack_details:
                    details_path = os.path.join(output_dir, 'aasr_attack_details.csv')
                    with open(details_path, 'w', newline='', encoding='utf-8') as f:
                        csv_writer = csv.writer(f)
                        
                        # Write header
                        csv_writer.writerow(['Angle', 'Confidence', 'Attack_Success'])
                        
                        # Write detailed info for each angle
                        for detail in attack_details:
                            csv_writer.writerow([
                                detail['angle'],
                                f"{detail['confidence']:.4f}",
                                detail['attack_success']
                            ])
            
            print(f"    AASR metrics saved to: {metrics_path}")
        
        return aasr_results
        
    except Exception as e:
        print(f"  Error during AASR calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def print_aasr_summary(aasr_results, texture_name=""):
    """Print AASR results summary"""
    if not aasr_results:
        return
    
    prefix = f"[{texture_name}] " if texture_name else ""
    
    try:
        print(f"\n{prefix}AASR Analysis Results:")
        print(f"  Average AASR: {aasr_results['AASR_average']:.2f}%")
        
        for key, value in aasr_results.items():
            if key.startswith('AASR_') and key != 'AASR_average':
                threshold = key.split('_')[1]
                aasr = value['AASR']
                success = value['successful_attacks']
                total = value['total_angles']
                print(f"  AASR({threshold}): {aasr:.2f}% ({success}/{total} angles)")
    except Exception as e:
        print(f"  Error printing AASR summary: {str(e)}")

def create_folder_aasr_summary(texture_results, output_path):
    """
    Create folder-level AASR summary CSV
    
    Parameters:
        texture_results: Dictionary of texture name -> AASR results
        output_path: Output CSV file path
    """
    if not texture_results:
        return
    
    try:
        # Get all thresholds
        first_texture = next(iter(texture_results.values()))
        thresholds = [key.split('_')[1] for key in first_texture.keys() 
                      if key.startswith('AASR_') and key != 'AASR_average']
        
        # Prepare data
        summary_data = []
        
        for texture_name, results in texture_results.items():
            row = {
                'Texture': texture_name,
                'Average_AASR': float(results.get('AASR_average', 0))
            }
            for threshold in thresholds:
                aasr_value = results.get(f'AASR_{threshold}', {}).get('AASR', 0)
                row[f'AASR_{threshold}'] = float(aasr_value)
            summary_data.append(row)
        
        # Add average row
        avg_row = {'Texture': 'Average'}
        avg_row['Average_AASR'] = np.mean([r['Average_AASR'] for r in summary_data])
        for threshold in thresholds:
            avg_row[f'AASR_{threshold}'] = np.mean([r[f'AASR_{threshold}'] for r in summary_data])
        summary_data.append(avg_row)
        
        # Save as CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if summary_data:
                # Get all field names
                fieldnames = list(summary_data[0].keys())
                csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                # Write header and data
                csv_writer.writeheader()
                for row in summary_data:
                    csv_writer.writerow(row)
        
        print(f"  Folder AASR summary saved to: {output_path}")
        
    except Exception as e:
        print(f"  Error creating folder AASR summary: {str(e)}")
        import traceback
        traceback.print_exc()

def process_texture_file(texture_path, model, background_tensor, output_dir, args, device):
    """Process a single texture file and return its results"""
    try:
        output_path, render_dir = setup_output_dirs(texture_path, output_dir)
        
        prompt_dir = os.path.dirname(texture_path)
        prompt = extract_prompt_from_path(os.path.basename(prompt_dir))
        
        print(f"\nProcessing texture: {texture_path}")
        print(f"  Output directory: {output_path}")
        
        texture_tensor = load_texture_image(
            texture_path, 
            size=args.resolution, 
            device=device
        )
        
        renderer = PerspectiveViewGenerator(
            dev=device,
            image_size=args.resolution,
            fov=args.renderer_fov,
            distance=args.renderer_distance
        )
        
        azim_angles = np.arange(-90+1, 90-1, args.azim_step)
        
        rendered_views, _, _ = renderer.render_on_backgrounds(
            texture_tensor.float(),
            background_tensor,
            longitudes=azim_angles.tolist(),
            latitudes=[0],
            distance=args.renderer_distance
        )
        
        # Composite rendered views with background
        bg_composited_views = []
        for j in range(len(azim_angles)):
            # Get single view
            view = rendered_views[j, 0]  # [C,H,W]
            
            # Ensure value range is [0,1]
            if view.min() < 0 or view.max() > 1:
                view = (view + 1) / 2
            view = view.clamp(0, 1)
            
            # Composite with background
            bg = background_tensor[0]  # Remove batch dimension [C,H,W]
            composited = composite_image(view, bg)
            bg_composited_views.append(composited.unsqueeze(0))  # Add batch dimension
        
        # Prepare CSV file
        csv_path = os.path.join(output_path, 'angle_confidence.csv')
        
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            
            header = ['file_path', 'prompt', 'ground_truth_label', 'camera_distance']
            header.extend([f'angle_{int(angle)}' for angle in azim_angles])
            csv_writer.writerow(header)
            
            rel_path = '/'.join(texture_path.split('/')[-3:])
            data_row = [rel_path, prompt, args.target_class_id, args.renderer_distance]
            
            # Stack composited views as batch
            bg_composited_batch = torch.stack(bg_composited_views)
            
            # Check shape
            if len(bg_composited_batch.shape) == 5:  # If shape is [V,B,C,H,W]
                V, B, C, H, W = bg_composited_batch.shape
                batch_views = bg_composited_batch.reshape(V*B, C, H, W)
            elif len(bg_composited_batch.shape) == 4:  # If shape is [V,C,H,W]
                V, C, H, W = bg_composited_batch.shape
                batch_views = bg_composited_batch
            else:
                raise ValueError(f"Unexpected shape for bg_composited_batch: {bg_composited_batch.shape}")
            
            # If needed, resize to model expected size
            if H != 640 or W != 640:
                batch_views_resized = F.interpolate(batch_views, 
                                                   size=(640, 640), 
                                                   mode='bilinear', 
                                                   align_corners=False)
            else:
                batch_views_resized = batch_views
            
            # Use batch detection
            all_confidences = batch_detect_stop_sign_unified(
                model, batch_views_resized, args.detector, batch_size=args.batch_size, target_class_id=args.target_class_id
            )
            
            # Add confidences to data row
            data_row.extend(all_confidences)
            
            # Save rendered images (if needed)
            if args.save_renders:
                for i, angle in enumerate(azim_angles):
                    render_path = os.path.join(render_dir, f"azim_{int(angle)}.png")
                    save_rendered_image(
                        bg_composited_views[i][0], 
                        render_path, 
                        title=f"Azimuth: {angle}°, Confidence: {all_confidences[i]:.4f}",
                        add_title=True
                    )
                    if (i + 1) % 10 == 0 or (i + 1) == len(azim_angles):
                        print(f"    Saved {i + 1}/{len(azim_angles)} rendered images")
            
            # Save front view (0° or closest to 0°)
            front_index = np.abs(azim_angles).argmin()  # Find index of angle closest to 0°
            front_angle = azim_angles[front_index]
            front_view_confidence = all_confidences[front_index]
            front_view_path = os.path.join(output_path, 'front_view.png')
            save_rendered_image(
                bg_composited_views[front_index][0],
                front_view_path,
                title=f"Front View (Azimuth: {front_angle}°), Confidence: {front_view_confidence:.4f}",
                add_title=True
            )
            print(f"  Front view saved to {front_view_path}")
            
            csv_writer.writerow(data_row)
        
        print(f"  Results saved to {csv_path}")
        
        # ========== AASR Calculation ==========
        aasr_results = None
        if args.enable_aasr and AASR_AVAILABLE:
            try:
                # Parse AASR thresholds
                aasr_thresholds = [float(t) for t in args.aasr_thresholds.split(',')]
                
                # Filter angle range
                angle_mask = [(args.aasr_min_angle <= angle <= args.aasr_max_angle) 
                             for angle in azim_angles]
                filtered_angles = [angle for angle, mask in zip(azim_angles, angle_mask) if mask]
                filtered_confidences = [conf for conf, mask in zip(all_confidences, angle_mask) if mask]
                
                # Calculate AASR
                aasr_output_dir = os.path.join(output_path, 'aasr_analysis')
                aasr_results = calculate_aasr_for_texture(
                    filtered_angles,
                    filtered_confidences,
                    thresholds=aasr_thresholds,
                    output_dir=aasr_output_dir
                )
                
                # Print AASR results
                filename = os.path.basename(texture_path)
                print_aasr_summary(aasr_results, filename)
                
            except Exception as e:
                print(f"  Warning: AASR calculation failed: {str(e)}")
                import traceback
                traceback.print_exc()
        elif args.enable_aasr and not AASR_AVAILABLE:
            print("  Warning: metrics.py not found, skipping AASR calculation")
        # ==============================
        
        # Create confidence curve plot
        plot_path = os.path.join(output_path, 'confidence_curve.png')
        filename = os.path.basename(texture_path)
        plot_confidence_curve(azim_angles, all_confidences, plot_path, title=filename)
        
        return {
            'texture_path': texture_path,
            'output_path': output_path,
            'prompt': prompt,
            'front_confidence': front_view_confidence,
            'max_confidence': max(all_confidences),
            'min_confidence': min(all_confidences),
            'avg_confidence': sum(all_confidences) / len(all_confidences),
            'angle_confidences': dict(zip([int(angle) for angle in azim_angles], all_confidences)),
            'aasr_results': aasr_results  # New: return AASR results
        }
    
    except Exception as e:
        print(f"Error processing {texture_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'texture_path': texture_path,
            'error': str(e)
        }

def main():
    args = parse_args()
    start_time = time.time()
    
    # Automatically add detector name and environment identifier to output_dir
    args.output_dir = os.path.join(args.output_dir, f"{args.detector}")
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    # Find all texture files in folder
    texture_files = find_texture_files(args.texture_folder, args.texture_extensions)
    print(f"Found {len(texture_files)} texture files to process")
    
    # Load detector
    detector_model = load_detector(args, device)
    print(f"Loaded {args.detector} detector, target class ID: {args.target_class_id}")
    
    # Load background image (reused for all textures)
    background_tensor = load_background_image(
        args.background_image,
        size=tuple(args.background_size),
        device=device
    )
    print(f"Loaded background image: {args.background_image}")
    
    # Prepare model pool and background pool (for multithreading)
    model_pool = []
    background_pool = []
    if args.num_workers > 1 and args.detector in ['yolov3', 'faster_rcnn', 'detr']:
        # Create model instance for each worker (max 4)
        num_models = min(args.num_workers, 4)
        print(f"Creating model pool with {num_models} model instances...")
        for i in range(num_models):
            if torch.cuda.device_count() > 1:
                worker_device = torch.device(f"cuda:{i % torch.cuda.device_count()}")
            else:
                worker_device = device
            
            if i == 0:
                model_pool.append(detector_model)
                background_pool.append(background_tensor)
            else:
                model_pool.append(load_detector(args, worker_device))
                background_pool.append(load_background_image(
                    args.background_image,
                    size=tuple(args.background_size),
                    device=worker_device
                ))
    else:
        model_pool = [detector_model]
        background_pool = [background_tensor]
    
    # Process all texture files
    all_results = []
    
    if args.num_workers <= 1:
        # Sequential processing
        for texture_file in tqdm(texture_files, desc="Processing textures"):
            result = process_texture_file(
                texture_file, 
                detector_model, 
                background_tensor,
                args.output_dir, 
                args, 
                device
            )
            all_results.append(result)
    else:
        # Parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = []
            for i, texture_file in enumerate(texture_files):
                # Select model and background from pools
                worker_index = i % len(model_pool)
                worker_model = model_pool[worker_index]
                worker_background = background_pool[worker_index]
                
                future = executor.submit(
                    process_texture_file,
                    texture_file,
                    worker_model,
                    worker_background,
                    args.output_dir,
                    args,
                    worker_model.device if hasattr(worker_model, 'device') else device
                )
                futures.append(future)
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing textures"):
                all_results.append(future.result())
    
    # Generate summary results
    texture_folder_parts = args.texture_folder.rstrip('/').split('/')
    if len(texture_folder_parts) >= 3:
        output_subdir = os.path.join(
            texture_folder_parts[-3],
            texture_folder_parts[-2],
            texture_folder_parts[-1]
        )
    elif len(texture_folder_parts) >= 2:
        output_subdir = os.path.join(texture_folder_parts[-2], texture_folder_parts[-1])
    elif len(texture_folder_parts) >= 1:
        output_subdir = texture_folder_parts[-1]
    else:
        output_subdir = ""

    summary_dir = os.path.join(args.output_dir, output_subdir)
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, args.summary_output)
    
    # Generate summary CSV
    with open(summary_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['texture_path', 'prompt', 'front_confidence', 'max_confidence', 'min_confidence', 'avg_confidence']
        csv_writer.writerow(header)
        
        for result in all_results:
            if 'error' not in result:
                csv_writer.writerow([
                    result['texture_path'],
                    result['prompt'],
                    result['front_confidence'],
                    result['max_confidence'],
                    result['min_confidence'],
                    result['avg_confidence']
                ])
    
    # ========== Generate folder-level AASR summary ==========
    if args.enable_aasr and AASR_AVAILABLE:
        try:
            print("\nGenerating folder-level AASR summary...")
            
            # Collect AASR results for all textures
            texture_aasr_results = {}
            for result in all_results:
                if 'error' not in result and 'aasr_results' in result and result['aasr_results']:
                    texture_name = os.path.splitext(os.path.basename(result['texture_path']))[0]
                    texture_aasr_results[texture_name] = result['aasr_results']
            
            # Generate summary CSV
            if texture_aasr_results:
                folder_aasr_summary_path = os.path.join(summary_dir, 'folder_aasr_summary.csv')
                create_folder_aasr_summary(texture_aasr_results, folder_aasr_summary_path)
                
                # Print summary statistics
                import pandas as pd
                summary_df = pd.read_csv(folder_aasr_summary_path)
                avg_row = summary_df[summary_df['Texture'] == 'Average'].iloc[0]
                print(f"\nFolder average AASR: {avg_row['Average_AASR']:.2f}%")
                
        except Exception as e:
            print(f"Warning: Error generating AASR summary: {str(e)}")
            import traceback
            traceback.print_exc()
    # ==============================================
    
    # Print summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nProcessed {len(texture_files)} texture files in {elapsed_time:.2f} seconds")
    print(f"Using detector: {args.detector}")
    print(f"Detecting target class ID: {args.target_class_id}")
    print(f"Using background image: {args.background_image}")
    print(f"Summary results saved to {summary_path}")
    
    # Generate summary charts (original code, kept unchanged)
    if len(all_results) > 1:
        example_result = next((r for r in all_results if 'angle_confidences' in r), None)
        if example_result:
            angles = sorted(example_result['angle_confidences'].keys())
            
            # Save all textures' angle-confidence data to CSV
            all_textures_data_path = os.path.join(summary_dir, 'all_textures_data.csv')
            with open(all_textures_data_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                header = ['angle']
                for result in all_results:
                    if 'angle_confidences' in result:
                        texture_name = os.path.basename(result['texture_path'])
                        texture_name = os.path.splitext(texture_name)[0]
                        header.append(texture_name)
                csv_writer.writerow(header)
                
                for angle in angles:
                    row = [angle]
                    for result in all_results:
                        if 'angle_confidences' in result and angle in result['angle_confidences']:
                            row.append(result['angle_confidences'][angle])
                        else:
                            row.append('')
                    csv_writer.writerow(row)
            print(f"All textures' angle-confidence data saved to {all_textures_data_path}")
            
            # 1. Create comparison plot with all textures
            plt.figure(figsize=(40, 10))
            
            for result in all_results:
                if 'angle_confidences' in result:
                    texture_name = os.path.basename(result['texture_path'])
                    confidences = [result['angle_confidences'][angle] for angle in angles]
                    plt.plot(angles, confidences, '-', linewidth=1.5, label=texture_name)
            
            plt.xlabel('Azimuth Angle (degrees)')
            plt.ylabel('Detection Confidence')
            plt.title(f'Target Detection Confidence vs. Viewing Angle - All Textures ({args.detector.upper()} with Environment)')
            plt.grid(True)
            plt.xlim(-90, 90)
            plt.ylim(0, 1.05)
            
            plt.axvline(x=0, color='k', linestyle='--', label='Direct View (0°)')
            
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                      fancybox=True, shadow=True, ncol=min(8, len(all_results)), fontsize=8)
            
            plt.tight_layout(pad=2.0)
            
            summary_plot_path = os.path.join(summary_dir, 'all_textures_comparison.png')
            plt.savefig(summary_plot_path, dpi=300)
            plt.close()
            print(f"Comparison plot saved to {summary_plot_path}")
            
            # 2. Create average confidence plot
            plt.figure(figsize=(12, 8))
            
            mean_confidences = []
            std_confidences = []
            
            for angle in angles:
                angle_confidences = []
                for result in all_results:
                    if 'angle_confidences' in result and angle in result['angle_confidences']:
                        angle_confidences.append(result['angle_confidences'][angle])
                
                if angle_confidences:
                    mean_confidences.append(np.mean(angle_confidences))
                    std_confidences.append(np.std(angle_confidences))
                else:
                    mean_confidences.append(0)
                    std_confidences.append(0)
            
            mean_confidence_data_path = os.path.join(summary_dir, 'mean_confidence_data.csv')
            with open(mean_confidence_data_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['angle', 'mean_confidence', 'std_dev_minus', 'std_dev_plus'])
                for i, angle in enumerate(angles):
                    mean = mean_confidences[i]
                    std = std_confidences[i]
                    csv_writer.writerow([angle, mean, mean - std, mean + std])
            print(f"Mean confidence data saved to {mean_confidence_data_path}")
            
            # Plot mean curve
            plt.plot(angles, mean_confidences, 'b-', linewidth=3, label='Mean Confidence')
            
            # Add confidence interval (±1 standard deviation)
            plt.fill_between(
                angles,
                np.array(mean_confidences) - np.array(std_confidences),
                np.array(mean_confidences) + np.array(std_confidences),
                alpha=0.3, color='b', label='±1 Std Dev'
            )
            
            plt.xlabel('Azimuth Angle (degrees)', fontsize=12)
            plt.ylabel('Detection Confidence', fontsize=12)
            plt.title(f'Mean Target Detection Confidence vs. Viewing Angle ({args.detector.upper()} with Environment)', fontsize=14)
            plt.grid(True)
            plt.xlim(-90, 90)
            plt.ylim(0, 1.05)
            
            # Add vertical line at 0 degrees (direct view)
            plt.axvline(x=0, color='g', linestyle='--', label='Direct View (0°)')
            
            # Add confidence threshold horizontal lines
            plt.axhline(y=0.25, color='r', linestyle=':', alpha=0.7, label='0.25 Confidence')
            plt.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='0.50 Confidence')
            plt.axhline(y=0.75, color='g', linestyle=':', alpha=0.7, label='0.75 Confidence')
            
            # Add text annotations for key statistics
            max_mean = max(mean_confidences)
            max_angle = angles[mean_confidences.index(max_mean)]
            
            # Calculate minimum in central region, avoiding extreme angles
            center_start_idx = next(i for i, a in enumerate(angles) if a >= -45)
            center_end_idx = next(i for i, a in enumerate(angles) if a >= 45)
            center_confidences = mean_confidences[center_start_idx:center_end_idx]
            center_angles = angles[center_start_idx:center_end_idx]
            
            min_mean = min(center_confidences)
            min_angle = center_angles[center_confidences.index(min_mean)]
            
            plt.annotate(f'Max: {max_mean:.2f} at {max_angle}°', 
                        xy=(max_angle, max_mean), xytext=(max_angle+10, max_mean),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
            
            plt.annotate(f'Min: {min_mean:.2f} at {min_angle}°', 
                        xy=(min_angle, min_mean), xytext=(min_angle-30, min_mean-0.1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
            
            plt.legend(loc='lower center', fontsize=10)
            plt.tight_layout(pad=1.5)
            
            # Save mean plot
            mean_plot_path = os.path.join(summary_dir, 'all_textures_mean.png')
            plt.savefig(mean_plot_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
            plt.close()
            print(f"Mean confidence plot saved to {mean_plot_path}")
            
            # 3. Create heatmap showing confidence at different angles for all textures
            plt.figure(figsize=(15, 10))
            
            # Prepare heatmap data
            heatmap_data = []
            texture_names = []
            
            for result in all_results:
                if 'angle_confidences' in result:
                    texture_name = os.path.splitext(os.path.basename(result['texture_path']))[0]
                    texture_names.append(texture_name)
                    
                    # Get confidence at all angles for this texture
                    confidences = [result['angle_confidences'][angle] for angle in angles]
                    heatmap_data.append(confidences)
            
            # Convert to NumPy array
            heatmap_array = np.array(heatmap_data)
            
            # Create heatmap
            plt.imshow(heatmap_array, aspect='auto', cmap='viridis', 
                      extent=[angles[0], angles[-1], 0, len(texture_names)])
            
            # Set ticks and labels
            plt.colorbar(label='Detection Confidence')
            plt.xlabel('Azimuth Angle (degrees)')
            plt.ylabel('Texture')
            plt.title(f'Target Detection Confidence Heatmap ({args.detector.upper()} with Environment)')
            
            # Set y-axis tick labels (texture names)
            plt.yticks(np.arange(len(texture_names)) + 0.5, texture_names, fontsize=8)
            
            # Add vertical line at 0 degrees
            plt.axvline(x=0, color='r', linestyle='--', linewidth=1)
            
            plt.tight_layout()
            
            # Save heatmap
            heatmap_path = os.path.join(summary_dir, 'confidence_heatmap.png')
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confidence heatmap saved to {heatmap_path}")
            
            # 4. Create bar chart sorted by maximum confidence
            plt.figure(figsize=(14, 8))
            
            # Sort results by maximum confidence
            sorted_results = sorted(
                [r for r in all_results if 'max_confidence' in r],
                key=lambda x: x['max_confidence'],
                reverse=True
            )
            
            # Extract sorted data
            sorted_texture_names = [os.path.splitext(os.path.basename(r['texture_path']))[0] for r in sorted_results]
            max_confidences = [r['max_confidence'] for r in sorted_results]
            avg_confidences = [r['avg_confidence'] for r in sorted_results]
            
            # Set x positions
            x = np.arange(len(sorted_texture_names))
            width = 0.35
            
            # Plot bar chart
            bars1 = plt.bar(x - width/2, max_confidences, width, label='Maximum Confidence')
            bars2 = plt.bar(x + width/2, avg_confidences, width, label='Average Confidence')
            
            # Set chart properties
            plt.xlabel('Texture')
            plt.ylabel('Confidence')
            plt.title(f'Maximum and Average Target Detection Confidence by Texture ({args.detector.upper()} with Environment)')
            plt.xticks(x, sorted_texture_names, rotation=45, ha='right', fontsize=8)
            plt.ylim(0, 1.05)
            plt.grid(True, axis='y', linestyle='--', alpha=0.7)
            plt.legend()
            
            # Add value labels for maximum values
            for bar in bars1:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', rotation=0, fontsize=8)
            
            plt.tight_layout()
            
            # Save bar chart
            bar_chart_path = os.path.join(summary_dir, 'confidence_ranking.png')
            plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Confidence ranking plot saved to {bar_chart_path}")
            
            # 5. Calculate and display angle range statistics
            # Define angle ranges (e.g., front view, side view, etc.)
            angle_ranges = [
                ('Front View', -15, 15),
                ('Moderate Side View', -45, -15),
                ('Moderate Side View', 15, 45),
                ('Extreme Side View', -90, -45),
                ('Extreme Side View', 45, 90)
            ]
            
            # Calculate average confidence for each texture in different angle ranges
            range_stats = {}
            for range_name, angle_min, angle_max in angle_ranges:
                range_stats[f"{range_name} ({angle_min}° to {angle_max}°)"] = []
                
                for result in all_results:
                    if 'angle_confidences' in result:
                        # Get confidences in this angle range
                        range_confidences = [
                            conf for angle, conf in result['angle_confidences'].items()
                            if angle_min <= angle <= angle_max
                        ]
                        
                        if range_confidences:
                            # Calculate average confidence for this texture in this angle range
                            range_avg = sum(range_confidences) / len(range_confidences)
                            texture_name = os.path.splitext(os.path.basename(result['texture_path']))[0]
                            range_stats[f"{range_name} ({angle_min}° to {angle_max}°)"].append((texture_name, range_avg))
            
            # Create angle range statistics CSV
            range_stats_path = os.path.join(summary_dir, 'angle_range_stats.csv')
            with open(range_stats_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                # Write header
                csv_writer.writerow(['Angle Range', 'Texture', 'Average Confidence'])
                
                # Write statistics for each angle range
                for range_name, stats in range_stats.items():
                    # Sort by average confidence
                    sorted_stats = sorted(stats, key=lambda x: x[1], reverse=True)
                    
                    for texture_name, avg_conf in sorted_stats:
                        csv_writer.writerow([range_name, texture_name, avg_conf])
            
            print(f"Angle range statistics saved to {range_stats_path}")
            
            # 6. Create detector performance summary
            if 'error' not in all_results[0]:
                performance_summary = {
                    'detector': args.detector.upper(),
                    'target_class_id': args.target_class_id,
                    'background': os.path.basename(args.background_image),
                    'total_textures': len([r for r in all_results if 'error' not in r]),
                    'avg_front_confidence': np.mean([r['front_confidence'] for r in all_results if 'error' not in r]),
                    'avg_max_confidence': np.mean([r['max_confidence'] for r in all_results if 'error' not in r]),
                    'avg_min_confidence': np.mean([r['min_confidence'] for r in all_results if 'error' not in r]),
                    'avg_avg_confidence': np.mean([r['avg_confidence'] for r in all_results if 'error' not in r]),
                }
                
                # Create performance summary CSV
                performance_path = os.path.join(summary_dir, f'{args.detector}_performance_summary.csv')
                with open(performance_path, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    csv_writer.writerow(['Metric', 'Value'])
                    for key, value in performance_summary.items():
                        csv_writer.writerow([key, value])
                
                print(f"Detector environment performance summary saved to {performance_path}")
                
                # Print performance summary
                print(f"\n{args.detector.upper()} Detector Performance Summary in Environment:")
                print(f"  Detecting target class ID: {performance_summary['target_class_id']}")
                print(f"  Environment background: {performance_summary['background']}")
                print(f"  Number of textures processed: {performance_summary['total_textures']}")
                print(f"  Average front confidence: {performance_summary['avg_front_confidence']:.4f}")
                print(f"  Average maximum confidence: {performance_summary['avg_max_confidence']:.4f}")
                print(f"  Average minimum confidence: {performance_summary['avg_min_confidence']:.4f}")
                print(f"  Overall average confidence: {performance_summary['avg_avg_confidence']:.4f}")

if __name__ == "__main__":
    main()