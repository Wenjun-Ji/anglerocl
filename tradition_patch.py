#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

# Training script for angle-robust adversarial patch using traditional optimization

import os
import argparse
import logging
import time
import json
import random
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm.auto import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from typing import Tuple, Dict, List, Any

# Import YOLO related modules
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, xyxy2xywh
from utils.torch_utils import select_device
from adv_patch_gen.utils.loss import MaxProbExtractor
from adv_patch_gen.utils.patch import PerspectiveViewGenerator

# Import metrics
from metrics import calculate_EAR, calculate_ARS, calculate_ASI

# Set up logger
logger = get_logger(__name__)

# ==============================================================================
# Utility Functions
# ==============================================================================

def setup_logging(output_dir):
    """Set up logging system"""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, f"training_{time.strftime('%Y%m%d_%H%M%S')}.log")
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def seed_everything(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_angles():
    """Get training angles for rendering."""
    # Training angles: -75°, -60°, -30°, -15°, 0°, 15°, 30°, 60°, 75°
    angles = [-75, -60, -30, -15, 0, 15, 30, 60, 75]
    return angles


def get_test_angles():
    """Get test angles for validation."""
    # Test angles: from -85° to 85°, sampled every 10°
    angles = list(range(-85, 90, 10))
    
    # Ensure all training angles are included in test angles
    train_angles = get_train_angles()
    for angle in train_angles:
        if angle not in angles:
            angles.append(angle)
    
    # Sort angles
    angles.sort()
    
    return angles


def generate_patch(patch_type, size=(300, 300), device='cuda'):
    """
    Generate initial patch
    
    Args:
        patch_type: Generation type, 'gray', 'random', 'black', or 'white'
        size: Patch size, default is 300x300
        device: Device, default is cuda
    
    Returns:
        Initialized patch tensor
    """
    if patch_type == 'gray':
        patch = torch.ones(3, size[0], size[1], device=device) * 0.5
    elif patch_type == 'random':
        patch = torch.rand(3, size[0], size[1], device=device)
    elif patch_type == 'black':
        patch = torch.zeros(3, size[0], size[1], device=device)
    elif patch_type == 'white':
        patch = torch.ones(3, size[0], size[1], device=device)
    else:
        raise ValueError(f"Unknown patch type: {patch_type}")
    return patch


def load_patch_from_file(file_path, device='cuda'):
    """Load pre-trained patch from file"""
    if file_path.endswith('.pt'):
        patch = torch.load(file_path, map_location=device)
    else:
        # Assume it's an image file
        image = Image.open(file_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        patch = transform(image).to(device)
    return patch


def tensor_to_pil(tensor):
    """
    Convert PyTorch tensor to PIL image
    
    Args:
        tensor: Tensor of shape [C, H, W] with value range [0, 1]
    
    Returns:
        PIL Image
    """
    if tensor.dim() == 4 and tensor.size(0) == 1:  # Batch dimension is 1
        tensor = tensor.squeeze(0)
    
    if tensor.dim() != 3:
        raise ValueError(f"Expected tensor with 3 dimensions, got {tensor.dim()}")
    
    # Convert to CPU and numpy
    tensor = tensor.detach().cpu()
    
    # Check value range
    if tensor.min() < 0 or tensor.max() > 1:
        # Assume range is [-1, 1], convert to [0, 1]
        tensor = (tensor + 1) / 2
        tensor = tensor.clamp(0, 1)
    
    # Convert to PIL
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return Image.fromarray(tensor)


def calculate_ear_loss(detection_results, angles, threshold=0.8, scale=10):
    """
    Calculate EAR (Effective Angle Range) loss.
    
    Goal: Maximize the effective detection angle range.
    Method: Use ReLU to directly penalize detection results below threshold.
    
    Args:
        detection_results: Detection confidence results [V*B]
        angles: Corresponding angle list
        threshold: Detection threshold
        scale: Scale factor to control loss sensitivity
    
    Returns:
        loss: EAR loss value
    """
    # Reshape detection results by batch and angle
    V = len(angles)
    B = detection_results.shape[0] // V
    detections = detection_results.view(B, V)  # [B, V]
    
    # Calculate gap between each angle and threshold (only consider below threshold)
    threshold_gaps = F.relu(threshold - detections)
    
    # Apply scale factor
    scaled_gaps = scale * threshold_gaps
    
    # Calculate average loss per batch
    ear_loss = scaled_gaps.mean(dim=1)  # [B]
    
    # Return average loss across all batches
    return ear_loss.mean()


# ==============================================================================
# Dataset Class
# ==============================================================================

class ImageFolderDataset(Dataset):
    """Simple image folder dataset for loading background images"""
    def __init__(self, folder_path, transform=None, size=512):
        self.folder_path = folder_path
        self.transform = transform
        self.size = size
        
        # Get all image files
        self.image_files = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(root, file))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transformation
        if self.transform:
            image = self.transform(image)
        else:
            # Default transformation
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
            image = transform(image)
        
        return image


# ==============================================================================
# Training Functions
# ==============================================================================

def train_epoch(adv_patch, yolo_model, renderer, prob_extractor, optimizer, args, accelerator, epoch, data_loader=None):
    """
    Train one epoch
    
    Args:
        adv_patch: Patch tensor
        yolo_model: YOLO model
        renderer: Angle renderer
        prob_extractor: Probability extractor
        optimizer: Optimizer
        args: Configuration parameters
        accelerator: Accelerator
        epoch: Current epoch
        data_loader: Data loader (optional)
    
    Returns:
        Average loss and loss components
    """
    train_angles = get_train_angles()
    total_loss = 0
    ear_loss_total = 0
    
    batch_count = 0
    
    # Create progress bar using tqdm
    if data_loader:
        progress_bar = tqdm(data_loader, desc=f"Training Epoch {epoch}")
        batches = progress_bar
    else:
        progress_bar = tqdm(range(args.batch_size_per_epoch), desc=f"Training Epoch {epoch}")
        batches = range(args.batch_size_per_epoch)
    
    for batch_idx, data in enumerate(batches):
        # Get background images
        if data_loader:
            background = data.to(accelerator.device)
        else:
            # If no data loader, use random noise as background
            background = torch.rand(args.batch_size, 3, args.resolution, args.resolution).to(accelerator.device)
        
        # Use render_on_backgrounds method to render patch on backgrounds at different angles
        rendered_views, _, _ = renderer.render_on_backgrounds(
            adv_patch.float(),
            background,
            longitudes=train_angles,
            latitudes=[0],
            distance=args.renderer_distance
        )
        
        # Reshape tensor for detector [V, B, C, H, W] -> [V*B, C, H, W]
        V, B, C, H, W = rendered_views.shape
        rendered_views = rendered_views.reshape(V * B, C, H, W)
        
        # Visualize rendered_views (only save first batch)
        if batch_idx == 0:
            vis_dir = os.path.join(args.output_dir, "visualization")
            os.makedirs(vis_dir, exist_ok=True)
            tensor_to_pil(rendered_views[0]).save(os.path.join(vis_dir, f"rendered_view_epoch_{epoch}.png"))
        
        # Resize to YOLO input size
        resized_views = F.interpolate(
            rendered_views, 
            size=(640, 640),
            mode='bilinear',
            align_corners=False
        )
        
        # YOLO detection
        detection_output = yolo_model(resized_views.float())[0]
        max_prob = prob_extractor(detection_output)
        
        # Calculate loss (only use EAR loss)
        loss = calculate_ear_loss(max_prob, train_angles, args.detection_threshold)
        
        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Constrain patch values to [0,1] range
        with torch.no_grad():
            adv_patch.data.clamp_(0, 1)
        
        # Accumulate loss
        batch_count += 1
        total_loss += loss.item()
        ear_loss_total += loss.item()
        
        # Update progress bar
        if isinstance(progress_bar, tqdm):
            progress_bar.set_postfix({
                'loss': loss.item(),
                'ear_loss': loss.item()
            })
        
        # Periodically save intermediate results
        if batch_idx % args.save_interval == 0 and batch_idx > 0:
            # Save current patch visualization
            vis_dir = os.path.join(args.output_dir, "visualization")
            os.makedirs(vis_dir, exist_ok=True)
            
            # Save patch
            patch_image = tensor_to_pil(adv_patch)
            patch_image.save(os.path.join(vis_dir, f"patch_epoch_{epoch}_batch_{batch_idx}.png"))
            
            # Save sample image with patch applied
            if rendered_views.shape[0] > 0:
                sample_image = tensor_to_pil(rendered_views[0])
                sample_image.save(os.path.join(vis_dir, f"sample_epoch_{epoch}_batch_{batch_idx}.png"))
    
    # Calculate average loss
    avg_loss = total_loss / batch_count if batch_count > 0 else 0
    avg_ear_loss = ear_loss_total / batch_count if batch_count > 0 else 0
    
    # Log results
    logger.info(
        f"Epoch {epoch} | "
        f"Loss: {avg_loss:.4f} | "
        f"EAR: {avg_ear_loss:.4f}"
    )
    
    # Return average loss and loss components
    avg_loss_components = {
        'total_loss': avg_loss,
        'ear_loss': avg_ear_loss
    }
    
    torch.cuda.empty_cache()
    
    return avg_loss, avg_loss_components


# ==============================================================================
# Validation Functions
# ==============================================================================

def create_validation_plots(results, angles, args, save_dir, epoch):
    """
    Create visualization plots for validation results
    
    Args:
        results: Validation results dictionary
        angles: Angle list
        args: Configuration parameters
        save_dir: Save directory
        epoch: Current epoch
    """
    # Convert results to lists
    angle_list = sorted(angles)
    avg_confidences = [np.mean(results["confidences"][angle]) if results["confidences"][angle] else 0 
                      for angle in angle_list]
    avg_detection_rates = [np.mean(results["detection_rates"][angle]) if results["detection_rates"][angle] else 0 
                          for angle in angle_list]
    
    # Create confidence plot
    plt.figure(figsize=(12, 6))
    plt.plot(angle_list, avg_confidences, 'b-o', linewidth=2, markersize=6)
    plt.axhline(y=args.detection_threshold, color='r', linestyle='--', 
               label=f'Threshold ({args.detection_threshold})')
    plt.xlabel('Angle (degrees)', fontsize=12)
    plt.ylabel('Average Confidence', fontsize=12)
    plt.title(f'Average Detection Confidence vs Angle (Epoch {epoch})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confidence_plot.png'), dpi=300)
    plt.close()
    
    # Create detection rate plot
    plt.figure(figsize=(12, 6))
    plt.plot(angle_list, avg_detection_rates, 'g-o', linewidth=2, markersize=6)
    plt.xlabel('Angle (degrees)', fontsize=12)
    plt.ylabel('Detection Rate', fontsize=12)
    plt.title(f'Detection Rate vs Angle (Epoch {epoch})', fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'detection_rate_plot.png'), dpi=300)
    plt.close()
    
    # Save raw data
    data_file = os.path.join(save_dir, 'validation_results.json')
    with open(data_file, 'w') as f:
        json_data = {
            'angles': angle_list,
            'avg_confidences': avg_confidences,
            'avg_detection_rates': avg_detection_rates
        }
        json.dump(json_data, f, indent=2)


def validate_patch(adv_patch, yolo_model, renderer, prob_extractor, args, accelerator, epoch, data_loader=None):
    """Validate current patch effectiveness (memory-optimized version)"""
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Use test angles consistent with training
    test_angles = get_test_angles()
    
    # Create validation output directory
    validation_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)
    epoch_dir = os.path.join(validation_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Dictionary to store results
    results = {
        "confidences": {angle: [] for angle in test_angles},
        "detection_rates": {angle: [] for angle in test_angles}
    }
    
    # Reduce validation sample count
    num_examples = 1  # Only use 1 sample for validation
    
    # Prepare validation data
    if data_loader:
        # Only use 1 background image
        val_data = []
        val_loader_iter = iter(data_loader)
        try:
            val_data.append(next(val_loader_iter))
        except StopIteration:
            # If data loader is empty, create random background
            val_data.append(torch.rand(1, 3, args.resolution, args.resolution).to(accelerator.device))
    else:
        # Create 1 random background
        val_data = [torch.rand(1, 3, args.resolution, args.resolution).to(accelerator.device)]
    
    # Visualize val_data
    val_image = tensor_to_pil(val_data[0][0])
    val_image.save(os.path.join(epoch_dir, "background.png"))
    
    # Test all angles on 1 background
    background = val_data[0]
    # Ensure background is single sample
    if background.shape[0] > 1:
        background = background[0:1].to(accelerator.device)
    else:
        background = background.to(accelerator.device)
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Use minimal resource configuration for rendering
    with torch.no_grad():  # Ensure no gradient computation
        rendered_views, _, _ = renderer.render_on_backgrounds(
            adv_patch.float(),
            background,
            longitudes=test_angles,
            latitudes=[0],
            distance=args.renderer_distance
        )
    
    # Reshape tensor [V, B, C, H, W] -> [V*B, C, H, W]
    V, B, C, H, W = rendered_views.shape
    rendered_views = rendered_views.reshape(V * B, C, H, W)
    
    # Visualize all rendered views
    renders_dir = os.path.join(epoch_dir, "rendered_views")
    os.makedirs(renders_dir, exist_ok=True)
    logger.info(f"Saving {rendered_views.shape[0]} rendered views to {renders_dir}")

    # Save each rendered view
    for i in range(rendered_views.shape[0]):
        # Get angle for current view
        angle_idx = i // B
        angle = test_angles[angle_idx] if angle_idx < len(test_angles) else "unknown"
        
        # Convert tensor to PIL image and save
        img = tensor_to_pil(rendered_views[i])
        img.save(os.path.join(renders_dir, f"angle_{angle}_view_{i}.png"))
        
        # Log every 10 images to prevent excessive logging
        if i % 10 == 0:
            logger.info(f"Saved {i+1}/{rendered_views.shape[0]} rendered views")
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Resize to YOLO input size
    resized_views = F.interpolate(
        rendered_views, 
        size=(640, 640),
        mode='bilinear',
        align_corners=False
    )
    
    # Free memory
    del rendered_views
    torch.cuda.empty_cache()
    
    # YOLO detection - process images in batches to reduce memory usage
    batch_size = 1  # Process only 1 image at a time
    predictions = []
    
    for i in range(0, resized_views.shape[0], batch_size):
        # Get current batch
        batch = resized_views[i:i+batch_size]
        
        # Perform detection
        with torch.no_grad():
            batch_output = yolo_model(batch.float())[0]
        
        # Perform NMS
        batch_pred = non_max_suppression(
            batch_output, 
            conf_thres=0.0001,
            iou_thres=0.45,
            classes=[args.objective_class_id] if args.objective_class_id is not None else None,
            max_det=100
        )
        
        # Save prediction results
        predictions.extend(batch_pred)
        
        # Free memory
        del batch_output, batch_pred, batch
        torch.cuda.empty_cache()
    
    # Extract confidences
    for angle_idx, angle in enumerate(test_angles):
        batch_offset = angle_idx * B
        for b in range(B):
            det_idx = batch_offset + b
            if det_idx < len(predictions):
                det = predictions[det_idx]
                if len(det) > 0:
                    confidence = float(det[:, 4].max().item())
                else:
                    confidence = 0.0
                results["confidences"][angle].append(confidence)
                results["detection_rates"][angle].append(1 if confidence >= args.detection_threshold else 0)
    
    # Free memory
    del predictions, resized_views
    torch.cuda.empty_cache()
    
    # Log confidence statistics before validation ends
    all_confidences = []
    for values in results["confidences"].values():
        all_confidences.extend(values)

    if all_confidences:
        logger.info(f"Confidence statistics - Min: {min(all_confidences):.4f}, "
                   f"Max: {max(all_confidences):.4f}, "
                   f"Avg: {sum(all_confidences)/len(all_confidences):.4f}")
        # Print confidence distribution
        bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist = [0] * (len(bins)-1)
        for conf in all_confidences:
            for i in range(len(bins)-1):
                if bins[i] <= conf < bins[i+1]:
                    hist[i] += 1
                    break
        logger.info(f"Confidence distribution: {hist}")
    
    # Calculate averages
    avg_confidences = {angle: sum(values)/len(values) if values else 0 
                      for angle, values in results["confidences"].items()}
    avg_detection_rates = {angle: sum(values)/len(values) if values else 0 
                          for angle, values in results["detection_rates"].items()}
    
    # Calculate metrics
    angle_confidences = [avg_confidences[angle] for angle in sorted(test_angles)]
    total_range, ear_ranges = calculate_EAR(sorted(test_angles), angle_confidences, threshold=args.detection_threshold)
    ear_score = total_range / 180.0
    ars_results = calculate_ARS(sorted(test_angles), angle_confidences)
    ars_score = ars_results["ARS_uniform"]
    asi_results = calculate_ASI(sorted(test_angles), angle_confidences)
    asi_score = asi_results["ASI"]
    composite_score = (ear_score + ars_score + asi_score) / 3.0
    
    # Log results
    logger.info(f"Validation Results (Epoch {epoch}):")
    logger.info(f"EAR Score: {ear_score:.4f} (Effective Angular Range: {total_range} degrees)")
    logger.info(f"ARS Score: {ars_score:.4f}")
    logger.info(f"ASI Score: {asi_score:.4f}")
    logger.info(f"Composite Score: {composite_score:.4f}")
    
    # Log confidence information for each angle
    logger.info("Confidence by Angle:")
    for angle in sorted(test_angles):
        conf = avg_confidences.get(angle, 0)
        det_rate = avg_detection_rates.get(angle, 0)
        logger.info(f"  Angle {angle:+3d}°: Confidence={conf:.4f}, Detection Rate={det_rate:.2f}")
    
    # Create validation result plots
    create_validation_plots(results, test_angles, args, epoch_dir, epoch)
    
    # Save patch image
    patch_image = tensor_to_pil(adv_patch)
    patch_image.save(os.path.join(epoch_dir, f"patch.png"))
    
    # Save metrics
    metrics = {
        "epoch": epoch,
        "ear_score": ear_score,
        "ars_score": ars_score,
        "asi_score": asi_score,
        "composite_score": composite_score,
        "effective_angular_range": total_range,
        "ear_ranges": ear_ranges
    }
    
    with open(os.path.join(epoch_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    return metrics


# ==============================================================================
# Saving Functions
# ==============================================================================

def save_patch(adv_patch, args, epoch):
    """
    Save current training patch
    
    Args:
        adv_patch: Patch tensor
        args: Configuration parameters
        epoch: Current epoch
    """
    save_dir = os.path.join(args.output_dir, "patches")
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to PIL image and save
    patch_image = tensor_to_pil(adv_patch)
    patch_image.save(os.path.join(save_dir, f"patch_epoch_{epoch}.png"))
    
    # Save as tensor
    torch.save(adv_patch, os.path.join(save_dir, f"patch_epoch_{epoch}.pt"))


def save_final_patch(adv_patch, args):
    """
    Save final trained patch
    
    Args:
        adv_patch: Patch tensor
        args: Configuration parameters
    """
    # Create final output directory
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    
    # Save as PNG
    patch_image = tensor_to_pil(adv_patch)
    patch_image.save(os.path.join(final_dir, "final_patch.png"))
    
    # Save as high-resolution PNG
    patch_hires = F.interpolate(
        adv_patch.unsqueeze(0),
        size=(1024, 1024),
        mode='bilinear',
        align_corners=False
    ).squeeze(0)
    patch_hires_image = tensor_to_pil(patch_hires)
    patch_hires_image.save(os.path.join(final_dir, "final_patch_hires.png"))
    
    # Save as PyTorch model
    torch.save(adv_patch, os.path.join(final_dir, "final_patch.pt"))
    
    logger.info(f"Final patch saved to {final_dir}")


def plot_training_trends(metrics_history, output_dir):
    """
    Plot training trends
    
    Args:
        metrics_history: Dictionary list containing metrics for each epoch
        output_dir: Output directory
    """
    trends_dir = os.path.join(output_dir, "trends")
    os.makedirs(trends_dir, exist_ok=True)
    
    # Extract data
    epochs = [m['epoch'] for m in metrics_history]
    ear_scores = [m['ear_score'] for m in metrics_history]
    ars_scores = [m['ars_score'] for m in metrics_history]
    asi_scores = [m['asi_score'] for m in metrics_history]
    composite_scores = [m['composite_score'] for m in metrics_history]
    
    # Plot composite score trend
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, composite_scores, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Composite Score', fontsize=12)
    plt.title('Patch Training Progress - Composite Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(trends_dir, 'composite_score_trend.png'), dpi=300)
    plt.close()
    
    # Plot individual metrics trend
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, ear_scores, 'r-o', label='EAR Score', linewidth=2, markersize=6)
    plt.plot(epochs, ars_scores, 'g-s', label='ARS Score', linewidth=2, markersize=6)
    plt.plot(epochs, asi_scores, 'b-^', label='ASI Score', linewidth=2, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Patch Training Progress - Individual Metrics', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(trends_dir, 'individual_metrics_trend.png'), dpi=300)
    plt.close()
    
    # Save trend data
    trend_data = {
        'epochs': epochs,
        'ear_scores': ear_scores,
        'ars_scores': ars_scores, 
        'asi_scores': asi_scores,
        'composite_scores': composite_scores
    }
    
    with open(os.path.join(trends_dir, 'training_trends.json'), 'w') as f:
        json.dump(trend_data, f, indent=2)


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Adversarial Patch Training")
    
    # Basic parameters
    parser.add_argument("--output_dir", type=str, default="adv_patch_results", 
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=None, 
                       help="Random seed")
    
    # Patch related parameters
    parser.add_argument("--patch_size", type=int, default=300,
                       help="Initial patch size")
    parser.add_argument("--patch_width", type=int, default=100,
                       help="Patch width when applied to image")
    parser.add_argument("--patch_height", type=int, default=100,
                       help="Patch height when applied to image")
    parser.add_argument("--patch_type", type=str, default="random",
                       choices=["gray", "random", "black", "white"],
                       help="Initial patch type")
    parser.add_argument("--init_patch", type=str, default=None,
                       help="Initial patch file path (if starting from pre-trained patch)")
    parser.add_argument("--random_position", action="store_true",
                       help="Whether to randomly place patch")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=100,
                       help="Total number of training epochs")
    parser.add_argument("--batch_size_per_epoch", type=int, default=32,
                       help="Number of batches per epoch")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Number of samples per batch")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--save_epochs", type=int, default=10,
                       help="Save results every N epochs")
    parser.add_argument("--save_interval", type=int, default=50,
                       help="Save intermediate results every N batches within epoch")
    parser.add_argument("--use_lr_scheduler", action="store_true",
                        help="Whether to use learning rate scheduler")
    
    # Dataset parameters
    parser.add_argument("--use_dataset", action="store_true", default=True,
                       help="Whether to use real image dataset as background")
    parser.add_argument("--data_dir", type=str, default="/zhaotingfeng/jiwenjun/anglerocl/environment",
                       help="Background image dataset directory")
    
    # YOLO and renderer parameters
    parser.add_argument("--yolo_weights_file", type=str, default="/zhaotingfeng/jiwenjun/anglerocl/checkpoints/yolov5s.pt",
                       help="YOLO weights file path")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Image resolution")
    parser.add_argument("--renderer_fov", type=float, default=60.0,
                       help="Renderer field of view")
    parser.add_argument("--renderer_distance", type=float, default=700.0,
                       help="Camera to object distance in renderer")
    
    # Loss function parameters - consistent with stable diffusion version
    parser.add_argument("--detection_threshold", type=float, default=0.8,
                       help="Detection confidence threshold")
    parser.add_argument("--objective_class_id", type=int, default=11,
                       help="Target class ID")
    
    # Validation parameters
    parser.add_argument("--validation_epochs", type=int, default=10,
                       help="Run validation every N epochs")
    parser.add_argument("--num_validation_examples", type=int, default=10,
                       help="Number of samples to use during validation")
    
    # Mixed precision and accelerator parameters
    parser.add_argument("--mixed_precision", type=str, default="no",
                       choices=["no", "fp16", "bf16"],
                       help="Whether to use mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--report_to", type=str, default="tensorboard",
                       choices=["tensorboard", "wandb", "all"],
                       help="Logging method")
    
    return parser.parse_args()


# ==============================================================================
# Main Function
# ==============================================================================

def main():
    """Main function"""
    args = parse_args()
    
    # Set output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    log_file = setup_logging(args.output_dir)
    
    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)
        seed_everything(args.seed)
    
    # Initialize accelerator
    logging_dir = os.path.join(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    # Log basic information
    logger.info(f"Log file created: {log_file}")
    logger.info(f"Random seed set: {args.seed}")
    logger.info(f"Using device: {accelerator.device}")
    logger.info(f"Mixed precision: {args.mixed_precision}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    
    # Log training angle settings
    train_angles = get_train_angles()
    test_angles = get_test_angles()
    logger.info(f"Training angles: {train_angles}")
    logger.info(f"Test angles: {test_angles}")
    
    # Load YOLO model
    logger.info(f"Loading YOLO model: {args.yolo_weights_file}")
    yolo_model = DetectMultiBackend(args.yolo_weights_file, device=accelerator.device, dnn=False, data=None, fp16=False)
    yolo_model.eval()
    args.n_classes = len(yolo_model.names)
    # Initialize probability extractor
    prob_extractor = MaxProbExtractor(args).to(accelerator.device)
    
    # Create angle renderer
    logger.info("Initializing angle renderer")
    renderer = PerspectiveViewGenerator(
        dev=accelerator.device,
        image_size=args.resolution,
        fov=args.renderer_fov,
        distance=args.renderer_distance
    )
    
    # Initialize patch
    if args.init_patch:
        logger.info(f"Loading initial patch from file: {args.init_patch}")
        adv_patch = load_patch_from_file(args.init_patch, device=accelerator.device)
    else:
        logger.info(f"Creating new {args.patch_type} type patch")
        adv_patch = generate_patch(args.patch_type, size=(args.patch_size, args.patch_size), device=accelerator.device)
    
    adv_patch.requires_grad_(True)
    
    # Set up optimizer
    logger.info(f"Initializing optimizer with learning rate: {args.learning_rate}")
    optimizer = torch.optim.Adam([adv_patch], lr=args.learning_rate)
    
    # Set up learning rate scheduler (optional)
    if args.use_lr_scheduler:
        logger.info("Using ReduceLROnPlateau learning rate scheduler")
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    else:
        scheduler = None
    
    # Prepare dataset (if needed)
    if args.use_dataset and args.data_dir:
        logger.info(f"Loading background image dataset: {args.data_dir}")
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        dataset = ImageFolderDataset(args.data_dir, transform=transform, size=args.resolution)
        data_loader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=4
        )
        logger.info(f"Loaded {len(dataset)} background images")
    else:
        logger.info("Using random noise as background")
        data_loader = None
    
    # Record important parameters for reproducibility
    param_file = os.path.join(args.output_dir, "parameters.json")
    with open(param_file, 'w') as f:
        params = {k: v if not isinstance(v, torch.Tensor) else str(v.shape) for k, v in vars(args).items()}
        json.dump(params, f, indent=2)
    
    # Set loss target function - logic retained from stable diffusion version
    loss_target = args.loss_target if hasattr(args, "loss_target") else "obj*cls"
    if loss_target == "obj":
        args.loss_target = lambda obj, cls: obj
    elif loss_target == "cls":
        args.loss_target = lambda obj, cls: cls
    elif loss_target in {"obj * cls", "obj*cls"}:
        args.loss_target = lambda obj, cls: obj * cls
    else:
        raise NotImplementedError(f"Loss target {loss_target} not been implemented")
    
    # Prepare to save training metrics history
    metrics_history = []
    
    # Training loop
    logger.info(f"Starting training for {args.num_epochs} epochs")
    for epoch in range(args.num_epochs):
        # Train one epoch
        avg_loss, loss_components = train_epoch(
            adv_patch, 
            yolo_model, 
            renderer, 
            prob_extractor, 
            optimizer, 
            args, 
            accelerator, 
            epoch, 
            data_loader
        )
        
        # Update learning rate scheduler
        if scheduler:
            scheduler.step(avg_loss)
        
        # Save intermediate results
        if epoch % args.save_epochs == 0 or epoch == args.num_epochs - 1:
            save_patch(adv_patch, args, epoch)
            logger.info(f"Saved patch for epoch {epoch}")
        
        # Validation
        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            metrics = validate_patch(
                adv_patch, 
                yolo_model, 
                renderer, 
                prob_extractor, 
                args, 
                accelerator, 
                epoch, 
                data_loader
            )
            metrics_history.append(metrics)
        
        # Print current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current learning rate: {current_lr:.6f}")
        
        # Log to TensorBoard or WandB
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                tracker.writer.add_scalar("train/loss", avg_loss, epoch)
                tracker.writer.add_scalar("train/ear_loss", loss_components['ear_loss'], epoch)
                tracker.writer.add_scalar("train/learning_rate", current_lr, epoch)
                
                if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
                    tracker.writer.add_scalar("validation/ear_score", metrics['ear_score'], epoch)
                    tracker.writer.add_scalar("validation/ars_score", metrics['ars_score'], epoch)
                    tracker.writer.add_scalar("validation/asi_score", metrics['asi_score'], epoch)
                    tracker.writer.add_scalar("validation/composite_score", metrics['composite_score'], epoch)
                
                # Add patch image
                patch_np = adv_patch.detach().cpu().numpy().transpose(1, 2, 0)
                tracker.writer.add_image("patch", patch_np, epoch, dataformats='HWC')
            
            elif tracker.name == "wandb":
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/ear_loss": loss_components['ear_loss'],
                        "train/learning_rate": current_lr,
                        "epoch": epoch
                    })
                    
                    if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
                        wandb.log({
                            "validation/ear_score": metrics['ear_score'],
                            "validation/ars_score": metrics['ars_score'],
                            "validation/asi_score": metrics['asi_score'],
                            "validation/composite_score": metrics['composite_score'],
                            "epoch": epoch
                        })
                        
                        # Add patch image
                        patch_image = tensor_to_pil(adv_patch)
                        wandb.log({"patch": wandb.Image(patch_image, caption=f"Epoch {epoch}")})
        
        # Clean up memory
        torch.cuda.empty_cache()
    
    # Save final results
    save_final_patch(adv_patch, args)
    
    # Plot training trends
    if metrics_history:
        plot_training_trends(metrics_history, args.output_dir)
    
    logger.info(f"Training completed, final results saved to {args.output_dir}")
    
    # Close trackers
    accelerator.end_training()


if __name__ == "__main__":
    import random  # Import random module
    
    # Set environment variable for wandb offline mode if needed
    os.environ["WANDB_MODE"] = "offline"
    
    try:
        main()
    except Exception as e:
        logger.exception(f"Error occurred during training: {e}")
        raise