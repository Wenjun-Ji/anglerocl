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

# Training script for angle-robust textual inversion using diffusion models

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

import argparse
import logging
import math
import os
import random
import shutil
import warnings
import json
import numpy as np
import PIL
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

# Custom imports
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from adv_patch_gen.utils.patch import PerspectiveViewGenerator
from adv_patch_gen.utils.loss import MaxProbExtractor
import matplotlib.pyplot as plt

from tools import (
    save_execution_code,
    create_enhanced_single_prompt_analysis,
    create_enhanced_comparison_summary,
    create_enhanced_variance_distribution_plot,
    create_enhanced_all_prompts_performance_plot,
    create_enhanced_summary_text,
    setup_logging,
    analyze_embedding
)
from metrics import (
    calculate_EAR,
    calculate_ARS,
    calculate_ASI
)

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.32.0.dev0")

logger = get_logger(__name__)

# ==============================================================================
# Utility Functions
# ==============================================================================

def seed_everything(seed=11):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_train_angles():
    """Get training angles for rendering."""
    # Training angles: -72°, -54°, -36°, -18°, 0°, 18°, 36°, 54°, 72°
    angles = [-72, -54, -36, -18, 0, 18, 36, 54, 72]
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


def tensor_to_pil(tensor):
    """
    Convert PyTorch tensor to PIL image.
    
    Args:
        tensor: PyTorch tensor of shape [C, H, W] or [1, C, H, W]
    
    Returns:
        PIL Image
    """
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    
    if tensor.dim() != 3:
        raise ValueError(f"Expected tensor with 3 dimensions, got {tensor.dim()}")
    
    # Convert to CPU and numpy
    tensor = tensor.detach().cpu()
    
    # Check value range and normalize to [0, 1]
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
# Validation Functions
# ==============================================================================

# Define prompt templates and corresponding directory names
PROMPTS_AND_DIRS = [
    ("a blue square stop sign", "blue_square_stop_sign"),
    ("a blue square stop sign with 'abcd' on it", "blue_square_stop_sign_with__abcd__on_it"),
    ("a blue square stop sign with 'abcd' on it and checkerboard paint on it", "blue_square_stop_sign_with__abcd__on_it_and_checkerboard_paint_on_it"),
    ("a blue square stop sign with 'hello' on it", "blue_square_stop_sign_with__hello__on_it"),
    ("a blue square stop sign with 'hello' on it and checkerboard paint on it", "blue_square_stop_sign_with__hello__on_it_and_checkerboard_paint_on_it"),
    ("a blue square stop sign with checkerboard paint on it", "blue_square_stop_sign_with_checkerboard_paint_on_it"),
    ("a blue stop sign", "blue_stop_sign"),
    ("a blue stop sign with 'abcd' on it", "blue_stop_sign_with__abcd__on_it"),
    ("a blue stop sign with 'abcd' on it and checkerboard paint on it", "blue_stop_sign_with__abcd__on_it_and_checkerboard_paint_on_it"),
    ("a blue stop sign with 'hello' on it", "blue_stop_sign_with__hello__on_it"),
    ("a blue stop sign with 'hello' on it and checkerboard paint on it", "blue_stop_sign_with__hello__on_it_and_checkerboard_paint_on_it"),
    ("a blue stop sign with checkerboard paint on it", "blue_stop_sign_with_checkerboard_paint_on_it"),
    ("a square stop sign", "square_stop_sign"),
    ("a square stop sign with 'abcd' on it", "square_stop_sign_with__abcd__on_it"),
    ("a square stop sign with 'abcd' on it and checkerboard paint on it", "square_stop_sign_with__abcd__on_it_and_checkerboard_paint_on_it"),
    ("a square stop sign with 'hello' on it", "square_stop_sign_with__hello__on_it"),
    ("a square stop sign with 'hello' on it and checkerboard paint on it", "square_stop_sign_with__hello__on_it_and_checkerboard_paint_on_it"),
    ("a square stop sign with checkerboard paint on it", "square_stop_sign_with_checkerboard_paint_on_it"),
    ("a stop sign", "stop_sign"),
    ("a stop sign with 'abcd' on it", "stop_sign_with__abcd__on_it"),
    ("a stop sign with 'abcd' on it and checkerboard paint on it", "stop_sign_with__abcd__on_it_and_checkerboard_paint_on_it"),
    ("a stop sign with 'hello' on it", "stop_sign_with__hello__on_it"),
    ("a stop sign with 'hello' on it and checkerboard paint on it", "stop_sign_with__hello__on_it_and_checkerboard_paint_on_it"),
    ("a stop sign with 'world' on it", "stop_sign_with__world__on_it"),
    ("a stop sign with 'world' on it and polkadot paint on it", "stop_sign_with__world__on_it_and_polkadot_paint_on_it"),
    ("a stop sign with checkerboard paint on it", "stop_sign_with_checkerboard_paint_on_it"),
    ("a stop sign with polkadot paint on it", "stop_sign_with_polkadot_paint_on_it"),
    ("a triangle stop sign", "triangle_stop_sign"),
    ("a triangle stop sign with 'world' on it", "triangle_stop_sign_with__world__on_it"),
    ("a triangle stop sign with 'world' on it and polkadot paint on it", "triangle_stop_sign_with__world__on_it_and_polkadot_paint_on_it"),
    ("a triangle stop sign with polkadot paint on it", "triangle_stop_sign_with_polkadot_paint_on_it"),
    ("a yellow stop sign", "yellow_stop_sign"),
    ("a yellow stop sign with 'world' on it", "yellow_stop_sign_with__world__on_it"),
    ("a yellow stop sign with 'world' on it and polkadot paint on it", "yellow_stop_sign_with__world__on_it_and_polkadot_paint_on_it"),
    ("a yellow stop sign with polkadot paint on it", "yellow_stop_sign_with_polkadot_paint_on_it"),
    ("a yellow triangle stop sign", "yellow_triangle_stop_sign"),
    ("a yellow triangle stop sign with 'world' on it", "yellow_triangle_stop_sign_with__world__on_it"),
    ("a yellow triangle stop sign with 'world' on it and polkadot paint on it", "yellow_triangle_stop_sign_with__world__on_it_and_polkadot_paint_on_it"),
    ("a yellow triangle stop sign with polkadot paint on it", "yellow_triangle_stop_sign_with_polkadot_paint_on_it"),
]


def log_validation(logger, text_encoder, tokenizer, unet, vae, args, accelerator, weight_dtype, epoch, global_step, placeholder_token_ids, initializer_token_id, initial_placeholder_embeds=None):
    """
    Run validation and log results.
    
    Args:
        logger: Logger instance
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        unet: UNet model
        vae: VAE model
        args: Arguments
        accelerator: Accelerator instance
        weight_dtype: Weight dtype
        epoch: Current epoch
        global_step: Current global step
        placeholder_token_ids: Placeholder token IDs
        initializer_token_id: Initializer token ID
        initial_placeholder_embeds: Initial placeholder embeddings
    
    Returns:
        List of generated images
    """
    logger.info(
        f"Running validation... \n Validating angle robustness on {len(PROMPTS_AND_DIRS)} templates."
    )
    logger.info(f"Current epoch: {epoch}, Current global step: {global_step}")
    logger.info(f"Generating {args.num_samples} samples per prompt for more robust evaluation")
    
    # Create embedding analysis directory structure
    base_embedding_dir = os.path.join(args.output_dir, "embedding_analysis")
    os.makedirs(base_embedding_dir, exist_ok=True)
    epoch_embed_dir = os.path.join(base_embedding_dir, f"epoch_{epoch}_step_{global_step}")
    os.makedirs(epoch_embed_dir, exist_ok=True)
    
    # Create trend directory
    trend_dir = os.path.join(args.output_dir, "validation_trends")
    os.makedirs(trend_dir, exist_ok=True)
    
    # Load historical trend data
    trend_file = os.path.join(trend_dir, "validation_trends.json")
    if os.path.exists(trend_file):
        with open(trend_file, 'r') as f:
            trend_data = json.load(f)
    else:
        trend_data = {
            "steps": [],
            "epochs": [],
            "placeholder_metrics": {
                "avg_confidence": [],
                "ear_scores": [],
                "ars_scores": [],
                "asi_scores": [],
                "composite_scores": []
            },
            "initial_token_metrics": {
                "avg_confidence": [],
                "ear_scores": [],
                "ars_scores": [],
                "asi_scores": [],
                "composite_scores": []
            },
            "clean_metrics": {
                "avg_confidence": [],
                "ear_scores": [],
                "ars_scores": [],
                "asi_scores": [],
                "composite_scores": []
            }
        }
    
    # Get all placeholder token names (convert from IDs to names)
    placeholder_tokens = tokenizer.convert_ids_to_tokens(placeholder_token_ids)
    
    # Analyze each vector's embedding separately
    for i, token_id in enumerate(placeholder_token_ids):
        vector_name = placeholder_tokens[i]
        vector_embed_dir = os.path.join(epoch_embed_dir, f"vector_{i}")
        os.makedirs(vector_embed_dir, exist_ok=True)
        
        # Use initial embedding if available for this vector
        vector_initial_embed = None
        if initial_placeholder_embeds is not None and i < len(initial_placeholder_embeds):
            vector_initial_embed = initial_placeholder_embeds[i]
        
        analyze_embedding(
            logger,
            accelerator.unwrap_model(text_encoder),
            tokenizer,
            [token_id],
            initializer_token_id,
            global_step,
            vector_embed_dir,
            base_embedding_dir,
            vector_name,
            vector_initial_embed
        )
    
    # Create pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    
    # Get training angles
    train_angles = get_train_angles()
    
    # Get test angles (includes intermediate angles not used in training)
    test_angles = get_test_angles()
    
    # Create angle renderer
    renderer = PerspectiveViewGenerator(
        dev=accelerator.device,
        image_size=args.resolution,
        fov=args.renderer_fov,
        distance=args.renderer_distance
    )
    
    # Load YOLO model for detection
    yolo_model = DetectMultiBackend(args.yolo_weights_file, device=accelerator.device, dnn=False, data=None, fp16=False)
    yolo_model.eval()
    prob_extractor = MaxProbExtractor(args).to(accelerator.device)
    
    # Modified result storage structure with multi-sample support
    results = {
        "placeholder_token": {
            "prompts": [],
            "samples": {angle: [] for angle in test_angles},
            "confidences": {angle: [] for angle in test_angles},
            "detection_rates": {angle: [] for angle in test_angles},
            "images": []
        },
        "initial_token": {
            "prompts": [],
            "samples": {angle: [] for angle in test_angles},
            "confidences": {angle: [] for angle in test_angles},
            "detection_rates": {angle: [] for angle in test_angles},
            "images": []
        },
        "clean": {
            "prompts": [],
            "samples": {angle: [] for angle in test_angles},
            "confidences": {angle: [] for angle in test_angles},
            "detection_rates": {angle: [] for angle in test_angles},
            "images": []
        }
    }
    
    # Create validation output directory structure
    validation_dir = os.path.join(args.output_dir, "validation")
    os.makedirs(validation_dir, exist_ok=True)
    epoch_dir = os.path.join(validation_dir, f"epoch_{epoch}_step_{global_step}")
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Record validation parameters
    validation_params = {
        "epoch": epoch,
        "global_step": global_step,
        "num_samples": args.num_samples,
        "num_vectors": args.num_vectors,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "validation_steps": args.validation_steps,
        "validation_epochs": args.validation_epochs,
        "train_batch_size": args.train_batch_size,
    }

    # Save validation parameters to JSON file
    with open(os.path.join(epoch_dir, "validation_params.json"), "w") as f:
        json.dump(validation_params, f, indent=4)
    
    # Create individual prompt directory
    individual_prompts_dir = os.path.join(epoch_dir, "individual_prompts")
    os.makedirs(individual_prompts_dir, exist_ok=True)
    
    # Create samples directory for saving each sample's images
    samples_dir = os.path.join(epoch_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    # Select subset of templates for validation to reduce computation
    num_templates = min(40, len(PROMPTS_AND_DIRS))
    selected_templates = PROMPTS_AND_DIRS[:num_templates]
    
    # For each template, generate three types: with identifier, with initializer word, and without
    for template_idx, (prompt_template, dir_name) in enumerate(selected_templates):
        # Create template directory
        template_dir = os.path.join(epoch_dir, dir_name)
        os.makedirs(template_dir, exist_ok=True)
        
        # Create sample subdirectory
        template_samples_dir = os.path.join(samples_dir, dir_name)
        os.makedirs(template_samples_dir, exist_ok=True)
        
        # 1. Prompt with placeholder identifier
        prompt_with_placeholder = prompt_template.replace("a ", f"a {args.placeholder_token} ")
        results["placeholder_token"]["prompts"].append(prompt_with_placeholder)
        
        # 2. Prompt with initializer word
        prompt_with_initial = prompt_template.replace("a ", f"a {args.initializer_token} ")
        results["initial_token"]["prompts"].append(prompt_with_initial)
        
        # 3. Clean prompt (without any token or initializer word)
        prompt_clean = prompt_template
        results["clean"]["prompts"].append(prompt_clean)
        
        logger.info(f"Validating template {template_idx+1}/{num_templates}: '{prompt_template}'")
        
        # Results for single prompt (generate multiple samples per prompt)
        single_prompt_results = {
            "placeholder_token": {
                "samples": {angle: [] for angle in test_angles},
                "confidences": {angle: None for angle in test_angles},
                "std_devs": {angle: None for angle in test_angles},
                "max_values": {angle: None for angle in test_angles},
                "min_values": {angle: None for angle in test_angles},
                "detection_rates": {angle: None for angle in test_angles},
            },
            "initial_token": {
                "samples": {angle: [] for angle in test_angles},
                "confidences": {angle: None for angle in test_angles},
                "std_devs": {angle: None for angle in test_angles},
                "max_values": {angle: None for angle in test_angles},
                "min_values": {angle: None for angle in test_angles},
                "detection_rates": {angle: None for angle in test_angles},
            },
            "clean": {
                "samples": {angle: [] for angle in test_angles},
                "confidences": {angle: None for angle in test_angles},
                "std_devs": {angle: None for angle in test_angles},
                "max_values": {angle: None for angle in test_angles},
                "min_values": {angle: None for angle in test_angles},
                "detection_rates": {angle: None for angle in test_angles},
            }
        }
        
        # Clear memory after processing each template
        torch.cuda.empty_cache()
        
        # Generate and evaluate for each of the three prompt types
        for key, prompt in [
            ("placeholder_token", prompt_with_placeholder), 
            ("initial_token", prompt_with_initial), 
            ("clean", prompt_clean)
        ]:
            # Store all sample images for this prompt
            sample_images = []
            
            # Generate multiple samples for each prompt
            for sample_idx in range(args.num_samples):
                # Set different random seeds to generate different samples
                sample_seed = args.seed + sample_idx if args.seed is not None else None
                generator = None if sample_seed is None else torch.Generator(device=accelerator.device).manual_seed(sample_seed)
            
                # Generate image
                with torch.autocast(accelerator.device.type):
                    image = pipeline(prompt, num_inference_steps=25, generator=generator).images[0]
                
                # Save sample image
                sample_filename = f"{key}_sample_{sample_idx+1}.png"
                image.save(os.path.join(template_samples_dir, sample_filename))
                
                sample_images.append(image)
                
                # Convert PIL image to PyTorch tensor
                image_tensor = transforms.ToTensor()(image).unsqueeze(0).to(accelerator.device)
                
                # Render views from different angles
                rendered_views, lon_angles, _ = renderer(
                    image_tensor,
                    longitudes=test_angles,
                    latitudes=[0],
                    distance=args.renderer_distance
                )
                
                # [A, 1, 3, H, W] -> [A, 3, H, W]
                rendered_views = rendered_views.squeeze(1)
                
                # Resize to fit YOLO
                resized_views = F.interpolate(
                    rendered_views,
                    size=(640, 640),
                    mode='bilinear',
                    align_corners=False
                )
                
                # YOLO detection
                detection_output = yolo_model(resized_views.float())[0]
                
                # Use NMS to process detection results
                pred = non_max_suppression(
                    detection_output, 
                    conf_thres=0.0001,
                    iou_thres=0.45,
                    classes=[11] if args.objective_class_id == 11 else None,
                    max_det=100
                )

                # Extract confidences
                confidences = []
                for det in pred:
                    if len(det) > 0:
                        # Get highest confidence
                        confidence = float(det[:, 4].max().item())
                    else:
                        confidence = 0.0
                    confidences.append(confidence)

                # Save confidence results for each angle
                for i, angle in enumerate(test_angles):
                    conf = confidences[i]
                    # Add to sample collection
                    single_prompt_results[key]["samples"][angle].append(conf)
                    # Add to global results
                    results[key]["samples"][angle].append(conf)
            
            # Calculate statistics for each angle
            for angle in test_angles:
                # Get all sample results for this prompt at this angle
                sample_values = single_prompt_results[key]["samples"][angle]
                
                # Calculate statistics
                mean_conf = np.mean(sample_values)
                std_dev = np.std(sample_values)
                max_val = np.max(sample_values)
                min_val = np.min(sample_values)
                detection_rate = sum([1 for v in sample_values if v >= args.detection_threshold]) / len(sample_values)
                
                # Store statistics
                single_prompt_results[key]["confidences"][angle] = mean_conf
                single_prompt_results[key]["std_devs"][angle] = std_dev
                single_prompt_results[key]["max_values"][angle] = max_val
                single_prompt_results[key]["min_values"][angle] = min_val
                single_prompt_results[key]["detection_rates"][angle] = detection_rate
                
                # Add to global results (average)
                results[key]["confidences"][angle].append(mean_conf)
                results[key]["detection_rates"][angle].append(detection_rate)
            
            # Store first image from samples (or choose most representative image)
            if sample_images:
                # Save a representative image
                image_filename = f"{key}.png"
                sample_images[0].save(os.path.join(template_dir, image_filename))
                results[key]["images"].append(sample_images[0])
        
        # Generate enhanced analysis plot for this individual prompt (includes std dev and max/min values)
        create_enhanced_single_prompt_analysis(
            single_prompt_results, 
            train_angles,
            test_angles, 
            args, 
            prompt_template,
            os.path.join(individual_prompts_dir, f"prompt_{template_idx+1}_{dir_name}.png")
        )
    
    # Calculate average, std dev, max and min for each angle
    avg_results = {}
    for key in ["placeholder_token", "initial_token", "clean"]:
        avg_results[key] = {
            "avg_confidences": {angle: sum(values)/len(values) for angle, values in results[key]["confidences"].items()},
            "avg_detection_rates": {angle: sum(values)/len(values) for angle, values in results[key]["detection_rates"].items()},
            # Add sample statistics
            "all_samples": {angle: results[key]["samples"][angle] for angle in test_angles},
            # Calculate statistics for all samples
            "std_all_samples": {angle: np.std(results[key]["samples"][angle]) for angle in test_angles},
            "max_all_samples": {angle: np.max(results[key]["samples"][angle]) for angle in test_angles},
            "min_all_samples": {angle: np.min(results[key]["samples"][angle]) for angle in test_angles},
            # Calculate statistics for average of each prompt
            "std_avg_confidences": {angle: np.std(results[key]["confidences"][angle]) for angle in test_angles},
            "max_avg_confidences": {angle: np.max(results[key]["confidences"][angle]) for angle in test_angles},
            "min_avg_confidences": {angle: np.min(results[key]["confidences"][angle]) for angle in test_angles},
            "std_detection_rates": {angle: np.std(results[key]["detection_rates"][angle]) for angle in test_angles},
            "max_detection_rates": {angle: np.max(results[key]["detection_rates"][angle]) for angle in test_angles},
            "min_detection_rates": {angle: np.min(results[key]["detection_rates"][angle]) for angle in test_angles}
        }
    
    # Calculate key metrics and update trend data
    metric_keys = ["placeholder_token", "initial_token", "clean"]
    metric_map = {
        "placeholder_token": "placeholder_metrics",
        "initial_token": "initial_token_metrics",
        "clean": "clean_metrics"
    }
    
    for key in metric_keys:
        # Calculate angle robustness metrics
        all_confidences = []
        for angle_values in results[key]["samples"].values():
            all_confidences.extend(angle_values)
        
        # Average confidence
        avg_confidence = np.mean(all_confidences)
        
        # Calculate EAR, ARS, ASI using evaluation metric functions
        angle_confidences = [avg_results[key]["avg_confidences"][angle] for angle in sorted(test_angles)]
        
        # Calculate EAR
        total_range, ear_ranges = calculate_EAR(sorted(test_angles), angle_confidences, threshold=0.5)
        ear_score = total_range / 180.0  # Normalize to [0,1]
        
        # Calculate ARS
        ars_results = calculate_ARS(sorted(test_angles), angle_confidences)
        ars_score = ars_results["ARS_uniform"]
        
        # Calculate ASI
        asi_results = calculate_ASI(sorted(test_angles), angle_confidences)
        asi_score = asi_results["ASI"]
        
        # Composite score
        composite_score = (ear_score + ars_score + asi_score) / 3.0
        
        # Record to trend data
        metric_key = metric_map[key]
        trend_data[metric_key]["avg_confidence"].append(avg_confidence)
        trend_data[metric_key]["ear_scores"].append(ear_score)
        trend_data[metric_key]["ars_scores"].append(ars_score)
        trend_data[metric_key]["asi_scores"].append(asi_score)
        trend_data[metric_key]["composite_scores"].append(composite_score)
    
    # Record step and epoch
    trend_data["steps"].append(global_step)
    trend_data["epochs"].append(epoch)
    
    # Save trend data
    with open(trend_file, 'w') as f:
        json.dump(trend_data, f, indent=2)
    
    # Plot trend graphs
    plot_validation_trends(trend_data, trend_dir, global_step, accelerator)
    
    # Create enhanced comparison analysis plot
    create_enhanced_comparison_summary(avg_results, results, train_angles, test_angles, args, os.path.join(epoch_dir, "comparison_summary.png"))
    
    # Create variance distribution plot
    variance_plot_path = os.path.join(epoch_dir, "variance_distribution.png")
    create_enhanced_variance_distribution_plot(results, test_angles, args, variance_plot_path, args.num_samples)
    
    # Create all prompts performance plot
    all_prompts_path = os.path.join(epoch_dir, "all_prompts_performance.png")
    create_enhanced_all_prompts_performance_plot(results, test_angles, args, all_prompts_path)
    
    # Log to logger
    summary_text = create_enhanced_summary_text(avg_results, train_angles, test_angles, args, args.num_samples)
    logger.info(f"Validation Results (Epoch {epoch}):")
    logger.info(f"With {args.placeholder_token} (avg across {args.num_samples} samples per prompt):")
    logger.info(f"Average Confidence: {avg_results['placeholder_token']['avg_confidences']}")
    logger.info(f"Average Detection Rate: {avg_results['placeholder_token']['avg_detection_rates']}")
    logger.info(f"With {args.initializer_token}:")
    logger.info(f"Average Confidence: {avg_results['initial_token']['avg_confidences']}")
    logger.info(f"Average Detection Rate: {avg_results['initial_token']['avg_detection_rates']}")
    logger.info(f"Clean prompt:")
    logger.info(f"Average Confidence: {avg_results['clean']['avg_confidences']}")
    logger.info(f"Average Detection Rate: {avg_results['clean']['avg_detection_rates']}")
    logger.info(f"Improvement Summary:\n{summary_text}")
    
    # Log images and performance metrics if using trackers
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            # Log comparison images
            for key, data in results.items():
                sample_images = data["images"][:min(4, len(data["images"]))]
                if sample_images:
                    np_images = np.stack([np.asarray(img) for img in sample_images])
                    tracker.writer.add_images(f"validation/{key}", np_images, epoch, dataformats="NHWC")
            
            # Log performance metrics
            for key in ["placeholder_token", "initial_token", "clean"]:
                for angle, value in avg_results[key]["avg_confidences"].items():
                    tracker.writer.add_scalar(f"validation/{key}/confidence_{angle}", value, epoch)
                for angle, value in avg_results[key]["avg_detection_rates"].items():
                    tracker.writer.add_scalar(f"validation/{key}/detection_rate_{angle}", value, epoch)
                # Log sample standard deviations
                for angle, value in avg_results[key]["std_all_samples"].items():
                    tracker.writer.add_scalar(f"validation/{key}/std_all_samples_{angle}", value, epoch)
                # Log variations between prompts
                for angle, value in avg_results[key]["std_avg_confidences"].items():
                    tracker.writer.add_scalar(f"validation/{key}/std_prompt_confidence_{angle}", value, epoch)
            
            # Add enhanced visualization charts
            tracker.writer.add_image(f"validation/all_prompts_performance", 
                                     np.array(Image.open(all_prompts_path)), 
                                     epoch, dataformats="HWC")
            
            # Add individual prompt analysis plots
            sample_prompt_analyses = sorted(os.listdir(individual_prompts_dir))[:5]
            for i, analysis_file in enumerate(sample_prompt_analyses):
                img = Image.open(os.path.join(individual_prompts_dir, analysis_file))
                img_np = np.array(img)
                tracker.writer.add_image(f"validation/prompt_analysis_{i}", img_np, epoch, dataformats="HWC")
            
        if tracker.name == "wandb":
            # Log comparison images
            for key, name in [
                ("placeholder_token", f"with_{args.placeholder_token}"), 
                ("initial_token", f"with_{args.initializer_token}"), 
                ("clean", "clean")
            ]:
                sample_images = results[key]["images"][:min(4, len(results[key]["images"]))]
                if sample_images:
                    tracker.log({
                        f"validation/{name}": [
                            wandb.Image(image, caption=f"{i}: {results[key]['prompts'][i]}") 
                            for i, image in enumerate(sample_images)
                        ]
                    })
            
            # Log performance metrics
            wandb_metrics = {}
            for key, name in [
                ("placeholder_token", f"with_{args.placeholder_token}"), 
                ("initial_token", f"with_{args.initializer_token}"), 
                ("clean", "clean")
            ]:
                for angle, value in avg_results[key]["avg_confidences"].items():
                    wandb_metrics[f"validation/{name}/confidence_{angle}"] = value
                for angle, value in avg_results[key]["avg_detection_rates"].items():
                    wandb_metrics[f"validation/{name}/detection_rate_{angle}"] = value
                # Log sample standard deviations
                for angle, value in avg_results[key]["std_all_samples"].items():
                    wandb_metrics[f"validation/{name}/std_all_samples_{angle}"] = value
                # Log variations between prompts
                for angle, value in avg_results[key]["std_avg_confidences"].items():
                    wandb_metrics[f"validation/{name}/std_prompt_confidence_{angle}"] = value
            
            tracker.log(wandb_metrics)
            
            # Log enhanced visualization charts
            tracker.log({
                "comparison_summary": wandb.Image(os.path.join(epoch_dir, "comparison_summary.png")),
                "validation/variance_distribution": wandb.Image(variance_plot_path),
                "validation/all_prompts_performance": wandb.Image(all_prompts_path)
            })
            
            # Log individual prompt analysis plots
            sample_prompt_analyses = sorted(os.listdir(individual_prompts_dir))[:5]
            for i, analysis_file in enumerate(sample_prompt_analyses):
                tracker.log({
                    f"validation/prompt_analysis_{i}": wandb.Image(os.path.join(individual_prompts_dir, analysis_file))
                })
    
    del pipeline
    torch.cuda.empty_cache()
    
    # Return all generated images
    all_images = []
    for key in ["placeholder_token", "initial_token", "clean"]:
        all_images.extend(results[key]["images"])
    return all_images


def plot_validation_trends(trend_data, trend_dir, current_step, accelerator):
    """Plot validation metric trends."""
    steps = trend_data["steps"]
    
    # 1. Plot composite score trend
    plt.figure(figsize=(12, 8))
    
    plt.plot(steps, trend_data["placeholder_metrics"]["composite_scores"], 
             'b-o', label=f'With placeholder token', linewidth=2, markersize=8)
    plt.plot(steps, trend_data["initial_token_metrics"]["composite_scores"], 
             'g-s', label=f'With initializer token', linewidth=2, markersize=8)
    plt.plot(steps, trend_data["clean_metrics"]["composite_scores"], 
             'r-^', label='Clean (no token)', linewidth=2, markersize=8)
    
    plt.xlabel('Global Step', fontsize=14)
    plt.ylabel('Composite Score', fontsize=14)
    plt.title('Angle Robustness Composite Score Trend', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(trend_dir, f'composite_score_trend_{current_step}.png'), dpi=300)
    plt.close()
    
    # 2. Plot individual metric trends
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    metrics = [
        ("avg_confidence", "Average Confidence"),
        ("ear_scores", "EAR Score"),
        ("ars_scores", "ARS Score"),
        ("asi_scores", "ASI Score")
    ]
    
    for idx, (metric_key, metric_name) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        ax.plot(steps, trend_data["placeholder_metrics"][metric_key], 
                'b-o', label=f'With placeholder token', linewidth=2)
        ax.plot(steps, trend_data["initial_token_metrics"][metric_key], 
                'g-s', label=f'With initializer token', linewidth=2)
        ax.plot(steps, trend_data["clean_metrics"][metric_key], 
                'r-^', label='Clean (no token)', linewidth=2)
        
        ax.set_xlabel('Global Step', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} Trend', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(trend_dir, f'individual_metrics_trend_{current_step}.png'), dpi=300)
    plt.close()
    
    # 3. Plot performance improvement graph
    plt.figure(figsize=(12, 8))
    
    # Calculate improvement percentage relative to clean
    placeholder_improvement = []
    initializer_improvement = []
    
    for i in range(len(steps)):
        clean_score = trend_data["clean_metrics"]["composite_scores"][i]
        placeholder_score = trend_data["placeholder_metrics"]["composite_scores"][i]
        initializer_score = trend_data["initial_token_metrics"]["composite_scores"][i]
        if clean_score > 0:
            placeholder_improvement.append((placeholder_score - clean_score) / clean_score * 100)
            initializer_improvement.append((initializer_score - clean_score) / clean_score * 100)
        else:
            placeholder_improvement.append(0)
            initializer_improvement.append(0)
    
    plt.plot(steps, placeholder_improvement, 'b-o', label='Placeholder token improvement', linewidth=2)
    plt.plot(steps, initializer_improvement, 'g-s', label='Initializer token improvement', linewidth=2)
    
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Global Step', fontsize=14)
    plt.ylabel('Improvement over Clean (%)', fontsize=14)
    plt.title('Performance Improvement Trend', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(trend_dir, f'improvement_trend_{current_step}.png'), dpi=300)
    plt.close()
    
    # 4. Add trend plots to TensorBoard and WandB
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            # Add trend plots to TensorBoard
            composite_img = plt.imread(os.path.join(trend_dir, f'composite_score_trend_{current_step}.png'))
            individual_img = plt.imread(os.path.join(trend_dir, f'individual_metrics_trend_{current_step}.png'))
            improvement_img = plt.imread(os.path.join(trend_dir, f'improvement_trend_{current_step}.png'))
            
            tracker.writer.add_image("validation/composite_score_trend", 
                                   np.array(composite_img), 
                                   current_step, dataformats="HWC")
            tracker.writer.add_image("validation/individual_metrics_trend", 
                                   np.array(individual_img), 
                                   current_step, dataformats="HWC")
            tracker.writer.add_image("validation/improvement_trend", 
                                   np.array(improvement_img), 
                                   current_step, dataformats="HWC")
            
        elif tracker.name == "wandb":
            # Add trend plots to WandB
            tracker.log({
                "validation/composite_score_trend": wandb.Image(os.path.join(trend_dir, f'composite_score_trend_{current_step}.png')),
                "validation/individual_metrics_trend": wandb.Image(os.path.join(trend_dir, f'individual_metrics_trend_{current_step}.png')),
                "validation/improvement_trend": wandb.Image(os.path.join(trend_dir, f'improvement_trend_{current_step}.png'))
            })


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    """Save learned embeddings."""
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


# ==============================================================================
# Argument Parsing
# ==============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training script for angle-robust textual inversion.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1950,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stable-diffusion-v1-5/stable-diffusion-v1-5",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default="<angle-robust>",
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", 
        type=str, 
        default="robust",
        help="A token to use as initializer word."
    )
    parser.add_argument("--repeats", type=int, default=50, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="runs/text-inversion-model1/${timestamp}",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=195000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        default=True,
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", 
        type=int, 
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and Nvidia Ampere GPU or Intel Gen 4 Xeon (and later) ."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=1950,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500000000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", 
        action="store_true", 
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_true",
        help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.",
    )
    
    # Custom arguments for angle-robust training
    parser.add_argument(
        "--yolo_weights_file",
        type=str,
        default="/zhaotingfeng/jiwenjun/anglerocl/checkpoints/yolov5s.pt",
        help="Path to the YOLOv5 weights file.",
    )
    parser.add_argument(
        "--renderer_fov",
        type=float,
        default=60.0,
        help="Field of view for the perspective renderer.",
    )
    parser.add_argument(
        "--renderer_distance",
        type=float,
        default=700.0,
        help="Distance from the camera to the object in the perspective renderer.",
    )
    parser.add_argument(
        "--objective_class_id",
        type=int,
        default=11,
        help="The class ID of the object to be detected (e.g., stop sign).",
    )
    parser.add_argument(
        "--detection_threshold",
        type=float,
        default=0.8,
        help="The threshold for object detection confidence.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of denoising steps for the diffusion model.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to generate for each prompt during validation.",
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


# ==============================================================================
# Training Data
# ==============================================================================

# NDDA stop sign prompt templates
ndda_stopsign_template = [
    "a {} blue square stop sign",
    "a {} blue square stop sign with 'abcd' on it",
    "a {} blue square stop sign with 'abcd' on it and checkerboard paint on it",
    "a {} blue square stop sign with 'hello' on it",
    "a {} blue square stop sign with 'hello' on it and checkerboard paint on it",
    "a {} blue square stop sign with checkerboard paint on it",
    "a {} blue stop sign",
    "a {} blue stop sign with 'abcd' on it",
    "a {} blue stop sign with 'abcd' on it and checkerboard paint on it",
    "a {} blue stop sign with 'hello' on it",
    "a {} blue stop sign with 'hello' on it and checkerboard paint on it",
    "a {} blue stop sign with checkerboard paint on it",
    "a {} square stop sign",
    "a {} square stop sign with 'abcd' on it",
    "a {} square stop sign with 'abcd' on it and checkerboard paint on it",
    "a {} square stop sign with 'hello' on it",
    "a {} square stop sign with 'hello' on it and checkerboard paint on it",
    "a {} square stop sign with checkerboard paint on it",
    "a {} stop sign",
    "a {} stop sign with 'abcd' on it",
    "a {} stop sign with 'abcd' on it and checkerboard paint on it",
    "a {} stop sign with 'hello' on it",
    "a {} stop sign with 'hello' on it and checkerboard paint on it",
    "a {} stop sign with 'world' on it",
    "a {} stop sign with 'world' on it and polkadot paint on it",
    "a {} stop sign with checkerboard paint on it",
    "a {} stop sign with polkadot paint on it",
    "a {} triangle stop sign",
    "a {} triangle stop sign with 'world' on it",
    "a {} triangle stop sign with 'world' on it and polkadot paint on it",
    "a {} triangle stop sign with polkadot paint on it",
    "a {} yellow stop sign",
    "a {} yellow stop sign with 'world' on it",
    "a {} yellow stop sign with 'world' on it and polkadot paint on it",
    "a {} yellow stop sign with polkadot paint on it",
    "a {} yellow triangle stop sign",
    "a {} yellow triangle stop sign with 'world' on it",
    "a {} yellow triangle stop sign with 'world' on it and polkadot paint on it",
    "a {} yellow triangle stop sign with polkadot paint on it",
]


class TextPromptDataset(Dataset):
    """Simplified dataset class for handling text prompts with {} placeholders."""
    
    def __init__(
        self,
        prompts,
        tokenizer,
        placeholder_token="*",
        repeats=50,
    ):
        self.tokenizer = tokenizer
        self.placeholder_token = placeholder_token
        self.repeats = repeats
        
        # Store original prompts
        self.base_prompts = prompts
        
        # Create prompts with placeholder token
        self.augmented_prompts = []
        for prompt in self.base_prompts:
            # Use format method to fill {} placeholder
            augmented_prompt = prompt.format(self.placeholder_token)
            self.augmented_prompts.append(augmented_prompt)
        
    def __len__(self):
        return len(self.augmented_prompts) * self.repeats

    def __getitem__(self, i):
        # Calculate actual prompt index
        prompt_idx = i // self.repeats
        
        # Get augmented prompt
        text = self.augmented_prompts[prompt_idx]
        
        print(text)
        
        # Tokenize text
        tokenized_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        
        print(tokenized_text)
        
        return {
            "input_ids": tokenized_text,
            "original_prompt": self.base_prompts[prompt_idx],
            "augmented_prompt": text,
        }


# ==============================================================================
# Main Training Function
# ==============================================================================

def main():
    args = parse_args()

    # Handle timestamp in output directory
    if "${timestamp}" in args.output_dir:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = args.output_dir.replace("${timestamp}", timestamp)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    save_execution_code(
        output_dir=args.output_dir,
        script_path=__file__,
        additional_files=[
            "tools.py",
            "metrics.py",
        ]
    )
    
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    
    logger.info(f"Output directory set to: {args.output_dir}")
    # Setup logging to both console and file
    log_file = setup_logging(logging_dir)
    logger.info(f"Log file created at: {log_file}")

    # Disable AMP for MPS
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now
    if args.seed is not None:
        set_seed(args.seed)
        seed_everything(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    if args.tokenizer_name:
        tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
    elif args.pretrained_model_name_or_path:
        tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    
    # Load scheduler and models
    noise_scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.pretrained_model_name_or_path, 
        subfolder="scheduler",
        algorithm_type="dpmsolver++",
        solver_order=2
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )
    
    # Load YOLO model and setup renderer
    device = accelerator.device
    yolo_model = DetectMultiBackend(args.yolo_weights_file, device=device, dnn=False, data=None, fp16=False)
    yolo_model.eval()
    args.n_classes = len(yolo_model.names)
    prob_extractor = MaxProbExtractor(args).to(device)
    renderer = PerspectiveViewGenerator(
        dev=device,
        image_size=args.resolution,
        fov=args.renderer_fov,
        distance=args.renderer_distance
    )
    logger.info(f"Successfully loaded YOLO model: {args.yolo_weights_file}")
    
    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.placeholder_token]

    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    # Add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = tokenizer.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()
    
    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    
    # Calculate trainable and total parameters
    trainable_params = sum(p.numel() for p in text_encoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in text_encoder.parameters())
    
    # More robust detection of trained components
    trained_components = []
    trainable_param_count = {}

    # Recursively traverse model structure and count trainable parameters for each submodule
    def count_trainable_params(module, name=''):
        trainable = 0
        for param in module.parameters(recurse=False):
            if param.requires_grad:
                trainable += param.numel()
        
        # Record module if it has trainable parameters
        if trainable > 0:
            trainable_param_count[name] = trainable
        
        # Recursively check all submodules
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            count_trainable_params(child, full_name)

    # Execute statistics
    count_trainable_params(text_encoder)

    # Sort components by parameter count
    sorted_components = sorted(trainable_param_count.items(), key=lambda x: x[1], reverse=True)

    # Select components with the most parameters or large proportion
    significant_components = []
    threshold = 0.01  # Parameter proportion threshold, record if > 1%
    for name, count in sorted_components:
        percentage = count / trainable_params * 100
        if percentage > threshold:
            significant_components.append(f"{name} ({percentage:.1f}%)")
        
        # Show at most top 5 main components
        if len(significant_components) >= 5:
            break
    
    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode
        unet.train()
        text_encoder.gradient_checkpointing_enable()
        unet.enable_gradient_checkpointing()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, "
                    "please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )
    
    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        text_encoder.get_input_embeddings().parameters(),  # Only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    
    # Create generator for DataLoader on CPU
    dataloader_generator = torch.Generator()
    if args.seed is not None:
        dataloader_generator.manual_seed(args.seed)

    # Create generator for CUDA operations on GPU (if needed)
    cuda_generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        cuda_generator.manual_seed(args.seed)
    
    # Dataset and DataLoaders creation
    train_dataset = TextPromptDataset(
        prompts=ndda_stopsign_template,
        tokenizer=tokenizer,
        placeholder_token=" ".join(tokenizer.convert_ids_to_tokens(placeholder_token_ids)),
        repeats=args.repeats,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers, generator=dataloader_generator
    )
    if args.validation_epochs is not None:
        warnings.warn(
            f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
            " Deprecated validation_epochs in favor of `validation_steps`"
            f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
            FutureWarning,
            stacklevel=2,
        )
        args.validation_steps = args.validation_epochs * len(train_dataset)
    
    # Scheduler and math around the number of training steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    text_encoder.train()
    # Prepare everything with our `accelerator`
    text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        text_encoder, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    # Move vae and unet to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration
    # The trackers initializes automatically on the main process
    if accelerator.is_main_process:
        accelerator.init_trackers("", config=vars(args))
    
    # Setup loss target for MaxProbExtractor
    loss_target = args.loss_target if hasattr(args, "loss_target") else "obj*cls"
    if loss_target == "obj":
        args.loss_target = lambda obj, cls: obj
    elif loss_target == "cls":
        args.loss_target = lambda obj, cls: cls
    elif loss_target in {"obj * cls", "obj*cls"}:
        args.loss_target = lambda obj, cls: obj * cls
    else:
        raise NotImplementedError(f"Loss target {loss_target} not been implemented")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    train_angles = get_train_angles()
    
    # Improved training start log
    logger.info("***** Running training *****")
    logger.info(f"  Total train steps = {args.max_train_steps}")
    logger.info(f"  Total train epochs = {args.num_train_epochs}")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Vector length = {args.num_vectors}")
    logger.info(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    logger.info(f"  Train angle: {train_angles}")

    if significant_components:
        logger.info(f"  Main trainable components: {', '.join(significant_components)}")
    else:
        logger.info(f"  No significant trainable components found")
        
    global_step = 0
    first_epoch = 0
    
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=True,
    )

    # Keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()
    
    # Save initial embedding state before training starts and perform comprehensive analysis
    logger.info("Saving initial embedding state for later comparison")
    with torch.no_grad():
        # Get initial embeddings
        unwrapped_model = accelerator.unwrap_model(text_encoder)
        
        # Save initial embeddings for all vectors
        initial_placeholder_embeds = []
        for token_id in placeholder_token_ids:
            initial_embed = unwrapped_model.get_input_embeddings().weight[token_id].clone().detach()
            initial_placeholder_embeds.append(initial_embed)
        
        # Create initial analysis directory
        initial_embed_dir = os.path.join(args.output_dir, "embedding_analysis", "initial")
        os.makedirs(initial_embed_dir, exist_ok=True)
        
        # Create separate analysis for each vector
        for i, (token_id, initial_embed) in enumerate(zip(placeholder_token_ids, initial_placeholder_embeds)):
            vector_name = tokenizer.convert_ids_to_tokens([token_id])[0]
            vector_embed_dir = os.path.join(initial_embed_dir, f"vector_{i}")
            os.makedirs(vector_embed_dir, exist_ok=True)
            
            # Create histogram for initial embedding of each vector
            plt.figure(figsize=(10, 6))
            plt.hist(initial_embed.cpu().numpy(), bins=50, alpha=0.7, color='blue')
            plt.title(f'Initial {vector_name} Embedding Distribution')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(vector_embed_dir, "initial_embedding_distribution.png"))
            plt.close()
            
            # Perform analysis for single vector
            analyze_embedding(
                logger,
                unwrapped_model,
                tokenizer,
                [token_id],
                initializer_token_id,
                0,  # Step 0 indicates initial state
                vector_embed_dir,
                os.path.join(args.output_dir, "embedding_analysis"),
                vector_name,
                initial_embed
            )
            
            logger.info(f"Initial embedding {i} saved with shape: {initial_embed.shape}")

    # Initialize loss window variables
    loss_window = []
    window_size = 100
    epoch_losses = []
    
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder.train()

        # Reset loss list at the start of each epoch
        epoch_losses = []
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(text_encoder):
                # Get the text embedding for conditioning
                logger.info(f"step{step} prompt: {batch['original_prompt']}")
                encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

                # Create unconditional embeddings for classifier-free guidance
                uncond_input = tokenizer(
                    [""] * batch["input_ids"].shape[0], 
                    padding="max_length", 
                    max_length=batch["input_ids"].shape[1], 
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                uncond_embeddings = text_encoder(uncond_input)[0].to(dtype=weight_dtype)
                
                # Concatenate for classifier-free guidance
                text_embeddings = torch.cat([uncond_embeddings, encoder_hidden_states])
                
                # Set up scheduler
                noise_scheduler.set_timesteps(args.num_inference_steps)  # Use fewer steps for faster generation
                guidance_scale = 7.5
                
                # Initialize random latents
                bsz = batch["input_ids"].shape[0]
                latents = torch.randn(
                    (bsz, unet.config.in_channels, args.resolution // 8, args.resolution // 8),
                    device=accelerator.device,
                    generator=cuda_generator,
                    dtype=weight_dtype
                )
                latents = latents * noise_scheduler.init_noise_sigma
                
                # Denoising loop
                for t in noise_scheduler.timesteps:
                    # Expand latents for classifier-free guidance
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
                    
                    # Predict noise residual
                    noise_pred = unet(
                        latent_model_input, 
                        t, 
                        encoder_hidden_states=text_embeddings
                    ).sample
                    
                    # Apply classifier-free guidance
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    
                    # Update latents
                    latents = noise_scheduler.step(noise_pred, t, latents).prev_sample
                
                # Decode latents to images
                latents = 1 / 0.18215 * latents
                adv_images = vae.decode(latents).sample

                # Normalize images to [0,1] range
                adv_images_norm = (adv_images + 1) / 2
                adv_images_norm = adv_images_norm.clamp(0, 1)  # [B,C,H,W]
                
                # Render views from different angles
                rendered_views, lon_angles, _ = renderer(  # [V,B,C,H,W]
                    adv_images_norm.float(),
                    longitudes=train_angles,
                    latitudes=[0],
                    distance=args.renderer_distance
                )
                # [V,B,C,H,W]->[V*B,C,H,W]
                V, B, C, H, W = rendered_views.shape
                rendered_views = rendered_views.view(V * B, C, H, W)

                resized_views = F.interpolate(  # [V*B,C,H,W]->[V*B,C,640,640]
                    rendered_views, 
                    size=(640, 640),
                    mode='bilinear',
                    align_corners=False
                ).to(dtype=weight_dtype)
                
                # YOLO detection
                detection_output = yolo_model(resized_views.float())[0]  # [V*B, 25200, 85]
                max_prob = prob_extractor(detection_output)  # [V*B]
                
                # Use simplified EAR loss function
                loss = calculate_ear_loss(max_prob, train_angles, threshold=args.detection_threshold)

                accelerator.backward(loss)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((len(tokenizer),), dtype=torch.bool)
                index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
                
                with torch.no_grad():
                    accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                # Update loss records
                current_lr = lr_scheduler.get_last_lr()[0]
                loss_value = loss.detach().item()
                
                # Update moving window statistics
                loss_window.append(loss_value)
                
                # Update current epoch statistics
                epoch_losses.append(loss_value)
                
                # Keep window size fixed
                if len(loss_window) > window_size:
                    loss_window.pop(0)
                
                # Calculate moving average
                avg_loss = sum(loss_window) / len(loss_window)
                
                # Log output
                total_epochs = args.num_train_epochs
                current_epoch = epoch + 1
                progress_percentage = (global_step / args.max_train_steps) * 100
                
                if global_step % 1 == 0:
                    logger.info(
                        f"Step {global_step}/{args.max_train_steps} ({progress_percentage:.1f}%) | "
                        f"Epoch {current_epoch}/{total_epochs} | "
                        f"Loss: {loss_value:.4f} (Avg: {avg_loss:.4f}) | "
                        f"LR: {current_lr:.6f}"
                    )
                
                # Log to wandb/tensorboard
                logs = {
                    "loss": loss_value,
                    "avg_loss": avg_loss,
                    "lr": current_lr,
                }
                accelerator.log(logs, step=global_step)

                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    weight_name = (
                        f"learned_embeds-steps-{global_step}.bin"
                        if args.no_safe_serialization
                        else f"learned_embeds-steps-{global_step}.safetensors"
                    )
                    save_path = os.path.join(args.output_dir, weight_name)
                    save_progress(
                        text_encoder,
                        placeholder_token_ids,
                        accelerator,
                        args,
                        save_path,
                        safe_serialization=not args.no_safe_serialization,
                    )

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # Before saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # Before we save the new checkpoint, we need to have at most `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if global_step % args.validation_steps == 0:
                        # Correctly calculate epoch based on global step and gradient accumulation
                        true_epoch = global_step // num_update_steps_per_epoch
                        
                        logger.info(f"Running validation at global_step={global_step}, calculated epoch={true_epoch}")
                        
                        # Pass epoch and global_step to validation function
                        images = log_validation(
                            logger,
                            text_encoder, 
                            tokenizer, 
                            unet, 
                            vae, 
                            args, 
                            accelerator, 
                            weight_dtype, 
                            true_epoch, 
                            global_step,
                            placeholder_token_ids,
                            initializer_token_id,
                            initial_placeholder_embeds
                        )
                
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Clear memory after each step
            torch.cuda.empty_cache()

            if global_step >= args.max_train_steps:
                break
                
        # Statistics at the end of each epoch
        if len(epoch_losses) > 0:
            epoch_avg_loss = sum(epoch_losses) / len(epoch_losses)
            
            logger.info(
                f"Epoch {current_epoch} Summary | "
                f"Avg Loss: {epoch_avg_loss:.4f}"
            )
            
            # Log to wandb/tensorboard
            epoch_logs = {
                "epoch/loss": epoch_avg_loss,
            }
            accelerator.log(epoch_logs, step=global_step)
        
        # Release memory
        torch.cuda.empty_cache()

    # Create the pipeline using the trained modules and save it
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_full_model = args.save_as_full_pipeline
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder),
                vae=vae,
                unet=unet,
                tokenizer=tokenizer,
            )
            pipeline.save_pretrained(args.output_dir)
        # Save the newly trained embeddings
        weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(
            text_encoder,
            placeholder_token_ids,
            accelerator,
            args,
            save_path,
            safe_serialization=not args.no_safe_serialization,
        )

    accelerator.end_training()

if __name__ == "__main__":
    main()