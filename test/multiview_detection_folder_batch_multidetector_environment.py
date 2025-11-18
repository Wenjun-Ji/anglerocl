#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-folder Batch Processing Script with Environment Compositing and AASR Analysis

This script extends the original perspective view detection script to:
1. Process multiple input folders
2. Analyze stop sign detection across different angles using perspective views
3. Composite rendered images onto real-world backgrounds
4. Support multiple detectors (YOLOv3/YOLOv5/YOLOv10/Faster R-CNN/DETR)
5. Calculate AASR (Angular Attack Success Rate) metrics
6. Group results by modification category (S: Shape, C: Color, T: Text, P: Pattern)
"""

import sys
import os
# Get absolute path of project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Add project root to Python path
sys.path.insert(0, project_root)

import glob
import argparse
import subprocess
from tqdm import tqdm
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
from matplotlib.ticker import MaxNLocator


# Functions for loading and updating processed folders
def load_processed_folders(processed_file_path):
    """Load list of already processed folders"""
    try:
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            # Read and strip whitespace from each line
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Warning: Processed folders record {processed_file_path} does not exist, creating new file.")
        # Create file if it doesn't exist
        with open(processed_file_path, 'w', encoding='utf-8') as f:
            pass
        return []

def add_to_processed_folders(processed_file_path, folder_name):
    """Add newly processed folder to record"""
    with open(processed_file_path, 'a', encoding='utf-8') as f:
        f.write(f"{folder_name}\n")


def classify_folder(folder_name):
    """
    Classify folder name into one of 16 categories
    
    Classification rules:
    - Origin: Original stop sign
    - S: Shape modification only
    - C: Color modification only
    - T: Text modification only
    - P: Pattern modification only
    - S+C: Shape and color modifications
    - S+T: Shape and text modifications
    - S+P: Shape and pattern modifications
    - C+T: Color and text modifications
    - C+P: Color and pattern modifications
    - T+P: Text and pattern modifications
    - S+C+T: Shape, color and text modifications
    - S+C+P: Shape, color and pattern modifications
    - S+T+P: Shape, text and pattern modifications
    - C+T+P: Color, text and pattern modifications
    - ALL: All modifications (shape, color, text, pattern)
    """
    # Feature detection logic
    has_shape = ('square' in folder_name) or ('triangle' in folder_name)
    has_color = ('blue' in folder_name) or ('yellow' in folder_name)
    has_text = ('with__abcd__on_it' in folder_name) or ('with__hello__on_it' in folder_name) or ('with__world__on_it' in folder_name)
    has_pattern = ('checkerboard_paint_on_it' in folder_name) or ('polkadot_paint_on_it' in folder_name)
    
    # Special case: original stop sign
    if folder_name == 'stop_sign':
        return 'Origin'
    
    # Count total features
    total_features = sum([has_shape, has_color, has_text, has_pattern])
    
    # Classification
    if total_features == 1:
        if has_shape:
            return 'S'
        elif has_color:
            return 'C'
        elif has_text:
            return 'T'
        elif has_pattern:
            return 'P'
    elif total_features == 2:
        if has_shape and has_color:
            return 'S+C'
        elif has_shape and has_text:
            return 'S+T'
        elif has_shape and has_pattern:
            return 'S+P'
        elif has_color and has_text:
            return 'C+T'
        elif has_color and has_pattern:
            return 'C+P'
        elif has_text and has_pattern:
            return 'T+P'
    elif total_features == 3:
        if has_shape and has_color and has_text:
            return 'S+C+T'
        elif has_shape and has_color and has_pattern:
            return 'S+C+P'
        elif has_shape and has_text and has_pattern:
            return 'S+T+P'
        elif has_color and has_text and has_pattern:
            return 'C+T+P'
    elif total_features == 4:
        return 'ALL'
    
    # Default case
    return 'Unknown'

def create_global_aasr_summary(folder_results, output_path):
    """
    Create global AASR summary CSV
    
    Parameters:
        folder_results: Nested dictionary of folder_name -> (texture_name -> AASR results)
        output_path: Output CSV file path
        
    Returns:
        Summary DataFrame
    """
    if not folder_results:
        return pd.DataFrame()
    
    # Get all thresholds
    first_folder = next(iter(folder_results.values()))
    first_texture = next(iter(first_folder.values()))
    thresholds = [key.split('_')[1] for key in first_texture.keys() 
                  if key.startswith('AASR_') and key != 'AASR_average']
    
    # Prepare data
    summary_data = []
    
    for folder_name, textures in folder_results.items():
        # Calculate average AASR for all textures in this folder
        folder_avg_aasr = np.mean([results.get('AASR_average', 0) 
                                   for results in textures.values()])
        
        row = {
            'Folder': folder_name,
            'Average_AASR': folder_avg_aasr,
            'Num_Textures': len(textures)
        }
        
        for threshold in thresholds:
            folder_threshold_aasrs = [results.get(f'AASR_{threshold}', {}).get('AASR', 0) 
                                     for results in textures.values()]
            row[f'AASR_{threshold}'] = np.mean(folder_threshold_aasrs)
        
        summary_data.append(row)
    
    # Add average row
    avg_row = {'Folder': 'Average'}
    avg_row['Average_AASR'] = np.mean([r['Average_AASR'] for r in summary_data])
    avg_row['Num_Textures'] = sum([r['Num_Textures'] for r in summary_data])
    for threshold in thresholds:
        avg_row[f'AASR_{threshold}'] = np.mean([r[f'AASR_{threshold}'] for r in summary_data])
    summary_data.append(avg_row)
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    
    return summary_df

def add_grouped_visualizations(summary_output_dir, all_summaries_df, all_angle_data, sensitivity_df, consistency_df):
    """
    Add visualizations grouped by category to existing data
    
    Args:
        summary_output_dir: Output directory
        all_summaries_df: Summary DataFrame of all folders
        all_angle_data: Angle-confidence data of all folders
        sensitivity_df: Angle sensitivity analysis data
        consistency_df: Texture consistency analysis data
    """
    print("Generating category-grouped visualizations...")
    
    # Add category information to data
    if all_summaries_df is not None and 'folder' in all_summaries_df.columns:
        all_summaries_df['category'] = all_summaries_df['folder'].apply(classify_folder)
    
    if all_angle_data is not None:
        # Create grouped angle-confidence data
        grouped_angle_data = {}
        
        # Get all angles
        all_angles = sorted(list(set().union(*[df.index for df in all_angle_data.values()])))
        
        # Classify each folder
        folder_categories = {folder: classify_folder(folder) for folder in all_angle_data.keys()}
        
        # Group data by category
        for category in sorted(set(folder_categories.values())):
            category_folders = [folder for folder, cat in folder_categories.items() if cat == category]
            if not category_folders:
                continue
                
            # Aggregate all folder data for this category
            category_data = pd.DataFrame(index=all_angles)
            for folder in category_folders:
                if folder in all_angle_data:
                    # Add folder data as new column
                    category_data[folder] = all_angle_data[folder]
            
            # Calculate average confidence for this category
            grouped_angle_data[category] = category_data.mean(axis=1)
        
        # Plot grouped angle-confidence comparison
        plt.figure(figsize=(14, 8))
        
        # Define easily distinguishable colors
        colors = plt.cm.tab20(np.linspace(0, 1, len(grouped_angle_data)))
        
        # Plot curve for each category
        for i, (category, data) in enumerate(sorted(grouped_angle_data.items())):
            plt.plot(data.index, data.values, '-', linewidth=2, label=category, color=colors[i])
        
        plt.title('Angle-Confidence Comparison Across Categories (With Environment)', fontsize=14)
        plt.xlabel('Azimuth (degrees)', fontsize=12)
        plt.ylabel('Average Confidence', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(-90, 90)  # Use new angle range (-90 to 90)
        plt.ylim(0, 1.05)
        
        # Add vertical line at 0 degrees (front view)
        plt.axvline(x=0, color='k', linestyle='--', label='Front View (0°)')
        
        # Optimize legend
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                  fancybox=True, shadow=True, ncol=4, fontsize=10)
        
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(bottom=0.2)
        
        # Save grouped comparison plot
        plt.savefig(os.path.join(summary_output_dir, 'grouped_angle_confidence.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save grouped angle-confidence data to CSV
        grouped_df = pd.DataFrame(grouped_angle_data)
        grouped_df.to_csv(os.path.join(summary_output_dir, 'grouped_angle_data.csv'))
        print(f"Saved grouped angle-confidence data to {os.path.join(summary_output_dir, 'grouped_angle_data.csv')}")
    
    # Process front confidence data
    if all_summaries_df is not None and 'folder' in all_summaries_df.columns and 'front_confidence' in all_summaries_df.columns:
        # Calculate average front confidence for each category
        category_avg_conf = all_summaries_df.groupby('category')['front_confidence'].mean().sort_values(ascending=False)
        
        # Plot grouped front confidence bar chart
        plt.figure(figsize=(14, 8))
        ax = category_avg_conf.plot(kind='bar', color='skyblue')
        
        plt.title('Average Front Confidence by Category (Azimuth=0°, With Environment)', fontsize=14)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Average Confidence', fontsize=12)
        
        # Add value labels to all bars
        for i, v in enumerate(category_avg_conf):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout(pad=2.0)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(summary_output_dir, 'grouped_front_confidence.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved grouped front confidence plot to {os.path.join(summary_output_dir, 'grouped_front_confidence.png')}")
    
    # Process confidence statistics data
    if all_summaries_df is not None and 'folder' in all_summaries_df.columns:
        # Calculate max, average and min confidence for each category
        category_stats = all_summaries_df.groupby('category').agg({
            'max_confidence': 'mean',
            'avg_confidence': 'mean',
            'min_confidence': 'mean'
        }).sort_values('avg_confidence', ascending=False)
        
        # Plot grouped confidence statistics bar chart
        plt.figure(figsize=(14, 8))
        ax = category_stats.plot(kind='bar', figsize=(14, 8))
        
        plt.title('Confidence Statistics by Category (With Environment)', fontsize=14)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Confidence', fontsize=12)
        
        plt.legend(['Max Confidence', 'Avg Confidence', 'Min Confidence'], 
                  loc='upper right', fontsize=10)
        
        plt.tight_layout(pad=2.0)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(summary_output_dir, 'grouped_confidence_stats.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved grouped confidence statistics plot to {os.path.join(summary_output_dir, 'grouped_confidence_stats.png')}")
    
    # Process angle sensitivity data
    if sensitivity_df is not None:
        # Add category information
        sensitivity_df['category'] = sensitivity_df.index.map(classify_folder)
        
        # Calculate average sensitivity score for each category
        category_sensitivity = sensitivity_df.groupby('category')['sensitivity_score'].mean().sort_values()
        
        # Plot grouped sensitivity score bar chart
        plt.figure(figsize=(14, 8))
        ax = category_sensitivity.plot(kind='bar', color='salmon')
        
        plt.title('Average Angle Sensitivity Score by Category (With Environment)', fontsize=14)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Sensitivity Score (lower is better)', fontsize=12)
        
        # Add value labels to all bars
        for i, v in enumerate(category_sensitivity):
            offset = 0.04 if v >= 0 else -0.04
            ax.text(i, v + offset, f'{v:.3f}', ha='center', fontsize=10,
                   va='bottom' if v >= 0 else 'top')
        
        plt.tight_layout(pad=2.0)
        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(summary_output_dir, 'grouped_sensitivity_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved grouped angle sensitivity score plot to {os.path.join(summary_output_dir, 'grouped_sensitivity_scores.png')}")
    
    # Process texture consistency data
    if consistency_df is not None:
        # Add category information
        consistency_df['category'] = consistency_df.index.map(classify_folder)
        
        # Calculate average consistency for each category
        category_consistency = consistency_df.groupby('category')['mean_std'].mean().sort_values()
        category_num_textures = consistency_df.groupby('category')['num_textures'].sum()
        
        # Plot grouped consistency bar chart
        plt.figure(figsize=(14, 8))
        ax = category_consistency.plot(kind='bar', color='lightgreen')
        
        plt.title('Texture Consistency Within Categories (With Environment)', fontsize=14)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Average Standard Deviation', fontsize=12)
        
        # Add value labels to all bars
        for i, v in enumerate(category_consistency):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=10)
        
        # Add texture count labels
        for i, (idx, val) in enumerate(category_consistency.items()):
            # Place n=XX label inside the bar
            ax.text(i, 0.03, f"n={int(category_num_textures[idx])}", 
                   ha='center', fontsize=8, color='black', weight='bold')
        
        plt.tight_layout(pad=2.0)
        plt.ylim(0, max(category_consistency) * 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(summary_output_dir, 'grouped_texture_consistency.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved grouped texture consistency plot to {os.path.join(summary_output_dir, 'grouped_texture_consistency.png')}")
        
    # Create violin plot to visualize confidence distribution by category
    if all_angle_data is not None:
        # Prepare violin plot data
        violin_data = []
        
        for folder, data in all_angle_data.items():
            category = classify_folder(folder)
            for angle in data.index:
                violin_data.append({
                    'folder': folder,
                    'category': category,
                    'angle': angle,
                    'confidence': data[angle]
                })
        
        # Create DataFrame
        if violin_data:
            violin_df = pd.DataFrame(violin_data)
            
            # Sort categories by average confidence
            category_mean_conf = violin_df.groupby('category')['confidence'].mean().sort_values(ascending=False)
            category_order = category_mean_conf.index.tolist()
            
            # Create violin plot
            plt.figure(figsize=(14, 10))
            
            # Plot violin
            ax = sns.violinplot(
                x='category', 
                y='confidence', 
                data=violin_df,
                order=category_order,
                palette='viridis',
                scale='width',
                inner='quartile',
                linewidth=1,
                cut=0
            )
            
            # Add title and labels
            plt.title('Angle Confidence Distribution by Category (With Environment)', fontsize=14)
            plt.xlabel('Stop Sign Category', fontsize=12)
            plt.ylabel('Detection Confidence', fontsize=12)
            
            # Rotate x-axis labels to avoid overlap
            plt.xticks(rotation=45, ha='right')
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Save violin plot
            plt.savefig(os.path.join(summary_output_dir, 'grouped_confidence_violinplot.png'), 
                      dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved grouped confidence distribution violin plot to {os.path.join(summary_output_dir, 'grouped_confidence_violinplot.png')}")
    
    # Create heatmap to visualize category performance at different angles
    if all_angle_data is not None:
        # Prepare heatmap data
        heatmap_data = {}
        
        # Classify each folder
        folder_categories = {folder: classify_folder(folder) for folder in all_angle_data.keys()}
        
        # Get all angles
        all_angles = sorted(list(set().union(*[df.index for df in all_angle_data.values()])))
        
        # Calculate average confidence by category
        for category in sorted(set(folder_categories.values())):
            category_folders = [folder for folder, cat in folder_categories.items() if cat == category]
            if not category_folders:
                continue
            
            # Initialize angle data for this category
            category_angle_data = []
            
            for angle in all_angles:
                # Collect confidence at this angle for all folders
                angle_confidences = []
                for folder in category_folders:
                    if folder in all_angle_data and angle in all_angle_data[folder].index:
                        angle_confidences.append(all_angle_data[folder][angle])
                
                # Calculate average confidence
                if angle_confidences:
                    avg_conf = sum(angle_confidences) / len(angle_confidences)
                    category_angle_data.append(avg_conf)
                else:
                    category_angle_data.append(0)
            
            heatmap_data[category] = category_angle_data
        
        # Create heatmap
        plt.figure(figsize=(15, 10))
        
        # Convert data to NumPy array
        categories = list(heatmap_data.keys())
        heat_array = np.array([heatmap_data[cat] for cat in categories])
        
        # Plot heatmap
        im = plt.imshow(heat_array, aspect='auto', cmap='viridis', 
                        vmin=0, vmax=1)  # Ensure color range is 0-1
        
        # Set ticks and labels
        plt.colorbar(im, label='Average Confidence')
        plt.xlabel('Azimuth Angle (degrees)')
        plt.ylabel('Category')
        plt.title('Average Detection Confidence Heatmap by Category (With Environment)')
        
        # Set x-axis ticks (angles)
        # Select representative angles as ticks
        angle_indices = np.linspace(0, len(all_angles)-1, min(10, len(all_angles))).astype(int)
        plt.xticks(angle_indices, [all_angles[i] for i in angle_indices])
        
        # Set y-axis tick labels (category names)
        plt.yticks(np.arange(len(categories)), categories, fontsize=10)
        
        # Add vertical line at 0 degrees
        # Find index of angle closest to 0 degrees
        zero_angle_idx = np.argmin(np.abs(np.array(all_angles)))
        plt.axvline(x=zero_angle_idx, color='r', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        
        # Save heatmap
        heatmap_path = os.path.join(summary_output_dir, 'grouped_confidence_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved grouped confidence heatmap to {heatmap_path}")
    
    print("Grouped visualizations complete!")

def parse_args():
    parser = argparse.ArgumentParser(description='Multi-folder batch processing script with Environment Compositing and AASR')
    parser.add_argument('--base-dir', 
                      type=str, 
                      default="/zhaotingfeng/jiwenjun/anglerocl/data/stable_diffusion_1.5/ndda", 
                      help='Base directory containing all subdirectories to process')
    parser.add_argument('--output-base-dir', 
                      type=str, 
                      default="/zhaotingfeng/jiwenjun/anglerocl/multiview_output_back", 
                      help='Base directory to save all outputs')
    parser.add_argument('--texture-extensions', 
                      type=str, 
                      default="png,jpg,jpeg", 
                      help='Image extensions to process, comma separated')
    parser.add_argument('--exclude-folders', 
                      type=str, 
                      default="", 
                      help='Folders to exclude, comma separated')
    
    # Environment compositing related parameters
    parser.add_argument('--background-image', 
                      type=str, 
                      default="/zhaotingfeng/jiwenjun/anglerocl/environment/back.jpg", 
                      help='Path to the background image')
    parser.add_argument('--background-size', 
                      type=int, 
                      nargs=2, 
                      default=[1600, 900], 
                      help='Background size (width, height)')
    
    # Detector related parameters
    parser.add_argument('--detector', 
                    type=str, 
                    choices=['yolov3', 'yolov5', 'yolov10', 'faster_rcnn', 'detr'],
                    default='yolov5',
                    help='Detector to use')
    parser.add_argument('--yolov5-model', 
                    type=str, 
                    default="/zhaotingfeng/jiwenjun/anglerocl/checkpoints/yolov5n.pt", 
                    help='Path to YOLOv5 weights (only for yolov5)')
    parser.add_argument('--yolov10-model',
                    type=str,
                    default='/zhaotingfeng/jiwenjun/anglerocl/checkpoints/yolov10n.pt',
                    help='YOLOv10 model path')
    parser.add_argument('--mmdet-config',
                    type=str,
                    help='MMDetection config file path')
    parser.add_argument('--mmdet-checkpoint',
                    type=str,
                    help='MMDetection checkpoint file path')
    
    # Rendering parameters
    parser.add_argument('--script-path', 
                    type=str, 
                    default="/zhaotingfeng/jiwenjun/anglerocl/test/multiview_detection_folder_multidetector_environment.py", 
                    help='Path to the environment compositing script')
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
    parser.add_argument('--azim-step', 
                      type=int, 
                      default=1, 
                      help='Step size for azimuth angle (degrees)')
    parser.add_argument('--save-renders', 
                      action='store_true', 
                      help='Whether to save all rendered images')
    
    # Parallel parameters
    parser.add_argument('--parallel-folders', 
                      action='store_true', 
                      help='Whether to process folders in parallel (requires multiple GPUs)')
    parser.add_argument('--max-parallel', 
                      type=int, 
                      default=4, 
                      help='Maximum number of parallel folders')
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
                      help='AASR confidence thresholds, comma-separated')
    parser.add_argument('--aasr-min-angle',
                      type=float,
                      default=-89.0,
                      help='Minimum angle for AASR calculation')
    parser.add_argument('--aasr-max-angle',
                      type=float,
                      default=89.0,
                      help='Maximum angle for AASR calculation')
    
    return parser.parse_args()

def main():
    args = parse_args()
    start_time = time.time()
    
    # Set detector output directory, add environment identifier
    detector_dir = f"{args.detector}"
    
    args.output_base_dir = os.path.join(args.output_base_dir, detector_dir)
    
    # Get all subdirectories under base directory (various types of stop sign folders)
    all_folders = [f for f in glob.glob(os.path.join(args.base_dir, "*")) if os.path.isdir(f)]
    
    # If exclude folders are specified, remove them from the list
    if args.exclude_folders:
        exclude_list = [os.path.join(args.base_dir, folder.strip()) for folder in args.exclude_folders.split(",")]
        all_folders = [f for f in all_folders if f not in exclude_list]
    
    # Ensure only base output directory is created
    os.makedirs(args.output_base_dir, exist_ok=True)
    
    # Extract last two levels from input base directory path for reporting
    base_dir_parts = args.base_dir.rstrip('/').split('/')
    if len(base_dir_parts) >= 2:
        output_subpath = os.path.join(base_dir_parts[-2], base_dir_parts[-1])
    else:
        output_subpath = base_dir_parts[-1]
    
    # Determine path for processed_folders.txt
    expected_output_base = os.path.join(args.output_base_dir, output_subpath)
    processed_folders_file = os.path.join(expected_output_base, "processed_folders.txt")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(processed_folders_file), exist_ok=True)
    print(f"Will use {processed_folders_file} as processed folder record")
    
    # Load list of processed folders
    processed_folders = load_processed_folders(processed_folders_file)
    print(f"Found {len(processed_folders)} already processed folders")
    
    # Filter out already processed folders
    original_count = len(all_folders)
    all_folders = [folder for folder in all_folders 
                    if os.path.basename(folder) not in processed_folders]
    skipped_count = original_count - len(all_folders)
    
    print(f"Skipping {skipped_count} already processed folders")
    print(f"Remaining folders to process: {len(all_folders)}")
    
    if len(all_folders) == 0:
        print("All folders have been processed, exiting.")
        exit(0)
    
    print(f"Will process {len(all_folders)} folders:")
    for folder in all_folders:
        print(f"  - {os.path.basename(folder)}")
    
    # Process each folder
    if args.parallel_folders:
        # Process multiple folders in parallel
        from concurrent.futures import ProcessPoolExecutor
        import torch
        
        # Determine available GPU count
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        max_workers = min(args.max_parallel, gpu_count, len(all_folders))
        
        print(f"Using {max_workers} processes to process folders in parallel")
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for i, folder in enumerate(all_folders):
                folder_name = os.path.basename(folder)
                
                # Assign different GPU to each process (if multiple available)
                gpu_id = i % gpu_count if gpu_count > 1 else 0
                
                # Build command line arguments for perspective view script
                cmd_args = [
                    f"CUDA_VISIBLE_DEVICES={gpu_id}",
                    "python", args.script_path,
                    f"--texture-folder={folder}",
                    f"--output-dir={os.path.dirname(args.output_base_dir)}",
                    f"--detector={args.detector}",
                    f"--background-image={args.background_image}",
                    f"--background-size", f"{args.background_size[0]}", f"{args.background_size[1]}",
                    f"--texture-extensions={args.texture_extensions}",
                    f"--renderer-fov={args.renderer_fov}",
                    f"--renderer-distance={args.renderer_distance}",
                    f"--resolution={args.resolution}",
                    f"--azim-step={args.azim_step}",
                    f"--batch-size={args.batch_size}"
                ]
                
                # Add extra parameters based on detector type
                if args.detector == 'yolov5':
                    cmd_args.append(f"--yolov5-model={args.yolov5_model}")
                elif args.detector == 'yolov10':
                    cmd_args.append(f"--yolov10-model={args.yolov10_model}")
                elif args.detector in ['yolov3', 'faster_rcnn', 'detr']:
                    if args.mmdet_config:
                        cmd_args.append(f"--mmdet-config={args.mmdet_config}")
                    if args.mmdet_checkpoint:
                        cmd_args.append(f"--mmdet-checkpoint={args.mmdet_checkpoint}")
                
                # Add AASR parameters
                if args.enable_aasr:
                    cmd_args.append("--enable-aasr")
                    cmd_args.append(f"--aasr-thresholds={args.aasr_thresholds}")
                    cmd_args.append(f"--aasr-min-angle={args.aasr_min_angle}")
                    cmd_args.append(f"--aasr-max-angle={args.aasr_max_angle}")
                
                # If need to save rendered images, add corresponding parameter
                if args.save_renders:
                    cmd_args.append("--save-renders")
                
                # Submit task
                cmd_str = " ".join(cmd_args)
                future = executor.submit(subprocess.run, cmd_str, shell=True)
                futures.append((future, folder_name))
            
            # Wait for all tasks to complete
            for i, (future, folder_name) in enumerate(tqdm(futures, desc="Processing folders")):
                result = future.result()
                print(f"Completed processing folder {i+1}/{len(all_folders)}: {folder_name}")
                
                # If task completed successfully, add folder name to processed list
                if result.returncode == 0:
                    add_to_processed_folders(processed_folders_file, folder_name)
                    print(f"Added {folder_name} to processed folder list")
                elif result.returncode != 0:
                    print(f"Warning: Processing folder {folder_name} failed")
    
    else:
        # Process each folder sequentially
        for i, folder in enumerate(all_folders):
            folder_name = os.path.basename(folder)
            
            print(f"\nProcessing folder {i+1}/{len(all_folders)}: {folder_name}")
            
            # Build command line arguments for perspective view script
            cmd = [
                "python", args.script_path,
                f"--texture-folder={folder}",
                f"--output-dir={os.path.dirname(args.output_base_dir)}",
                f"--detector={args.detector}",
                f"--background-image={args.background_image}",
                f"--background-size", f"{args.background_size[0]}", f"{args.background_size[1]}",
                f"--texture-extensions={args.texture_extensions}",
                f"--renderer-fov={args.renderer_fov}",
                f"--renderer-distance={args.renderer_distance}",
                f"--resolution={args.resolution}",
                f"--azim-step={args.azim_step}",
                f"--batch-size={args.batch_size}"
            ]
            
            # Add extra parameters based on detector type
            if args.detector == 'yolov5':
                cmd.append(f"--yolov5-model={args.yolov5_model}")
            elif args.detector == 'yolov10':
                cmd.append(f"--yolov10-model={args.yolov10_model}")
            elif args.detector in ['yolov3', 'faster_rcnn', 'detr']:
                if args.mmdet_config:
                    cmd.append(f"--mmdet-config={args.mmdet_config}")
                if args.mmdet_checkpoint:
                    cmd.append(f"--mmdet-checkpoint={args.mmdet_checkpoint}")
            
            # Add AASR parameters
            if args.enable_aasr:
                cmd.append("--enable-aasr")
                cmd.append(f"--aasr-thresholds={args.aasr_thresholds}")
                cmd.append(f"--aasr-min-angle={args.aasr_min_angle}")
                cmd.append(f"--aasr-max-angle={args.aasr_max_angle}")
            
            # If need to save rendered images, add corresponding parameter
            if args.save_renders:
                cmd.append("--save-renders")
            
            # Print complete command for debugging
            cmd_str = " ".join(cmd)
            print(f"Executing command: {cmd_str}")
            
            # Execute command
            result = subprocess.run(" ".join(cmd), shell=True)
            
            # If task completed successfully, add folder name to processed list
            if result.returncode == 0:
                add_to_processed_folders(processed_folders_file, folder_name)
                print(f"Added {folder_name} to processed folder list")
            elif result.returncode != 0:
                print(f"Warning: Processing folder {folder_name} failed")
    
    # For summary section, we need to determine where original script placed output
    expected_output_base = os.path.join(args.output_base_dir, output_subpath)
    print(f"Expected output base location: {expected_output_base}")
    
    # Verify this directory exists before continuing
    if not os.path.exists(expected_output_base):
        print(f"Warning: Expected output directory {expected_output_base} does not exist.")
        print("Attempting to locate actual output directory...")
        
        # Try to locate folders by name
        all_folder_names = [os.path.basename(f) for f in all_folders]
        potential_output_dirs = []
        
        for root, dirs, _ in os.walk(args.output_base_dir):
            for dir_name in dirs:
                if dir_name in all_folder_names:
                    potential_output_dirs.append(os.path.join(root, dir_name))
        
        if potential_output_dirs:
            print(f"Found possible output directories: {len(potential_output_dirs)}")
            # Determine common parent directory
            common_parent = os.path.commonpath(potential_output_dirs)
            expected_output_base = common_parent
            print(f"Using detected output base location: {expected_output_base}")
        else:
            print("Could not find output directory. Summary may be incomplete.")
    
    # Create directory to store all summary results
    summary_output_dir = os.path.join(expected_output_base, 'all_folders_summary')
    os.makedirs(summary_output_dir, exist_ok=True)
    print(f"Created summary results directory: {summary_output_dir}")

    # Collect all folder information based on expected output location
    folder_info = []
    for folder in all_folders:
        folder_name = os.path.basename(folder)
        expected_output_dir = os.path.join(expected_output_base, folder_name)
        
        # Check if directory exists
        if not os.path.exists(expected_output_dir):
            print(f"Warning: Expected output directory {expected_output_dir} does not exist for {folder_name}")
            # Try to find actual output directory by searching
            found = False
            for root, dirs, _ in os.walk(expected_output_base):
                if folder_name in dirs:
                    actual_output_dir = os.path.join(root, folder_name)
                    print(f"Found actual output location: {actual_output_dir}")
                    folder_info.append({
                        'name': folder_name,
                        'input_path': folder,
                        'output_path': actual_output_dir
                    })
                    found = True
                    break
            
            if not found:
                print(f"Could not find output for {folder_name}, will exclude from summary")
        else:
            folder_info.append({
                'name': folder_name,
                'input_path': folder,
                'output_path': expected_output_dir
            })

    # 1. Aggregate all folders' summary_results.csv files
    all_summaries = []
    for folder in folder_info:
        # Find summary_results.csv file for this folder
        summary_file_pattern = os.path.join(folder['output_path'], '**', 'summary_results.csv')
        summary_files = glob.glob(summary_file_pattern, recursive=True)
        
        for summary_file in summary_files:
            if os.path.exists(summary_file):
                try:
                    df = pd.read_csv(summary_file)
                    # Add folder name column
                    df['folder'] = folder['name']
                    all_summaries.append(df)
                    print(f"Read summary file: {summary_file}")
                except Exception as e:
                    print(f"Error reading summary file {summary_file}: {str(e)}")

    # If summary files found, merge them
    if all_summaries:
        all_summaries_df = pd.concat(all_summaries, ignore_index=True)
        # Save merged summary file
        all_summaries_path = os.path.join(summary_output_dir, 'all_folders_summary.csv')
        all_summaries_df.to_csv(all_summaries_path, index=False)
        print(f"Saved merged summary to: {all_summaries_path}")
        
        # Create comparison charts between folders
        # 1. Average front confidence by folder
        plt.figure(figsize=(20, 10))
        folder_avg_conf = all_summaries_df.groupby('folder')['front_confidence'].mean().sort_values(ascending=False)
        ax = folder_avg_conf.plot(kind='bar', color='skyblue')

        plt.title('Average Front Confidence by Folder (Azimuth=0°, With Environment)', fontsize=16)
        plt.xlabel('Folder', fontsize=14)
        plt.ylabel('Average Confidence', fontsize=14)

        plt.xticks(rotation=45, ha='right', fontsize=10)

        # Add value labels to all bars
        for i, v in enumerate(folder_avg_conf):
            ax.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(bottom=0.25)

        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(summary_output_dir, 'folder_avg_front_confidence.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Max, average and min confidence by folder
        plt.figure(figsize=(20, 10))
        folder_stats = all_summaries_df.groupby('folder').agg({
            'max_confidence': 'mean',
            'avg_confidence': 'mean',
            'min_confidence': 'mean'
        }).sort_values('avg_confidence', ascending=False)

        ax = folder_stats.plot(kind='bar', figsize=(20, 10))

        plt.title('Confidence Statistics by Folder (With Environment)', fontsize=16)
        plt.xlabel('Folder', fontsize=14)
        plt.ylabel('Confidence', fontsize=14)

        plt.xticks(rotation=45, ha='right', fontsize=10)

        plt.legend(['Max Confidence', 'Avg Confidence', 'Min Confidence'], 
                  loc='upper right', fontsize=12)

        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(bottom=0.25)

        plt.grid(axis='y', alpha=0.3)

        plt.savefig(os.path.join(summary_output_dir, 'folder_confidence_stats.png'), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("No summary files found!")

    # 2. Aggregate and compare angle-confidence data for all folders
    all_angle_data = {}
    for folder in folder_info:
        angle_data_pattern = os.path.join(folder['output_path'], '**', 'all_textures_data.csv')
        angle_data_files = glob.glob(angle_data_pattern, recursive=True)
        
        for angle_data_file in angle_data_files:
            if os.path.exists(angle_data_file):
                try:
                    # Read angle-confidence data
                    angle_df = pd.read_csv(angle_data_file)
                    
                    # Calculate average confidence for each angle
                    angle_data = angle_df.set_index('angle')
                    mean_values = angle_data.mean(axis=1)
                    all_angle_data[folder['name']] = mean_values
                    print(f"Read angle data: {angle_data_file}")
                except Exception as e:
                    print(f"Error reading angle data file {angle_data_file}: {str(e)}")

    # If angle-confidence data found, create comparison plot
    if all_angle_data:
        # Create wider figure to accommodate all legends
        plt.figure(figsize=(40, 10))
        
        # Get all angles (different folders may have different angle sets)
        all_angles = sorted(list(set().union(*[df.index for df in all_angle_data.values()])))
        
        # Create new DataFrame to store all folders' data
        combined_df = pd.DataFrame(index=all_angles)
        for folder_name, data in all_angle_data.items():
            combined_df[folder_name] = data
        
        # Fill missing values (if any)
        combined_df = combined_df.interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
        
        # Save merged angle-confidence data
        combined_angle_data_path = os.path.join(summary_output_dir, 'all_folders_angle_data.csv')
        combined_df.to_csv(combined_angle_data_path)
        print(f"Saved merged angle data to: {combined_angle_data_path}")
        
        # Plot curve for each folder
        for folder_name in combined_df.columns:
            plt.plot(combined_df.index, combined_df[folder_name], linewidth=1.5, label=folder_name)
        
        plt.title('Angle-Confidence Comparison Across Folders (With Environment)', fontsize=14)
        plt.xlabel('Azimuth (degrees)', fontsize=12)
        plt.ylabel('Average Confidence', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim(-90, 90)
        plt.ylim(0, 1.05)
        
        # Add vertical line at 0 degrees (front view)
        plt.axvline(x=0, color='k', linestyle='--', label='Front View (0°)')
        
        # Place legend below chart
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                fancybox=True, shadow=True, ncol=min(8, len(combined_df.columns)), fontsize=8)
        
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(bottom=0.2)
        
        plt.savefig(os.path.join(summary_output_dir, 'all_folders_angle_confidence.png'), 
                dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create violin plot
        violin_data = []
        for folder_name in combined_df.columns:
            for angle in combined_df.index:
                confidence = combined_df.loc[angle, folder_name]
                if not pd.isna(confidence):
                    violin_data.append({
                        'folder': folder_name,
                        'angle': angle,
                        'confidence': confidence
                    })
        
        # Create DataFrame
        violin_df = pd.DataFrame(violin_data)
        
        # Sort folders by average confidence
        folder_mean_conf = violin_df.groupby('folder')['confidence'].mean().sort_values(ascending=False)
        folder_order = folder_mean_conf.index.tolist()
        
        # Create violin plot
        plt.figure(figsize=(20, 12))
        
        ax = sns.violinplot(
            x='folder', 
            y='confidence', 
            data=violin_df,
            order=folder_order,
            palette='viridis',
            scale='width',
            inner='quartile',
            linewidth=1,
            cut=0
        )
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.title('Angle Confidence Distribution by Folder (With Environment)', fontsize=16)
        plt.xlabel('Stop Sign Configuration', fontsize=14)
        plt.ylabel('Detection Confidence', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(bottom=0.25)
        
        plt.savefig(os.path.join(summary_output_dir, 'folder_confidence_violinplot.png'), 
                  dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create heatmap
        plt.figure(figsize=(15, 10))
        
        heatmap_data = []
        folder_names = []
        
        for folder_name in combined_df.columns:
            folder_names.append(folder_name)
            confidences = combined_df[folder_name].tolist()
            heatmap_data.append(confidences)
        
        heatmap_array = np.array(heatmap_data)
        
        plt.imshow(heatmap_array, aspect='auto', cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(label='Detection Confidence')
        plt.xlabel('Azimuth Angle (degrees)')
        plt.ylabel('Folder')
        plt.title('Detection Confidence Heatmap (With Environment)')

        plt.xticks(np.linspace(0, heatmap_array.shape[1]-1, 5), 
                np.linspace(combined_df.index.min(), combined_df.index.max(), 5).astype(int))

        plt.yticks(np.arange(len(folder_names)), folder_names, fontsize=8)

        zero_angle_idx = np.argmin(np.abs(combined_df.index.values))
        plt.axvline(x=zero_angle_idx, color='r', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        
        heatmap_path = os.path.join(summary_output_dir, 'confidence_heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confidence heatmap saved to {heatmap_path}")
    else:
        print("No angle-confidence data files found!")

    # 3. Analyze extreme angle sensitivity for all folders
    angle_sensitivity = {}
    if all_angle_data:
        for folder_name, data in all_angle_data.items():
            front_idx_position = data.index.get_indexer([0], method='nearest')[0]
            front_angle = data.index[front_idx_position]
            front_view_conf = data[front_angle]
            
            left_idx_position = data.index.get_indexer([-45], method='nearest')[0]
            left_angle = data.index[left_idx_position]
            left_view_conf = data[left_angle]
            
            right_idx_position = data.index.get_indexer([45], method='nearest')[0]
            right_angle = data.index[right_idx_position]
            right_view_conf = data[right_angle]
            
            left_change = (front_view_conf - left_view_conf) / front_view_conf if front_view_conf > 0 else 0
            right_change = (front_view_conf - right_view_conf) / front_view_conf if front_view_conf > 0 else 0
            sensitivity_score = (left_change + right_change) / 2
            
            angle_sensitivity[folder_name] = {
                'front_view': front_view_conf,
                'left_view': left_view_conf,
                'right_view': right_view_conf,
                'left_change': left_change,
                'right_change': right_change,
                'sensitivity_score': sensitivity_score
            }

        sensitivity_df = pd.DataFrame.from_dict(angle_sensitivity, orient='index')
        sensitivity_path = os.path.join(summary_output_dir, 'angle_sensitivity_analysis.csv')
        sensitivity_df.to_csv(sensitivity_path)
        print(f"Saved angle sensitivity analysis to: {sensitivity_path}")
        
        plt.figure(figsize=(20, 10))
        sensitivity_scores = sensitivity_df.sort_values('sensitivity_score')['sensitivity_score']
        ax = sensitivity_scores.plot(kind='bar', color='salmon')

        plt.title('Angle Sensitivity Score by Folder (With Environment)', fontsize=16)
        plt.xlabel('Folder', fontsize=14)
        plt.ylabel('Sensitivity Score (lower is better)', fontsize=14)

        plt.xticks(rotation=45, ha='right', fontsize=10)

        for i, v in enumerate(sensitivity_scores):
            offset = 0.04 if v >= 0 else -0.04
            ax.text(i, v + offset, f'{v:.3f}', ha='center', fontsize=9,
                va='bottom' if v >= 0 else 'top')

        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(bottom=0.25)

        plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)

        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(summary_output_dir, 'angle_sensitivity_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Analyze texture consistency within folders
    texture_consistency = {}
    for folder in folder_info:
        angle_data_pattern = os.path.join(folder['output_path'], '**', 'all_textures_data.csv')
        angle_data_files = glob.glob(angle_data_pattern, recursive=True)
        
        for angle_data_file in angle_data_files:
            if os.path.exists(angle_data_file):
                try:
                    angle_df = pd.read_csv(angle_data_file)
                    data = angle_df.set_index('angle')
                    
                    std_values = data.std(axis=1)
                    mean_std = std_values.mean()
                    max_std = std_values.max()
                    
                    texture_consistency[folder['name']] = {
                        'mean_std': mean_std,
                        'max_std': max_std,
                        'num_textures': data.shape[1]
                    }
                    
                    print(f"Analyzed internal consistency for folder: {folder['name']}")
                except Exception as e:
                    print(f"Error analyzing internal consistency for {angle_data_file}: {str(e)}")

    if texture_consistency:
        consistency_df = pd.DataFrame.from_dict(texture_consistency, orient='index')
        consistency_path = os.path.join(summary_output_dir, 'texture_consistency_analysis.csv')
        consistency_df.to_csv(consistency_path)
        print(f"Saved texture consistency analysis to: {consistency_path}")
        
        plt.figure(figsize=(20, 10))
        consistency_sorted = consistency_df.sort_values('mean_std')
        ax = consistency_sorted['mean_std'].plot(kind='bar', color='lightgreen')

        plt.title('Texture Consistency Within Folders (With Environment)', fontsize=16)
        plt.xlabel('Folder', fontsize=14)
        plt.ylabel('Average Standard Deviation', fontsize=14)

        plt.xticks(rotation=45, ha='right', fontsize=10)

        for i, v in enumerate(consistency_sorted['mean_std']):
            ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

        for i, (idx, row) in enumerate(consistency_sorted.iterrows()):
            bar_height = consistency_sorted['mean_std'].iloc[i]
            ax.text(i, 0.03, f"n={int(row['num_textures'])}", 
                        ha='center', fontsize=8, color='black', weight='bold')

        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(bottom=0.3)

        plt.ylim(0, max(consistency_sorted['mean_std']) * 1.1)

        plt.grid(axis='y', alpha=0.3)
        plt.savefig(os.path.join(summary_output_dir, 'texture_consistency.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # ========== Generate Global AASR Summary ==========
    if args.enable_aasr:
        try:
            print("\nGenerating global AASR summary...")
            
            # Collect AASR results from all folders
            global_aasr_results = {}
            
            for folder in folder_info:
                folder_name = folder['name']
                
                # Find folder_aasr_summary.csv for this folder
                folder_output_dir = os.path.join(expected_output_base, folder_name)
                folder_aasr_path = os.path.join(folder_output_dir, 'folder_aasr_summary.csv')
                
                if os.path.exists(folder_aasr_path):
                    try:
                        # Read folder's AASR summary
                        folder_df = pd.read_csv(folder_aasr_path)
                        
                        # Exclude "Average" row, keep only texture data
                        texture_rows = folder_df[folder_df['Texture'] != 'Average']
                        
                        # Convert to dictionary format
                        folder_textures = {}
                        for _, row in texture_rows.iterrows():
                            texture_name = row['Texture']
                            texture_aasr = {
                                'AASR_average': row['Average_AASR']
                            }
                            
                            # Add AASR for each threshold
                            for col in row.index:
                                if col.startswith('AASR_') and col != 'Average_AASR':
                                    threshold = col.split('_')[1]
                                    texture_aasr[col] = {
                                        'AASR': row[col],
                                        'threshold': float(threshold)
                                    }
                            
                            folder_textures[texture_name] = texture_aasr
                        
                        if folder_textures:
                            global_aasr_results[folder_name] = folder_textures
                        
                    except Exception as e:
                        print(f"Warning: Error reading AASR data for folder {folder_name}: {str(e)}")
            
            # Generate global summary
            if global_aasr_results:
                global_aasr_path = os.path.join(summary_output_dir, 'global_aasr_summary.csv')
                global_df = create_global_aasr_summary(global_aasr_results, global_aasr_path)
                
                print(f"Global AASR summary saved to: {global_aasr_path}")
                
                # Print global statistics
                avg_row = global_df[global_df['Folder'] == 'Average'].iloc[0]
                print(f"\nGlobal Average AASR: {avg_row['Average_AASR']:.2f}%")
                print(f"Total folders processed: {len(global_aasr_results)}")
                print(f"Total textures processed: {avg_row['Num_Textures']}")
                
        except Exception as e:
            print(f"Warning: Error generating global AASR summary: {str(e)}")
            import traceback
            traceback.print_exc()
    # ============================================

    # Generate grouped visualizations
    print("Generating category-grouped visualizations...")
    add_grouped_visualizations(
        summary_output_dir, 
        all_summaries_df if 'all_summaries_df' in locals() else None,
        all_angle_data if 'all_angle_data' in locals() else None,
        sensitivity_df if 'sensitivity_df' in locals() else None,
        consistency_df if 'consistency_df' in locals() else None
    )

    # 5. Analyze angle range statistics
    if all_angle_data:
        angle_ranges = [
            ('Front View', -15, 15),
            ('Moderate Side View', -45, -15),
            ('Moderate Side View', 15, 45),
            ('Extreme Side View', -90, -45),
            ('Extreme Side View', 45, 90)
        ]
        
        range_stats = {}
        for range_name, angle_min, angle_max in angle_ranges:
            range_stats[f"{range_name} ({angle_min}° to {angle_max}°)"] = []
            
            for folder_name, angle_data in all_angle_data.items():
                range_confidences = [
                    conf for angle, conf in zip(angle_data.index, angle_data.values)
                    if angle_min <= angle <= angle_max
                ]
                
                if range_confidences:
                    range_avg = sum(range_confidences) / len(range_confidences)
                    range_stats[f"{range_name} ({angle_min}° to {angle_max}°)"].append((folder_name, range_avg))
        
        range_stats_path = os.path.join(summary_output_dir, 'angle_range_stats.csv')
        with open(range_stats_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Angle Range', 'Folder', 'Average Confidence'])
            
            for range_name, stats in range_stats.items():
                sorted_stats = sorted(stats, key=lambda x: x[1], reverse=True)
                for folder_name, avg_conf in sorted_stats:
                    csv_writer.writerow([range_name, folder_name, avg_conf])
        
        print(f"Angle range statistics data saved to {range_stats_path}")

    # 6. Create environment detector performance summary
    if 'all_summaries_df' in locals():
        environment_performance_summary = {
            'detector': args.detector.upper(),
            'background': os.path.basename(args.background_image),
            'total_folders': len(folder_info),
            'total_textures': len(all_summaries_df),
            'avg_front_confidence': all_summaries_df['front_confidence'].mean(),
            'avg_max_confidence': all_summaries_df['max_confidence'].mean(),
            'avg_min_confidence': all_summaries_df['min_confidence'].mean(),
            'avg_avg_confidence': all_summaries_df['avg_confidence'].mean(),
        }
        
        performance_path = os.path.join(summary_output_dir, f'{args.detector}_performance_summary.csv')
        with open(performance_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Metric', 'Value'])
            for key, value in environment_performance_summary.items():
                csv_writer.writerow([key, value])
        
        print(f"Environment detector performance summary saved to {performance_path}")
        
        print(f"\n{args.detector.upper()} Detector Performance Summary in Environment:")
        print(f"  Background Image: {environment_performance_summary['background']}")
        print(f"  Folders Processed: {environment_performance_summary['total_folders']}")
        print(f"  Total Textures Processed: {environment_performance_summary['total_textures']}")
        print(f"  Average Front Confidence: {environment_performance_summary['avg_front_confidence']:.4f}")
        print(f"  Average Max Confidence: {environment_performance_summary['avg_max_confidence']:.4f}")
        print(f"  Average Min Confidence: {environment_performance_summary['avg_min_confidence']:.4f}")
        print(f"  Overall Average Confidence: {environment_performance_summary['avg_avg_confidence']:.4f}")

    # 7. Create environment summary report document
    report_path = os.path.join(summary_output_dir, 'environment_summary_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# Multi-Folder Environment Composited Stop Sign Detection Analysis Summary Report\n\n')
        f.write(f'Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        
        f.write('## Environment Setup\n\n')
        f.write(f'- **Detector**: {args.detector.upper()}\n')
        f.write(f'- **Background Image**: {args.background_image}\n')
        f.write(f'- **Background Size**: {args.background_size[0]} x {args.background_size[1]}\n')
        f.write(f'- **Renderer FOV**: {args.renderer_fov}°\n')
        f.write(f'- **Renderer Distance**: {args.renderer_distance}\n')
        f.write(f'- **Resolution**: {args.resolution} x {args.resolution}\n')
        f.write(f'- **Angle Step**: {args.azim_step}°\n')
        
        if args.enable_aasr:
            f.write(f'- **AASR Enabled**: Yes\n')
            f.write(f'- **AASR Thresholds**: {args.aasr_thresholds}\n')
            f.write(f'- **AASR Angle Range**: [{args.aasr_min_angle}°, {args.aasr_max_angle}°]\n')
        else:
            f.write(f'- **AASR Enabled**: No\n')
        f.write('\n')
        
        f.write('## Processed Folders\n\n')
        for i, folder in enumerate(folder_info):
            f.write(f"{i+1}. **{folder['name']}**\n")
            f.write(f"   - Input Path: {folder['input_path']}\n")
            f.write(f"   - Output Path: {folder['output_path']}\n\n")
        
        f.write('## Overall Performance Comparison\n\n')
        f.write('Performance comparison of folders across different metrics:\n\n')
        f.write('1. **Front Confidence** - Shows average detection confidence at 0-degree azimuth for textures in each folder.\n')
        f.write('2. **Confidence Statistics** - Shows max, average, and min detection confidence for textures in each folder.\n')
        f.write('3. **Angle-Confidence Comparison** - Compares average confidence performance across all azimuth angles for different folders.\n')
        f.write('4. **Angle Sensitivity** - Quantifies how much confidence drops from front view to side views.\n')
        f.write('5. **Texture Consistency** - Measures consistency of detection across different textures within each folder.\n')
        
        if args.enable_aasr and 'global_aasr_results' in locals() and global_aasr_results:
            f.write('6. **AASR Analysis** - Evaluates Angular Attack Success Rate of adversarial samples.\n')
        f.write('\n')
        
        f.write('## Key Findings\n\n')
        
        if 'folder_avg_conf' in locals():
            best_folder = folder_avg_conf.index[0]
            worst_folder = folder_avg_conf.index[-1]
            f.write(f"- **Best Front View Performance**: {best_folder} (Average Confidence: {folder_avg_conf.iloc[0]:.3f})\n")
            f.write(f"- **Worst Front View Performance**: {worst_folder} (Average Confidence: {folder_avg_conf.iloc[-1]:.3f})\n\n")
        
        if 'sensitivity_df' in locals():
            sensitivity_series = sensitivity_df['sensitivity_score']
            most_robust = sensitivity_series.idxmin()
            most_sensitive = sensitivity_series.idxmax()
            f.write(f"- **Most Robust to Angle Change**: {most_robust} (Sensitivity Score: {sensitivity_series.min():.3f})\n")
            f.write(f"- **Most Sensitive to Angle Change**: {most_sensitive} (Sensitivity Score: {sensitivity_series.max():.3f})\n\n")
        
        if 'consistency_df' in locals():
            consistency_series = consistency_df['mean_std']
            most_consistent = consistency_series.idxmin()
            least_consistent = consistency_series.idxmax()
            f.write(f"- **Most Consistent Across Textures**: {most_consistent} (Standard Deviation: {consistency_series.min():.3f})\n")
            f.write(f"- **Least Consistent Across Textures**: {least_consistent} (Standard Deviation: {consistency_series.max():.3f})\n\n")
        
        if args.enable_aasr and 'global_df' in locals() and len(global_df) > 0:
            avg_aasr_row = global_df[global_df['Folder'] == 'Average'].iloc[0]
            f.write(f"- **Global Average AASR**: {avg_aasr_row['Average_AASR']:.2f}%\n")
            
            # Find folders with highest and lowest AASR
            non_avg_rows = global_df[global_df['Folder'] != 'Average']
            if len(non_avg_rows) > 0:
                best_attack_folder = non_avg_rows.loc[non_avg_rows['Average_AASR'].idxmax(), 'Folder']
                best_attack_aasr = non_avg_rows['Average_AASR'].max()
                worst_attack_folder = non_avg_rows.loc[non_avg_rows['Average_AASR'].idxmin(), 'Folder']
                worst_attack_aasr = non_avg_rows['Average_AASR'].min()
                
                f.write(f"- **Highest Attack Success Rate**: {best_attack_folder} (AASR: {best_attack_aasr:.2f}%)\n")
                f.write(f"- **Lowest Attack Success Rate**: {worst_attack_folder} (AASR: {worst_attack_aasr:.2f}%)\n\n")
        
        f.write('## Environment Impact\n\n')
        f.write('Compositing stop signs into real-world background environments may affect detection performance. Main impacts include:\n\n')
        f.write('- Background noise and distractions may reduce detection confidence\n')
        f.write('- Background colors and textures may interact with stop sign features\n')
        f.write('- Environmental lighting and shadow effects may affect feature extraction\n')
        f.write('- Complex backgrounds may lead to higher false positive or false negative rates\n\n')
        
        f.write('## Detailed Results\n\n')
        f.write('All detailed analysis results and raw data can be found in the following CSV files:\n\n')
        f.write('- `all_folders_summary.csv` - Summary statistics for all folders\n')
        f.write('- `all_folders_angle_data.csv` - Angle-confidence data for all folders\n')
        f.write('- `angle_sensitivity_analysis.csv` - Angle sensitivity analysis\n')
        f.write('- `texture_consistency_analysis.csv` - Texture consistency analysis\n')
        f.write('- `grouped_angle_data.csv` - Angle-confidence data grouped by modification category\n')
        f.write('- `angle_range_stats.csv` - Statistics by angle range\n')
        f.write(f'- `{args.detector}_performance_summary.csv` - Detector environment performance summary\n')
        
        if args.enable_aasr:
            f.write('- `global_aasr_summary.csv` - Global AASR summary (attack success rate analysis)\n')
        f.write('\n')
        
        f.write('## Visualization Results\n\n')
        f.write('Detailed visual analysis can be found in the following image files:\n\n')
        f.write('- `folder_avg_front_confidence.png` - Average front confidence by folder\n')
        f.write('- `folder_confidence_stats.png` - Confidence statistics by folder\n')
        f.write('- `all_folders_angle_confidence.png` - Angle-confidence comparison across folders\n')
        f.write('- `folder_confidence_violinplot.png` - Angle confidence distribution by folder (violin plot)\n')
        f.write('- `angle_sensitivity_scores.png` - Angle sensitivity scores by folder\n')
        f.write('- `texture_consistency.png` - Texture consistency within folders\n')
        f.write('- `confidence_heatmap.png` - Confidence heatmap by folder and angle\n')
        
        f.write('\n## Grouped Visualization Results\n\n')
        f.write('Visual analysis grouped by modification category can be found in the following image files:\n\n')
        f.write('- `grouped_angle_confidence.png` - Angle-confidence comparison across categories\n')
        f.write('- `grouped_front_confidence.png` - Average front confidence by category\n')
        f.write('- `grouped_confidence_stats.png` - Confidence statistics by category\n')
        f.write('- `grouped_sensitivity_scores.png` - Angle sensitivity scores by category\n')
        f.write('- `grouped_texture_consistency.png` - Texture consistency within categories\n')
        f.write('- `grouped_confidence_violinplot.png` - Angle confidence distribution by category (violin plot)\n')
        f.write('- `grouped_confidence_heatmap.png` - Confidence heatmap by category and angle\n')
        
        f.write('\n## Category Classification Description\n\n')
        f.write('For grouped visualizations, folders are classified into the following categories based on modification types:\n\n')
        f.write('- **Origin**: Original stop sign, no modifications\n')
        f.write('- **S**: Shape modification only (square or triangle)\n')
        f.write('- **C**: Color modification only (blue or yellow)\n')
        f.write('- **T**: Text modification only (with "abcd", "hello", or "world")\n')
        f.write('- **P**: Pattern modification only (checkerboard or polkadot)\n')
        f.write('- **S+C**: Shape and color modifications\n')
        f.write('- **S+T**: Shape and text modifications\n')
        f.write('- **S+P**: Shape and pattern modifications\n')
        f.write('- **C+T**: Color and text modifications\n')
        f.write('- **C+P**: Color and pattern modifications\n')
        f.write('- **T+P**: Text and pattern modifications\n')
        f.write('- **S+C+T**: Shape, color and text modifications\n')
        f.write('- **S+C+P**: Shape, color and pattern modifications\n')
        f.write('- **S+T+P**: Shape, text and pattern modifications\n')
        f.write('- **C+T+P**: Color, text and pattern modifications\n')
        f.write('- **ALL**: All modifications (shape, color, text and pattern)\n')
        
        f.write('\n## Methodology\n\n')
        f.write('### Rendering and Environment Compositing Method\n\n')
        f.write('This analysis uses PerspectiveViewGenerator to render texture images from different angles, then composites them into real-world background environments. This approach builds on\n')
        f.write('directly applying perspective transformations to images by adding integration with the environment, more closely resembling the visual effect in real-world scenes. Angle range is from -90° to 90° (0° is front view).\n\n')
        
        f.write('### Angle Sensitivity Calculation\n\n')
        f.write('Angle sensitivity is measured by calculating the percentage drop in confidence from front view (0°) to moderate side views (-45° and 45°). Lower scores indicate greater robustness to angle changes.\n')
        f.write('Calculation formula: sensitivity_score = (front_to_left_drop + front_to_right_drop) / 2\n\n')
        
        f.write('### Texture Consistency Calculation\n\n')
        f.write('Texture consistency is measured by calculating the standard deviation of confidence across different textures within the same folder at each angle.\n')
        f.write('The average of all angle standard deviations is then taken as the consistency metric. Lower values indicate higher consistency across different textures.\n\n')
        
        if args.enable_aasr:
            f.write('### AASR (Angular Attack Success Rate) Calculation\n\n')
            f.write('AASR measures the ability of adversarial samples to maintain effectiveness across different angles. Calculation method:\n')
            f.write('- For a given confidence threshold, count the number of angles where confidence exceeds the threshold (attack success = detection success)\n')
            f.write('- AASR = (Number of successful attack angles / Total angles) × 100%\n')
            f.write('- Higher AASR indicates adversarial sample is successfully detected by detector at more angles (meaning adversarial attack failed)\n')
            f.write('- Lower AASR indicates adversarial sample evades detection at more angles (meaning adversarial attack succeeded)\n\n')
        
        f.write('### Environment Compositing Process\n\n')
        f.write('The environment compositing process uses the following steps:\n\n')
        f.write('1. Load background image and resize to specified size\n')
        f.write('2. Render stop sign texture from different angles\n')
        f.write('3. Place rendered texture at center of background image\n')
        f.write('4. Create blending mask to ensure smooth edge transition\n')
        f.write('5. Use selected detector to detect stop signs in composited images\n')
        f.write('6. Record and analyze detection confidence\n')
        if args.enable_aasr:
            f.write('7. Calculate AASR metrics to evaluate adversarial attack success rate\n')
        f.write('\n')
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Environment compositing analysis summary complete, report saved to: {report_path}")
    print(f"\nProcessed {len(all_folders)} folders in {elapsed_time:.2f} seconds")
    print(f"Summary results saved to {summary_output_dir}")

if __name__ == "__main__":
    main()