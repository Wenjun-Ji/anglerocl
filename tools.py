import os
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import logging


# Custom utility functions
import shutil
import inspect
import json
import sys

def save_execution_code(output_dir, script_path=None, additional_files=None):
    """
    Save the currently executing code to the output directory
    
    Args:
        output_dir (str): Output directory path
        script_path (str): Main script path, auto-detect if None
        additional_files (list): List of additional files to save
    """
    # Create code saving directory
    code_dir = os.path.join(output_dir, "source_code")
    os.makedirs(code_dir, exist_ok=True)
    
    # 1. Save main execution script
    if script_path is None:
        # Automatically get current executing script path
        script_path = inspect.getfile(inspect.currentframe().f_back)
    
    if os.path.exists(script_path):
        script_name = os.path.basename(script_path)
        shutil.copy2(script_path, os.path.join(code_dir, script_name))
        print(f"✓ Saved main script: {script_name}")
    
    # 2. Save related Python files
    script_dir = os.path.dirname(os.path.abspath(script_path))
    
    # Auto-detect and save Python files in the same directory
    for file in os.listdir(script_dir):
        if file.endswith('.py') and file != os.path.basename(script_path):
            src_path = os.path.join(script_dir, file)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, os.path.join(code_dir, file))
                print(f"✓ Saved related file: {file}")
    
    # 3. Save additionally specified files
    if additional_files:
        for file_path in additional_files:
            if os.path.exists(file_path):
                file_name = os.path.basename(file_path)
                shutil.copy2(file_path, os.path.join(code_dir, file_name))
                print(f"✓ Saved additional file: {file_name}")
    
    # 4. Save command line arguments
    if hasattr(sys, 'argv'):
        cmd_info = {
            "command_line": " ".join(sys.argv),
            "script_name": os.path.basename(script_path),
            "working_directory": os.getcwd(),
            "arguments": sys.argv[1:] if len(sys.argv) > 1 else []
        }
        
        with open(os.path.join(code_dir, "execution_info.json"), "w", encoding='utf-8') as f:
            json.dump(cmd_info, f, indent=4, ensure_ascii=False)
        print("✓ Saved execution info: execution_info.json")
    
    print(f"\n📁 All source code saved to: {code_dir}")
    return code_dir


def check_grad_status(tensor, name, debug_mode=True):
    """Check tensor gradient status and print detailed information, only output details in debug mode"""
    if not debug_mode:
        return tensor.requires_grad and tensor.grad_fn is not None
        
    if tensor is None:
        print(f"[WARNING] {name} is None!")
        return False
    
    requires_grad = tensor.requires_grad
    has_grad_fn = tensor.grad_fn is not None
    grad_fn_type = type(tensor.grad_fn).__name__ if has_grad_fn else "None"
    
    print(f"\n=== Gradient Check: {name} ===")
    print(f"Shape: {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    print(f"Device: {tensor.device}")
    print(f"Requires grad: {requires_grad}")
    print(f"Has grad_fn: {has_grad_fn}")
    print(f"Grad_fn type: {grad_fn_type}")
    print(f"Is leaf: {tensor.is_leaf}")
    print(f"Value range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
    print("=====================================\n")
    
    return requires_grad and has_grad_fn

# Add this near the top of your script, after the imports
def setup_logging(logging_dir):
    """
    Sets up logging to both console and file
    """
    # Create logs directory if it doesn't exist
    log_dir = logging_dir
    os.makedirs(log_dir, exist_ok=True)  # Ensure log directory exists
    
    # Create a timestamped log filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
                                 datefmt='%m/%d/%Y %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return log_file

# Create enhanced single prompt analysis plot (showing variance and max/min values)
def create_enhanced_single_prompt_analysis(single_results, train_angles, test_angles, args, prompt_template, save_path):
    """Create enhanced analysis plot for single prompt, including sample variance and max/min values"""
    fig, axs = plt.subplots(2, 1, figsize=(14, 12))
    
    # Sort angles
    angles = sorted(test_angles)
    train_idxs = [angles.index(a) for a in train_angles]
    
    # Plot confidence comparison
    for key, color, marker, label in [
        ("placeholder_token", 'b', 'o', f'with {args.placeholder_token}'),
        ("initial_token", 'r', 'x', f'with {args.initializer_token}'),
        ("clean", 'g', 's', 'clean prompt')
    ]:
        # Get confidence values, standard deviations, and max/min values
        y = [single_results[key]["confidences"][a] for a in angles]
        y_err = [single_results[key]["std_devs"][a] for a in angles]
        y_max = [single_results[key]["max_values"][a] for a in angles]
        y_min = [single_results[key]["min_values"][a] for a in angles]
        
        # Plot average line
        axs[0].plot(angles, y, color=color, marker=marker, label=f'{label} (avg)')
        
        # Plot standard deviation range
        axs[0].fill_between(angles, 
                           [max(0, y[i] - y_err[i]) for i in range(len(y))], 
                           [min(1, y[i] + y_err[i]) for i in range(len(y))], 
                           color=color, alpha=0.2)
        
        # Plot max/min values
        axs[0].plot(angles, y_max, color=color, linestyle='--', alpha=0.5, 
                   label=f'{label} (max)' if key == "placeholder_token" else "")
        axs[0].plot(angles, y_min, color=color, linestyle=':', alpha=0.5, 
                   label=f'{label} (min)' if key == "placeholder_token" else "")
    
    axs[0].axhline(y=args.detection_threshold, color='k', linestyle='--', label=f'threshold ({args.detection_threshold})')
    axs[0].set_title(f'Confidence Across Angles for: "{prompt_template}" (with variance)')
    axs[0].set_xlabel('Angle')
    axs[0].set_ylabel('Confidence')
    axs[0].set_ylim(0, 1)
    axs[0].legend()
    axs[0].grid(True)
    
    # Mark training angles
    for i in train_idxs:
        axs[0].axvline(x=angles[i], color='gray', linestyle=':', alpha=0.5)
    
    # Plot detection rate comparison
    for key, color, marker, label in [
        ("placeholder_token", 'b', 'o', f'with {args.placeholder_token}'),
        ("initial_token", 'r', 'x', f'with {args.initializer_token}'),
        ("clean", 'g', 's', 'clean prompt')
    ]:
        axs[1].plot(angles, 
                    [single_results[key]["detection_rates"][a] for a in angles], 
                    color=color, marker=marker, label=label)
    
    axs[1].set_title(f'Detection Rate Across Angles for: "{prompt_template}"')
    axs[1].set_xlabel('Angle')
    axs[1].set_ylabel('Detection Rate')
    axs[1].set_ylim(-0.1, 1.1)
    axs[1].legend()
    axs[1].grid(True)
    
    # Mark training angles
    for i in train_idxs:
        axs[1].axvline(x=angles[i], color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Create enhanced variance distribution plot
def create_enhanced_variance_distribution_plot(results, test_angles, args, save_path, num_samples):
    """Create a more detailed variance distribution plot, showing distribution of all samples"""
    # Calculate confidence distribution for each angle and each method
    fig, axs = plt.subplots(len(test_angles), 3, figsize=(18, 4*len(test_angles)))
    
    # If test_angles has only one element, matplotlib will reduce axs to one dimension, need to handle
    if len(test_angles) == 1:
        axs = np.array([axs])
    
    for i, angle in enumerate(sorted(test_angles)):
        for j, (key, title) in enumerate([
            ("placeholder_token", f"With {args.placeholder_token}"),
            ("initial_token", f"With {args.initializer_token}"),
            ("clean", "Clean Prompt")
        ]):
            # Get all confidence values for all prompts and all samples at this angle
            all_samples = results[key]["samples"][angle]
            
            # Plot histogram
            axs[i, j].hist(all_samples, bins=20, range=(0, 1), alpha=0.7, color=["blue", "red", "green"][j])
            axs[i, j].axvline(x=args.detection_threshold, color='black', linestyle='--', 
                             label=f'Detection Threshold ({args.detection_threshold})')
            axs[i, j].axvline(x=np.mean(all_samples), color=['darkblue', 'darkred', 'darkgreen'][j], 
                             linestyle='-', label=f'Mean ({np.mean(all_samples):.3f})')
            
            # Add standard deviation indicator lines
            std_dev = np.std(all_samples)
            mean_val = np.mean(all_samples)
            axs[i, j].axvline(x=mean_val + std_dev, color=['blue', 'red', 'green'][j], 
                             linestyle='--', alpha=0.5, label=f'±1σ ({std_dev:.3f})')
            axs[i, j].axvline(x=mean_val - std_dev, color=['blue', 'red', 'green'][j], 
                             linestyle='--', alpha=0.5)
            
            # Set title and axis labels
            axs[i, j].set_title(f"{title} at Angle {angle}° (n={len(all_samples)})")
            axs[i, j].set_xlabel("Confidence Score")
            axs[i, j].set_ylabel("Number of Samples")
            axs[i, j].set_xlim(0, 1)
            axs[i, j].legend()
            
            # Calculate percentage above threshold
            detection_rate = sum([1 for c in all_samples if c >= args.detection_threshold]) / len(all_samples)
            axs[i, j].text(0.05, 0.9, f"Detection Rate: {detection_rate:.2f}", 
                          transform=axs[i, j].transAxes, bbox=dict(facecolor='white', alpha=0.7))
            
            # Add sample count and variance info
            axs[i, j].text(0.05, 0.8, 
                          f"Samples: {len(all_samples)} ({num_samples}/prompt)\nStd Dev: {std_dev:.3f}", 
                          transform=axs[i, j].transAxes, bbox=dict(facecolor='white', alpha=0.7))
    
    # Save image - make sure to add these lines
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
            
def create_enhanced_all_prompts_performance_plot(results, test_angles, args, save_path):
    """Create an enhanced line plot showing average performance and sample variance for all prompts"""
    fig, axs = plt.subplots(3, 1, figsize=(14, 18))
    
    # Sort angles
    angles = sorted(test_angles)
    
    for i, (key, title, color) in enumerate([
        ("placeholder_token", f"With {args.placeholder_token}", "blue"),
        ("initial_token", f"With {args.initializer_token}", "red"),
        ("clean", "Clean Prompt", "green")
    ]):
        # Get number of all prompts
        num_prompts = len(results[key]["confidences"][angles[0]])
        
        # Draw a semi-transparent line for each prompt (using average values)
        for p in range(num_prompts):
            prompt_values = [results[key]["confidences"][angle][p] for angle in angles]
            axs[i].plot(angles, prompt_values, color=color, alpha=0.2, linewidth=0.5)
        
        # Draw average line for all prompts
        avg_values = [np.mean(results[key]["confidences"][angle]) for angle in angles]
        axs[i].plot(angles, avg_values, color=color, linewidth=3, label="Average across prompts")
        
        # Calculate and plot standard deviation range of averages (variation between prompts)
        std_values = [np.std(results[key]["confidences"][angle]) for angle in angles]
        axs[i].fill_between(angles, 
                           [max(0, avg_values[j] - std_values[j]) for j in range(len(angles))], 
                           [min(1, avg_values[j] + std_values[j]) for j in range(len(angles))], 
                           color=color, alpha=0.2)
        
        # Plot variance range between samples (variation among samples)
        sample_std_values = [np.std(results[key]["samples"][angle]) for angle in angles]
        avg_sample_values = [np.mean(results[key]["samples"][angle]) for angle in angles]
        axs[i].fill_between(angles, 
                           [max(0, avg_sample_values[j] - sample_std_values[j]) for j in range(len(angles))], 
                           [min(1, avg_sample_values[j] + sample_std_values[j]) for j in range(len(angles))], 
                           color=color, alpha=0.1, hatch='///',
                           label="Sample variance" if i == 0 else None)
        
        # Draw confidence threshold line
        axs[i].axhline(y=args.detection_threshold, color='black', linestyle='--', 
                      label=f'Detection Threshold ({args.detection_threshold})')
        
        # Add sample count information
        num_samples_per_prompt = len(results[key]["samples"][angles[0]]) // num_prompts
        total_samples = len(results[key]["samples"][angles[0]])
        
        # Set title and axis labels
        axs[i].set_title(f"{title} - Performance Across Angles ({num_prompts} prompts, {num_samples_per_prompt} samples/prompt, {total_samples} total)")
        axs[i].set_xlabel("Angle")
        axs[i].set_ylabel("Confidence Score")
        axs[i].set_ylim(0, 1)
        axs[i].legend()
        axs[i].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Create enhanced comparison summary plot (including sample variance information)
def create_enhanced_comparison_summary(avg_results, results, train_angles, test_angles, args, save_path):
    """Create enhanced comparison summary plot, including intra-sample and inter-sample variance information"""
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Sort angles
    angles = sorted(test_angles)
    train_idxs = [angles.index(a) for a in train_angles]
    test_idxs = [i for i in range(len(angles)) if i not in train_idxs]
    
    # Calculate confidence improvement (relative to initializer word and clean prompt)
    conf_imp_vs_initial = {angle: avg_results["placeholder_token"]["avg_confidences"][angle] - 
                          avg_results["initial_token"]["avg_confidences"][angle] for angle in angles}
    conf_imp_vs_clean = {angle: avg_results["placeholder_token"]["avg_confidences"][angle] - 
                        avg_results["clean"]["avg_confidences"][angle] for angle in angles}
    
    # Plot confidence comparison
    for key, color, marker, label_base in [
        ("placeholder_token", 'b', 'o', f'with {args.placeholder_token}'),
        ("initial_token", 'r', 'x', f'with {args.initializer_token}'),
        ("clean", 'g', 's', 'clean prompt')
    ]:
        # Get average confidence
        y = [avg_results[key]["avg_confidences"][a] for a in angles]
        
        # Get standard deviation between prompts
        y_prompt_err = [avg_results[key]["std_avg_confidences"][a] for a in angles]
        
        # Get standard deviation among samples (all samples)
        y_sample_err = [avg_results[key]["std_all_samples"][a] for a in angles]
        
        # Get max and min values (all samples)
        y_max = [avg_results[key]["max_all_samples"][a] for a in angles]
        y_min = [avg_results[key]["min_all_samples"][a] for a in angles]
        
        # Plot average line
        axs[0, 0].plot(angles, y, color=color, marker=marker, label=f'{label_base} (avg)')
        
        # Plot inter-prompt standard deviation range
        axs[0, 0].fill_between(angles, 
                              [max(0, y[i] - y_prompt_err[i]) for i in range(len(y))], 
                              [min(1, y[i] + y_prompt_err[i]) for i in range(len(y))], 
                              color=color, alpha=0.2, label=f'{label_base} (prompt σ)' if key == "placeholder_token" else None)
        
        # Plot inter-sample standard deviation range (with different fill style)
        axs[0, 0].fill_between(angles, 
                              [max(0, y[i] - y_sample_err[i]) for i in range(len(y))], 
                              [min(1, y[i] + y_sample_err[i]) for i in range(len(y))], 
                              color=color, alpha=0.1, hatch='///',
                              label=f'{label_base} (sample σ)' if key == "placeholder_token" else None)
        
        # Plot max/min values
        axs[0, 0].plot(angles, y_max, color=color, linestyle='--', alpha=0.5, 
                      label=f'{label_base} (max)' if key == "placeholder_token" else None)
        axs[0, 0].plot(angles, y_min, color=color, linestyle=':', alpha=0.5, 
                      label=f'{label_base} (min)' if key == "placeholder_token" else None)
    
    axs[0, 0].axhline(y=args.detection_threshold, color='k', linestyle='--', label=f'threshold ({args.detection_threshold})')
    axs[0, 0].set_title('Confidence Across Angles (with sample & prompt variance)')
    axs[0, 0].set_xlabel('Angle')
    axs[0, 0].set_ylabel('Confidence')
    axs[0, 0].set_ylim(0, 1)
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Mark training angles
    for i in train_idxs:
        axs[0, 0].axvline(x=angles[i], color='gray', linestyle=':', alpha=0.5)
    
    # Plot detection rate comparison
    for key, color, marker, label_base in [
        ("placeholder_token", 'b', 'o', f'with {args.placeholder_token}'),
        ("initial_token", 'r', 'x', f'with {args.initializer_token}'),
        ("clean", 'g', 's', 'clean prompt')
    ]:
        y = [avg_results[key]["avg_detection_rates"][a] for a in angles]
        y_err = [avg_results[key]["std_detection_rates"][a] for a in angles]
        y_max = [avg_results[key]["max_detection_rates"][a] for a in angles]
        y_min = [avg_results[key]["min_detection_rates"][a] for a in angles]
        
        axs[0, 1].plot(angles, y, color=color, marker=marker, label=f'{label_base} (avg)')
        axs[0, 1].fill_between(angles, 
                              [max(0, y[i] - y_err[i]) for i in range(len(y))], 
                              [min(1, y[i] + y_err[i]) for i in range(len(y))], 
                              color=color, alpha=0.2)
        axs[0, 1].plot(angles, y_max, color=color, linestyle='--', alpha=0.5, 
                      label=f'{label_base} (max)' if key == "placeholder_token" else None)
        axs[0, 1].plot(angles, y_min, color=color, linestyle=':', alpha=0.5, 
                      label=f'{label_base} (min)' if key == "placeholder_token" else None)
    
    axs[0, 1].set_title('Detection Rate Across Angles (with variance)')
    axs[0, 1].set_xlabel('Angle')
    axs[0, 1].set_ylabel('Detection Rate')
    axs[0, 1].set_ylim(0, 1)
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Mark training angles
    for i in train_idxs:
        axs[0, 1].axvline(x=angles[i], color='gray', linestyle=':', alpha=0.5)
    
    # Plot improvement relative to initializer word
    bar_width = 0.35
    bar_positions = np.arange(len(angles))
    
    # Calculate sample-level improvement standard deviation
    # For each angle, calculate improvement standard deviation among all samples
    sample_std_imp_vs_initial = []
    sample_std_imp_vs_clean = []
    
    for angle in angles:
        # Get all placeholder token and initial token samples
        placeholder_samples = avg_results["placeholder_token"]["all_samples"][angle]
        initial_samples = avg_results["initial_token"]["all_samples"][angle]
        clean_samples = avg_results["clean"]["all_samples"][angle]
        
        # Since samples may come from different prompts, we calculate mean and std of improvement separately
        min_samples = min(len(placeholder_samples), len(initial_samples), len(clean_samples))
        
        # Take same number of samples for comparison (this is a simplification, real case may need more complex pairing)
        imp_vs_initial_values = [placeholder_samples[i] - initial_samples[i] for i in range(min_samples)]
        imp_vs_clean_values = [placeholder_samples[i] - clean_samples[i] for i in range(min_samples)]
        
        sample_std_imp_vs_initial.append(np.std(imp_vs_initial_values))
        sample_std_imp_vs_clean.append(np.std(imp_vs_clean_values))
    
    # Plot bar chart (with error bars)
    axs[1, 0].bar(bar_positions - bar_width/2, 
                 [conf_imp_vs_initial[a] for a in angles], 
                 bar_width, 
                 label=f'vs {args.initializer_token}',
                 yerr=sample_std_imp_vs_initial,
                 capsize=5,
                 color='blue',
                 alpha=0.7)
    
    axs[1, 0].bar(bar_positions + bar_width/2, 
                 [conf_imp_vs_clean[a] for a in angles], 
                 bar_width, 
                 label='vs clean',
                 yerr=sample_std_imp_vs_clean,
                 capsize=5,
                 color='green',
                 alpha=0.7)
    
    axs[1, 0].set_title(f'Confidence Improvement of {args.placeholder_token} (with sample variance)')
    axs[1, 0].set_xlabel('Angle')
    axs[1, 0].set_ylabel('Confidence Improvement')
    axs[1, 0].set_xticks(bar_positions)
    axs[1, 0].set_xticklabels(angles)
    axs[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.5)
    axs[1, 0].legend()
    
    # Calculate overall improvement percentage
    train_imp_vs_initial = [conf_imp_vs_initial[angles[i]] for i in train_idxs]
    test_imp_vs_initial = [conf_imp_vs_initial[angles[i]] for i in test_idxs]
    train_imp_vs_clean = [conf_imp_vs_clean[angles[i]] for i in train_idxs]
    test_imp_vs_clean = [conf_imp_vs_clean[angles[i]] for i in test_idxs]
    
    # Create enhanced summary text
    summary_text = create_enhanced_summary_text(avg_results, train_angles, test_angles, args, len(placeholder_samples) // len(results["placeholder_token"]["prompts"]))
    
    axs[1, 1].axis('off')
    axs[1, 1].text(0.1, 0.5, summary_text, fontsize=9, va='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# Create enhanced summary text, including sample information
def create_enhanced_summary_text(avg_results, train_angles, test_angles, args, num_samples_per_prompt):
    """Create enhanced summary text including sample statistics"""
    # Sort angles
    angles = sorted(test_angles)
    train_idxs = [angles.index(a) for a in train_angles]
    test_idxs = [i for i in range(len(angles)) if i not in train_idxs]
    
    # Calculate confidence improvement
    conf_imp_vs_initial = {angle: avg_results["placeholder_token"]["avg_confidences"][angle] - 
                          avg_results["initial_token"]["avg_confidences"][angle] for angle in angles}
    conf_imp_vs_clean = {angle: avg_results["placeholder_token"]["avg_confidences"][angle] - 
                        avg_results["clean"]["avg_confidences"][angle] for angle in angles}
    
    # Calculate overall improvement percentage
    train_imp_vs_initial = [conf_imp_vs_initial[angles[i]] for i in train_idxs]
    test_imp_vs_initial = [conf_imp_vs_initial[angles[i]] for i in test_idxs]
    train_imp_vs_clean = [conf_imp_vs_clean[angles[i]] for i in train_idxs]
    test_imp_vs_clean = [conf_imp_vs_clean[angles[i]] for i in test_idxs]
    
    # Sample information
    num_prompts = len(avg_results["placeholder_token"]["avg_confidences"])
    total_samples = num_prompts * num_samples_per_prompt
    
    # Sample variance information - average sample standard deviation per angle
    sample_std_placeholder = np.mean([avg_results["placeholder_token"]["std_all_samples"][a] for a in angles])
    sample_std_initial = np.mean([avg_results["initial_token"]["std_all_samples"][a] for a in angles])
    sample_std_clean = np.mean([avg_results["clean"]["std_all_samples"][a] for a in angles])
    
    # Maximum sample variance - find angle with maximum sample variance
    max_std_placeholder_angle = max(angles, key=lambda a: avg_results["placeholder_token"]["std_all_samples"][a])
    max_std_placeholder = avg_results["placeholder_token"]["std_all_samples"][max_std_placeholder_angle]
    
    max_std_initial_angle = max(angles, key=lambda a: avg_results["initial_token"]["std_all_samples"][a])
    max_std_initial = avg_results["initial_token"]["std_all_samples"][max_std_initial_angle]
    
    max_std_clean_angle = max(angles, key=lambda a: avg_results["clean"]["std_all_samples"][a])
    max_std_clean = avg_results["clean"]["std_all_samples"][max_std_clean_angle]
    
    # Prompt variance information - average standard deviation between prompts
    prompt_std_placeholder = np.mean([avg_results["placeholder_token"]["std_avg_confidences"][a] for a in angles])
    prompt_std_initial = np.mean([avg_results["initial_token"]["std_avg_confidences"][a] for a in angles])
    prompt_std_clean = np.mean([avg_results["clean"]["std_avg_confidences"][a] for a in angles])
    
    # Max/min value information - global max/min of all samples
    placeholder_max = max([avg_results["placeholder_token"]["max_all_samples"][a] for a in angles])
    placeholder_min = min([avg_results["placeholder_token"]["min_all_samples"][a] for a in angles])
    initial_max = max([avg_results["initial_token"]["max_all_samples"][a] for a in angles])
    initial_min = min([avg_results["initial_token"]["min_all_samples"][a] for a in angles])
    clean_max = max([avg_results["clean"]["max_all_samples"][a] for a in angles])
    clean_min = min([avg_results["clean"]["min_all_samples"][a] for a in angles])
    
    summary_text = (
        f"Sample Information:\n"
        f"  Prompts: {num_prompts}, Samples/prompt: {num_samples_per_prompt}, Total: {total_samples}\n\n"
        f"Overall Improvement:\n"
        f"vs {args.initializer_token}:\n"
        f"  All angles: {sum(conf_imp_vs_initial.values())/len(conf_imp_vs_initial):.4f}\n"
        f"  Train angles: {sum(train_imp_vs_initial)/len(train_imp_vs_initial):.4f}\n"
        f"  Test angles: {sum(test_imp_vs_initial)/len(test_imp_vs_initial):.4f}\n\n"
        f"vs clean:\n"
        f"  All angles: {sum(conf_imp_vs_clean.values())/len(conf_imp_vs_clean):.4f}\n"
        f"  Train angles: {sum(train_imp_vs_clean)/len(train_imp_vs_clean):.4f}\n"
        f"  Test angles: {sum(test_imp_vs_clean)/len(test_imp_vs_clean):.4f}\n\n"
        f"Detection Rates:\n"
        f"  Train angles: {sum([avg_results['placeholder_token']['avg_detection_rates'][angles[i]] for i in train_idxs])/len(train_idxs):.2f} vs "
        f"{sum([avg_results['initial_token']['avg_detection_rates'][angles[i]] for i in train_idxs])/len(train_idxs):.2f} vs "
        f"{sum([avg_results['clean']['avg_detection_rates'][angles[i]] for i in train_idxs])/len(train_idxs):.2f}\n"
        f"  Test angles: {sum([avg_results['placeholder_token']['avg_detection_rates'][angles[i]] for i in test_idxs])/len(test_idxs):.2f} vs "
        f"{sum([avg_results['initial_token']['avg_detection_rates'][angles[i]] for i in test_idxs])/len(test_idxs):.2f} vs "
        f"{sum([avg_results['clean']['avg_detection_rates'][angles[i]] for i in test_idxs])/len(test_idxs):.2f}\n\n"
        f"Sample Variance Analysis:\n"
        f"  Avg Sample StdDev ({args.placeholder_token}): {sample_std_placeholder:.4f}\n"
        f"  Avg Sample StdDev ({args.initializer_token}): {sample_std_initial:.4f}\n"
        f"  Avg Sample StdDev (clean): {sample_std_clean:.4f}\n\n"
        f"  Max Sample StdDev ({args.placeholder_token}): {max_std_placeholder:.4f} at {max_std_placeholder_angle}°\n"
        f"  Max Sample StdDev ({args.initializer_token}): {max_std_initial:.4f} at {max_std_initial_angle}°\n"
        f"  Max Sample StdDev (clean): {max_std_clean:.4f} at {max_std_clean_angle}°\n\n"
        f"Prompt Variance Analysis:\n"
        f"  Avg Prompt StdDev ({args.placeholder_token}): {prompt_std_placeholder:.4f}\n"
        f"  Avg Prompt StdDev ({args.initializer_token}): {prompt_std_initial:.4f}\n"
        f"  Avg Prompt StdDev (clean): {prompt_std_clean:.4f}\n\n"
        f"Range Analysis (all samples):\n"
        f"  {args.placeholder_token}: [{placeholder_min:.4f}, {placeholder_max:.4f}]\n"
        f"  {args.initializer_token}: [{initial_min:.4f}, {initial_max:.4f}]\n"
        f"  clean: [{clean_min:.4f}, {clean_max:.4f}]"
    )
    
    return summary_text

def analyze_token_similarities(text_encoder, tokenizer, placeholder_token_ids, top_k=30):
    """Analyze similarity between placeholder token and other words in vocabulary"""
    # Get placeholder token embedding
    placeholder_embeds = text_encoder.get_input_embeddings().weight[min(placeholder_token_ids)].detach()
    
    # Calculate similarity with all vocabulary
    all_embeds = text_encoder.get_input_embeddings().weight.detach()
    similarities = F.cosine_similarity(placeholder_embeds.unsqueeze(0), all_embeds)
    
    # Get top_k most similar words
    values, indices = torch.topk(similarities, top_k)
    
    # Convert token id to words
    similar_tokens = []
    for idx, sim_val in zip(indices.cpu().numpy(), values.cpu().numpy()):
        token = tokenizer.convert_ids_to_tokens(int(idx))
        similar_tokens.append((token, sim_val))
    
    return similar_tokens

def analyze_semantic_clusters(text_encoder, tokenizer, placeholder_token_ids):
    """Analyze relationship between placeholder token and predefined semantic groups"""
    # Define semantic concept groups
    concept_groups = {
        'visual_attributes': ['color', 'shape', 'texture', 'pattern', 'appearance'],
        'spatial_concepts': ['angle', 'perspective', 'rotation', 'view', 'position'],
        'traffic_objects': ['sign', 'stop', 'traffic', 'road', 'signal'],
        'robustness_concepts': ['robust', 'stable', 'consistent', 'reliable', 'invariant'],
        'detection_concepts': ['detect', 'recognize', 'identify', 'observe', 'perceive']
    }
    
    # Get placeholder token embedding
    placeholder_embed = text_encoder.get_input_embeddings().weight[min(placeholder_token_ids)].detach()
    
    # Calculate average similarity with each concept group
    group_similarities = {}
    for group_name, concepts in concept_groups.items():
        # Get word id for each concept
        concept_ids = []
        for concept in concepts:
            try:
                token_id = tokenizer.encode(concept, add_special_tokens=False)[0]
                concept_ids.append(token_id)
            except:
                continue
        
        if not concept_ids:
            continue
            
        # Get concept embeddings
        concept_embeds = text_encoder.get_input_embeddings().weight[concept_ids].detach()
        
        # Calculate average similarity
        similarities = F.cosine_similarity(placeholder_embed.unsqueeze(0), concept_embeds)
        avg_similarity = similarities.mean().item()
        group_similarities[group_name] = avg_similarity
    
    return group_similarities

def analyze_semantic_directions(text_encoder, tokenizer, placeholder_token_ids):
    """Analyze projection of embedding on important semantic directions"""
    # Define some semantic contrast directions
    semantic_directions = [
        ('red', 'blue'),              # Color contrast
        ('square', 'circle'),         # Shape contrast
        ('front', 'side'),            # Viewpoint contrast
        ('close', 'far'),             # Distance contrast
        ('clear', 'blurry'),          # Clarity contrast
        ('easy', 'difficult'),        # Difficulty contrast
        ('visible', 'hidden'),        # Visibility contrast
        ('stop', 'go'),               # Traffic instruction contrast
        ('dangerous', 'safe'),        # Safety contrast
        ('straight', 'curved'),       # Shape feature contrast
        ('sharp', 'blunt'),           # Edge feature contrast
        ('day', 'night'),             # Lighting condition contrast
        ('upright', 'tilted')         # Pose contrast
    ]
    
    # Get placeholder token embedding
    placeholder_embed = text_encoder.get_input_embeddings().weight[min(placeholder_token_ids)].detach()
    
    # Calculate projection on each semantic direction
    direction_projections = {}
    for start, end in semantic_directions:
        # Get embeddings of two words
        try:
            start_id = tokenizer.encode(start, add_special_tokens=False)[0]
            end_id = tokenizer.encode(end, add_special_tokens=False)[0]
            
            start_embed = text_encoder.get_input_embeddings().weight[start_id].detach()
            end_embed = text_encoder.get_input_embeddings().weight[end_id].detach()
            
            # Calculate semantic direction
            direction = end_embed - start_embed
            direction = direction / direction.norm()
            
            # Calculate projection
            projection = torch.dot(placeholder_embed, direction).item()
            direction_projections[f"{start}->{end}"] = projection
        except:
            continue
    
    return direction_projections

def track_semantic_trajectory(logger, text_encoder, tokenizer, placeholder_token_ids, initial_embeds, global_step, output_dir, base_dir, placeholder_token):
    """Track movement trajectory of embedding in semantic space"""
    # Key concepts
    key_concepts = ['stop', 'sign', 'robust', 'angle', 'view', 'detect', 'traffic', 'perspective', 'warning']
    concept_ids = []
    valid_concepts = []
    
    # Get valid concept IDs
    for concept in key_concepts:
        try:
            token_id = tokenizer.encode(concept, add_special_tokens=False)[0]
            concept_ids.append(token_id)
            valid_concepts.append(concept)
        except:
            continue
    
    # Get current embedding
    current_embeds = text_encoder.get_input_embeddings().weight[min(placeholder_token_ids)].detach()
    
    # Calculate similarity with key concepts
    concept_embeds = text_encoder.get_input_embeddings().weight[concept_ids].detach()
    current_similarities = F.cosine_similarity(current_embeds.unsqueeze(0), concept_embeds)
    
    # Calculate initial embedding similarity with concepts
    initial_similarities = F.cosine_similarity(initial_embeds.unsqueeze(0), concept_embeds)
    
    # Calculate similarity shift
    similarity_shifts = current_similarities - initial_similarities
    
    # Create two versions of data recording:
    # 1. Global tracking directory (for trend analysis)
    global_shift_dir = os.path.join(base_dir, "global_semantic_trends")
    os.makedirs(global_shift_dir, exist_ok=True)
    
    # 2. Current epoch/step directory (for snapshot analysis)
    local_shift_dir = os.path.join(output_dir, "semantic_shifts")
    os.makedirs(local_shift_dir, exist_ok=True)
    
    # Save to global trend file
    global_shifts_file = os.path.join(global_shift_dir, "semantic_shifts.csv")
    with open(global_shifts_file, "a") as f:
        if not os.path.getsize(global_shifts_file) if os.path.exists(global_shifts_file) else 0:
            f.write("step," + ",".join(valid_concepts) + "\n")
        f.write(f"{global_step}," + ",".join([f"{sim.item():.6f}" for sim in current_similarities]) + "\n")
    
    # Create visualization of current snapshot
    plt.figure(figsize=(12, 8))
    
    # Plot current similarity
    plt.subplot(2, 1, 1)
    x = range(len(valid_concepts))
    plt.bar(x, current_similarities.cpu().numpy(), color='blue', alpha=0.7, label=f'Current ({global_step})')
    plt.bar(x, initial_similarities.cpu().numpy(), color='green', alpha=0.4, label='Initial')
    plt.xticks(x, valid_concepts, rotation=45)
    plt.ylabel('Cosine Similarity')
    plt.title(f'Semantic Similarity of {placeholder_token} to Key Concepts')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Plot similarity change
    plt.subplot(2, 1, 2)
    colors = ['green' if v >= 0 else 'red' for v in similarity_shifts.cpu().numpy()]
    plt.bar(x, similarity_shifts.cpu().numpy(), color=colors)
    plt.xticks(x, valid_concepts, rotation=45)
    plt.ylabel('Similarity Change')
    plt.title('Semantic Shift from Initial Embedding')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(local_shift_dir, "current_semantic_shift.png"))
    plt.close()
    
    # If sufficient historical data exists, plot trend
    if os.path.exists(global_shifts_file):
        try:
            import pandas as pd
            df = pd.read_csv(global_shifts_file)
            
            if len(df) > 1:  # Need at least two data points to plot trend
                plt.figure(figsize=(14, 10))
                for concept in valid_concepts:
                    if concept in df.columns:
                        plt.plot(df['step'], df[concept], marker='o', markersize=3, label=concept)
                
                plt.xlabel('Training Steps')
                plt.ylabel('Cosine Similarity')
                plt.title(f'Semantic Evolution of {placeholder_token} Embedding')
                plt.legend(loc='best')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(local_shift_dir, "semantic_trajectory.png"))
                plt.close()
        except Exception as e:
            logger.error(f"Error plotting semantic trajectory: {str(e)}")

def compare_with_initializer(logger, text_encoder, placeholder_token_ids, initializer_token_id, global_step, output_dir, base_dir, placeholder_token):
    """Compare difference between trained embedding and initializer token"""
    # Get embeddings
    placeholder_embed = text_encoder.get_input_embeddings().weight[min(placeholder_token_ids)].detach()
    initializer_embed = text_encoder.get_input_embeddings().weight[initializer_token_id].detach()
    
    # Calculate Euclidean distance and cosine similarity
    distance = torch.norm(placeholder_embed - initializer_embed).item()
    similarity = F.cosine_similarity(placeholder_embed.unsqueeze(0), initializer_embed.unsqueeze(0)).item()
    
    # Similarly create two versions of data:
    # 1. Global tracking directory
    global_comparison_dir = os.path.join(base_dir, "global_initializer_comparison")
    os.makedirs(global_comparison_dir, exist_ok=True)
    
    # 2. Current epoch/step directory
    local_comparison_dir = os.path.join(output_dir, "initializer_comparison")
    os.makedirs(local_comparison_dir, exist_ok=True)
    
    # Save to global trend file
    global_metrics_file = os.path.join(global_comparison_dir, "comparison_metrics.csv")
    with open(global_metrics_file, "a") as f:
        if not os.path.getsize(global_metrics_file) if os.path.exists(global_metrics_file) else 0:
            f.write("step,distance,similarity\n")
        f.write(f"{global_step},{distance},{similarity}\n")
    
    # Calculate component-level changes
    embed_diff = (placeholder_embed - initializer_embed).abs()
    top_k = 20  # Select 20 dimensions with largest changes
    values, indices = torch.topk(embed_diff, top_k)
    
    # Create visualization of current snapshot
    plt.figure(figsize=(10, 6))
    metrics = ['Euclidean Distance', 'Cosine Similarity']
    values_to_plot = [distance, similarity]
    colors = ['blue', 'green']
    
    plt.bar(metrics, values_to_plot, color=colors)
    plt.title(f'Comparison with Initializer Token (Step {global_step})')
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels
    for i, v in enumerate(values_to_plot):
        plt.text(i, v/2, f'{v:.4f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(local_comparison_dir, "initializer_comparison.png"))
    plt.close()
    
    # Plot dimensional difference
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_k), values.cpu().numpy())
    plt.title(f'Top {top_k} Dimensional Changes from Initializer (Step {global_step})')
    plt.xlabel('Dimension Index')
    plt.ylabel('Absolute Difference')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(local_comparison_dir, "dimension_changes.png"))
    plt.close()
    
    # If sufficient historical data exists, plot trend
    if os.path.exists(global_metrics_file):
        try:
            import pandas as pd
            df = pd.read_csv(global_metrics_file)
            
            if len(df) > 1:  # Need at least two data points to plot trend
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                ax1.plot(df['step'], df['distance'], 'b-', marker='o', markersize=3, label='Euclidean Distance')
                ax1.set_xlabel('Training Steps')
                ax1.set_ylabel('Distance')
                ax1.set_title(f'Distance from Initializer Token ({placeholder_token})')
                ax1.grid(True)
                ax1.legend()
                
                ax2.plot(df['step'], df['similarity'], 'r-', marker='o', markersize=3, label='Cosine Similarity')
                ax2.set_xlabel('Training Steps')
                ax2.set_ylabel('Similarity')
                ax2.set_title('Similarity to Initializer Token')
                ax2.grid(True)
                ax2.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(local_comparison_dir, "initializer_comparison_trend.png"))
                plt.close()
        except Exception as e:
            logger.error(f"Error plotting initializer comparison trend: {str(e)}")

def visualize_embedding_in_space(logger, text_encoder, tokenizer, placeholder_token_ids, output_dir, placeholder_token):
    """Visualize embedding relationship with other related words in semantic space"""
    from sklearn.decomposition import PCA
    import numpy as np
    
    try:
        # Get placeholder token embedding
        placeholder_embed = text_encoder.get_input_embeddings().weight[min(placeholder_token_ids)].detach().cpu().numpy()
        
        # Define related words to include in visualization
        related_concepts = [
            # Visual attributes
            'red', 'blue', 'green', 'yellow', 'square', 'circle', 'triangle', 
            # Traffic related
            'stop', 'sign', 'road', 'traffic', 'warning', 'signal',
            # Spatial concepts
            'angle', 'view', 'perspective', 'rotation', 'direction',
            # Robustness concepts
            'robust', 'stable', 'consistent', 'reliable', 'invariant',
            # Detection concepts
            'detect', 'recognize', 'identify'
        ]
        
        # Get related concept embeddings
        concept_embeds = []
        concept_labels = []
        
        for concept in related_concepts:
            try:
                token_ids = tokenizer.encode(concept, add_special_tokens=False)
                if len(token_ids) == 1:  # Only consider single token words
                    token_id = token_ids[0]
                    embed = text_encoder.get_input_embeddings().weight[token_id].detach().cpu().numpy()
                    concept_embeds.append(embed)
                    concept_labels.append(concept)
            except:
                continue
        
        # Add placeholder token
        concept_embeds.append(placeholder_embed)
        concept_labels.append(placeholder_token)
        
        # Use PCA to reduce to 2D
        if len(concept_embeds) > 2:  # Ensure enough samples
            embeds_array = np.vstack(concept_embeds)
            pca = PCA(n_components=2)
            reduced_embeds = pca.fit_transform(embeds_array)
            
            # Plot PCA
            plt.figure(figsize=(14, 10))
            
            # Plot points by category - for better visual effect
            categories = {
                'visual': ['red', 'blue', 'green', 'yellow', 'square', 'circle', 'triangle'],
                'traffic': ['stop', 'sign', 'road', 'traffic', 'warning', 'signal'],
                'spatial': ['angle', 'view', 'perspective', 'rotation', 'direction'],
                'robust': ['robust', 'stable', 'consistent', 'reliable', 'invariant'],
                'detection': ['detect', 'recognize', 'identify']
            }
            
            category_colors = {
                'visual': 'blue',
                'traffic': 'red',
                'spatial': 'green',
                'robust': 'purple',
                'detection': 'orange',
                'placeholder': 'black'
            }
            
            # Plot all points, colored by category
            for i, label in enumerate(concept_labels):
                if label == placeholder_token:
                    plt.scatter(reduced_embeds[i, 0], reduced_embeds[i, 1], c=category_colors['placeholder'], 
                                s=150, marker='*', label=placeholder_token, zorder=10)
                    plt.text(reduced_embeds[i, 0], reduced_embeds[i, 1], placeholder_token, 
                            fontsize=12, weight='bold', ha='center', va='bottom')
                else:
                    # Determine category
                    category = next((cat for cat, words in categories.items() if label in words), 'other')
                    plt.scatter(reduced_embeds[i, 0], reduced_embeds[i, 1], 
                                c=category_colors.get(category, 'gray'), s=50, alpha=0.7)
                    plt.text(reduced_embeds[i, 0], reduced_embeds[i, 1], label, fontsize=9, ha='center', va='bottom')
            
            # Add legend
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=cat)
                for cat, color in category_colors.items() if cat != 'placeholder'
            ]
            legend_elements.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='black', 
                                markersize=12, label=placeholder_token))
            
            plt.legend(handles=legend_elements, title='Categories')
            plt.title(f'PCA Visualization of {placeholder_token} in Embedding Space')
            plt.xlabel(f'PC1 (Var: {pca.explained_variance_ratio_[0]:.2f})')
            plt.ylabel(f'PC2 (Var: {pca.explained_variance_ratio_[1]:.2f})')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "embedding_pca.png"), dpi=300)
            plt.close()
            
            # Add an extra visualization: similarity heatmap with placeholder token
            plt.figure(figsize=(12, 8))
            
            # Only select regular concepts (excluding placeholder token)
            regular_concepts = concept_labels[:-1]
            
            # Calculate similarity
            placeholder_vector = placeholder_embed
            similarities = []
            for i, concept in enumerate(regular_concepts):
                sim = np.dot(concept_embeds[i], placeholder_vector) / (np.linalg.norm(concept_embeds[i]) * np.linalg.norm(placeholder_vector))
                similarities.append(sim)
            
            # Sort by similarity
            sorted_indices = np.argsort(similarities)[::-1]  # Descending order
            sorted_concepts = [regular_concepts[i] for i in sorted_indices]
            sorted_similarities = [similarities[i] for i in sorted_indices]
            
            # Create similarity heatmap
            plt.barh(range(len(sorted_concepts)), sorted_similarities, color='skyblue')
            plt.yticks(range(len(sorted_concepts)), sorted_concepts)
            plt.xlabel('Cosine Similarity')
            plt.title(f'Concept Similarity to {placeholder_token}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "concept_similarity_heatmap.png"), dpi=300)
            plt.close()
            
    except Exception as e:
        logger.error(f"Error in embedding space visualization: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def analyze_embedding_evolution(logger, text_encoder, placeholder_token_ids, global_step, output_dir, placeholder_token):
    """Analyze evolution of embedding itself, including norm, eigenvalues and other statistical properties"""
    try:
        # Get embedding
        embedding = text_encoder.get_input_embeddings().weight[min(placeholder_token_ids)].detach()
        
        # Calculate statistical properties
        norm = torch.norm(embedding).item()
        mean = embedding.mean().item()
        std = embedding.std().item()
        min_val = embedding.min().item()
        max_val = embedding.max().item()
        
        # Create storage directory
        statistics_dir = os.path.join(output_dir, "embedding_statistics")
        os.makedirs(statistics_dir, exist_ok=True)
        
        # Save statistics
        with open(os.path.join(statistics_dir, "embedding_stats.txt"), "w") as f:
            f.write(f"Embedding Statistics for {placeholder_token} at step {global_step}:\n")
            f.write(f"Norm: {norm:.6f}\n")
            f.write(f"Mean: {mean:.6f}\n")
            f.write(f"Standard Deviation: {std:.6f}\n")
            f.write(f"Min Value: {min_val:.6f}\n")
            f.write(f"Max Value: {max_val:.6f}\n")
        
        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(embedding.cpu().numpy(), bins=50, alpha=0.7, color='blue')
        plt.title(f'{placeholder_token} Embedding Values Distribution')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axvline(mean, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean:.4f}')
        plt.axvline(mean + std, color='green', linestyle='dashed', linewidth=1, label=f'Mean + StdDev: {mean+std:.4f}')
        plt.axvline(mean - std, color='green', linestyle='dashed', linewidth=1, label=f'Mean - StdDev: {mean-std:.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(statistics_dir, "embedding_distribution.png"))
        plt.close()
        
        # Create visualization of sorted values
        sorted_values, _ = torch.sort(embedding.flatten())
        plt.figure(figsize=(10, 6))
        plt.plot(sorted_values.cpu().numpy())
        plt.title(f'{placeholder_token} Embedding Sorted Values')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(statistics_dir, "embedding_sorted_values.png"))
        plt.close()
        
        return {
            "norm": norm,
            "mean": mean,
            "std": std,
            "min": min_val,
            "max": max_val
        }
    except Exception as e:
        logger.error(f"Error analyzing embedding evolution: {str(e)}")
        return {}
    
def analyze_embedding(logger, text_encoder, tokenizer, placeholder_token_ids, initializer_token_id, 
                     global_step, output_dir, base_dir, placeholder_token, initial_embed=None):
    """
    Perform comprehensive analysis on embedding and save results
    
    Parameters:
        text_encoder: Text encoder model
        tokenizer: Tokenizer
        placeholder_token_ids: Placeholder token IDs
        initializer_token_id: Initializer token ID
        global_step: Current training step (0 indicates initial state)
        output_dir: Output directory for current analysis results
        base_dir: Base directory for global analysis data
        placeholder_token: String representation of placeholder token
        initial_embed: Initial embedding state, if None use initializer token embedding
    
    Returns:
        dict: Dictionary containing analysis result summary
    """
    
    logger.info(f"Analyzing embedding at step {global_step}")
    results = {}
    
    with torch.no_grad():
        # Get current embedding
        placeholder_embed = text_encoder.get_input_embeddings().weight[min(placeholder_token_ids)].detach()
        
        # Handle initial embedding
        if initial_embed is None:
            initial_embed = text_encoder.get_input_embeddings().weight[initializer_token_id].detach().cpu()
            logger.info("Using initializer token embedding as reference")
        
        try:
            # Perform various analyses
            logger.info("Analyzing token similarities...")
            similar_tokens = analyze_token_similarities(text_encoder, tokenizer, placeholder_token_ids, top_k=50)
            results['similar_tokens'] = similar_tokens
            
            logger.info("Analyzing semantic clusters...")
            semantic_clusters = analyze_semantic_clusters(text_encoder, tokenizer, placeholder_token_ids)
            results['semantic_clusters'] = semantic_clusters
            
            logger.info("Analyzing semantic directions...")
            semantic_directions = analyze_semantic_directions(text_encoder, tokenizer, placeholder_token_ids)
            results['semantic_directions'] = semantic_directions
            
            logger.info("Comparing with initializer token...")
            compare_with_initializer(
                logger,
                text_encoder, 
                placeholder_token_ids, 
                initializer_token_id, 
                global_step, 
                output_dir,
                base_dir,
                placeholder_token
            )
            
            logger.info("Tracking semantic trajectory...")
            track_semantic_trajectory(
                logger,
                text_encoder, 
                tokenizer, 
                placeholder_token_ids, 
                initial_embed, 
                global_step, 
                output_dir,
                base_dir,
                placeholder_token
            )
            
            logger.info("Visualizing embedding in semantic space...")
            visualize_embedding_in_space(
                logger,
                text_encoder, 
                tokenizer, 
                placeholder_token_ids, 
                output_dir,
                placeholder_token
            )
            
            logger.info("Analyzing embedding evolution statistics...")
            stats = analyze_embedding_evolution(
                logger,
                text_encoder, 
                placeholder_token_ids, 
                global_step, 
                output_dir,
                placeholder_token
            )
            results['stats'] = stats
            
            # Save similar token list
            with open(os.path.join(output_dir, "similar_tokens.txt"), "w") as f:
                f.write(f"Step {global_step} - Tokens most similar to {placeholder_token}:\n")
                for token, sim in similar_tokens:
                    f.write(f"{token}: {sim:.6f}\n")
            
            # Plot similarity bar chart
            tokens, sims = zip(*similar_tokens[:20])  # Take top 20 for plotting
            plt.figure(figsize=(12, 8))
            plt.barh(tokens, sims)
            plt.xlabel('Cosine Similarity')
            plt.title(f'Tokens Most Similar to {placeholder_token} (Step {global_step})')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "similar_tokens.png"))
            plt.close()
            
            # Plot concept group similarity
            plt.figure(figsize=(10, 6))
            groups = list(semantic_clusters.keys())
            values = list(semantic_clusters.values())
            plt.bar(groups, values)
            plt.xlabel('Concept Groups')
            plt.ylabel('Average Similarity')
            plt.title(f'Semantic Cluster Analysis (Step {global_step})')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "semantic_clusters.png"))
            plt.close()
            
            # Plot semantic direction projections
            sorted_directions = sorted(semantic_directions.items(), key=lambda x: abs(x[1]), reverse=True)
            if sorted_directions:  # Ensure there is data
                directions, projections = zip(*sorted_directions[:10])  # Take top 10 directions
                
                plt.figure(figsize=(12, 8))
                plt.barh(directions, projections)
                plt.xlabel('Projection Value')
                plt.title(f'Semantic Direction Projections (Step {global_step})')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "semantic_directions.png"))
                plt.close()
            
            # Create comprehensive report file
            report_title = "Initial Embedding Analysis Report" if global_step == 0 else "Embedding Analysis Report"
            with open(os.path.join(output_dir, "embedding_analysis_report.md"), "w") as f:
                f.write(f"# {report_title}\n\n")
                if global_step > 0:
                    f.write(f"## Step {global_step}\n\n")
                
                # Add statistics section
                if stats:
                    f.write("## Statistical Properties\n\n")
                    f.write(f"- **Norm**: {stats.get('norm', 'N/A'):.6f}\n")
                    f.write(f"- **Mean**: {stats.get('mean', 'N/A'):.6f}\n")
                    f.write(f"- **Standard Deviation**: {stats.get('std', 'N/A'):.6f}\n")
                    f.write(f"- **Min Value**: {stats.get('min', 'N/A'):.6f}\n")
                    f.write(f"- **Max Value**: {stats.get('max', 'N/A'):.6f}\n\n")
                
                # Add similar words section
                f.write("## Most Similar Tokens\n\n")
                f.write("| Token | Similarity |\n")
                f.write("|-------|------------|\n")
                for token, sim in similar_tokens[:15]:
                    f.write(f"| `{token}` | {sim:.6f} |\n")
                f.write("\n")
                
                # Add semantic clustering section
                f.write("## Semantic Cluster Analysis\n\n")
                f.write("| Concept Group | Similarity |\n")
                f.write("|--------------|------------|\n")
                for group, value in semantic_clusters.items():
                    f.write(f"| {group} | {value:.6f} |\n")
                f.write("\n")
                
                # Add semantic direction section
                f.write("## Semantic Direction Projections\n\n")
                f.write("| Direction | Projection |\n")
                f.write("|-----------|------------|\n")
                for direction, projection in sorted_directions[:15]:
                    f.write(f"| {direction} | {projection:.6f} |\n")
                f.write("\n")
                
                # Add image references
                f.write("## Visualizations\n\n")
                f.write("### Embedding Distribution\n\n")
                f.write("![Embedding Distribution](embedding_statistics/embedding_distribution.png)\n\n")
                
                f.write("### Concept Similarity\n\n")
                f.write("![Similar Tokens](similar_tokens.png)\n\n")
                
                f.write("### Semantic Space Visualization\n\n")
                f.write("![Embedding PCA](embedding_pca.png)\n\n")
                
                f.write("### Semantic Shifts\n\n")
                f.write("![Semantic Shifts](semantic_shifts/current_semantic_shift.png)\n\n")
                
                f.write("### Initializer Comparison\n\n")
                f.write("![Initializer Comparison](initializer_comparison/initializer_comparison.png)\n\n")
            
            logger.info(f"Embedding analysis for step {global_step} completed")
            
        except Exception as e:
            logger.error(f"Error during embedding analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    return results