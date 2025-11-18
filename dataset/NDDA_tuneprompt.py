import os
import torch
import argparse
import numpy as np
import random
from tqdm import tqdm
from PIL import Image

# Functions for loading and updating processed folders
def load_processed_folders(processed_file_path):
    """Load the list of processed folders"""
    try:
        with open(processed_file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(f"Warning: Processed folder record {processed_file_path} does not exist, creating new file.")
        # Create the file if it doesn't exist
        os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
        with open(processed_file_path, 'w', encoding='utf-8') as f:
            pass
        return []

def add_to_processed_folders(processed_file_path, folder_name):
    """Add newly processed folder to the record"""
    with open(processed_file_path, 'a', encoding='utf-8') as f:
        f.write(f"{folder_name}\n")

def is_black_image(image_path, threshold=10):
    """
    Check if an image is completely black (or nearly black)
    
    Parameters:
    - image_path: Path to the image
    - threshold: Average pixel value threshold, below which is considered a black image
    
    Returns:
    - Returns True if the image is black, otherwise False
    """
    try:
        # Open the image
        img = Image.open(image_path)
        # Convert to grayscale
        img_gray = img.convert('L')
        # Convert to numpy array
        img_array = np.array(img_gray)
        # Calculate average pixel value
        mean_value = np.mean(img_array)
        
        # Check if it's a black image
        return mean_value < threshold
    except Exception as e:
        print(f"Error checking image: {e}")
        # If unable to open or process the image, default to needing regeneration
        return True

def check_folder_complete(folder_path, num_images, threshold=10):
    """
    Check if all images in the folder are generated and not black
    
    Returns:
    - List of all image indices that should be regenerated
    """
    images_to_regenerate = []
    missing_images = []
    black_images = []
    
    for i in range(num_images):
        image_path = os.path.join(folder_path, f"image_{i+1:03d}.png")
        
        # Check if the image exists
        if not os.path.exists(image_path):
            images_to_regenerate.append(i)
            missing_images.append(i)
            print(f"Found missing image: {image_path}")
            continue
        
        # Check if the image is black
        if is_black_image(image_path, threshold):
            images_to_regenerate.append(i)
            black_images.append(i)
            print(f"Found black image: {image_path}")
    
    # Print statistics
    if missing_images:
        print(f"Total missing {len(missing_images)} images, indices: {missing_images}")
    if black_images:
        print(f"Total found {len(black_images)} black images, indices: {black_images}")
    
    return images_to_regenerate

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Stable Diffusion images')
    parser.add_argument('--model-id', type=str, default="stable-diffusion-v1-5/stable-diffusion-v1-5", 
                        help='Stable Diffusion model ID')
    parser.add_argument('--output-dir', type=str, 
                        default="/zhaotingfeng/jiwenjun/anglerocl/data/stable_diffusion_1.5/stop_sign_prompts_tune/", 
                        help='Output directory')
    parser.add_argument('--num-images', type=int, default=50, 
                        help='Number of images to generate per prompt')
    parser.add_argument('--guidance-scale', type=float, default=7.5, 
                        help='Controls how closely the image matches the prompt')
    parser.add_argument('--num-steps', type=int, default=50, 
                        help='Number of inference steps, higher values typically produce better results')
    parser.add_argument('--black-threshold', type=int, default=10, 
                        help='Average pixel value threshold to determine if an image is black')
    parser.add_argument('--check-only', action='store_true', 
                        help='Only check folders, do not regenerate images')
    parser.add_argument('--regenerate-all', action='store_true', 
                        help='Regenerate all images marked as black')
    parser.add_argument('--group', type=str, choices=['prefix', 'middle', 'suffix', 'all'], default='all',
                        help='Dataset group to generate: prefix, middle, suffix, or all')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set device - use GPU if available, otherwise CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Stable Diffusion model
    print(f"Loading model: {args.model_id}")
    from diffusers import StableDiffusionPipeline
    
    # Only load the model when image generation is needed
    if not args.check_only:
        model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

        # repo_id_embeds = "/zhaotingfeng/jiwenjun/anglerocl/ckpt/20250419_115037/learned_embeds-steps-19500.safetensors"
        # pipe.load_textual_inversion(repo_id_embeds)
    
    # Define base folder names list
    base_folder_names = [
        "blue_square_stop_sign",
        "blue_square_stop_sign_with__abcd__on_it",
        "blue_square_stop_sign_with__abcd__on_it_and_checkerboard_paint_on_it",
        "blue_square_stop_sign_with__hello__on_it",
        "blue_square_stop_sign_with__hello__on_it_and_checkerboard_paint_on_it",
        "blue_square_stop_sign_with_checkerboard_paint_on_it",
        "blue_stop_sign",
        "blue_stop_sign_with__abcd__on_it",
        "blue_stop_sign_with__abcd__on_it_and_checkerboard_paint_on_it",
        "blue_stop_sign_with__hello__on_it",
        "blue_stop_sign_with__hello__on_it_and_checkerboard_paint_on_it",
        "blue_stop_sign_with_checkerboard_paint_on_it",
        "square_stop_sign",
        "square_stop_sign_with__abcd__on_it",
        "square_stop_sign_with__abcd__on_it_and_checkerboard_paint_on_it",
        "square_stop_sign_with__hello__on_it",
        "square_stop_sign_with__hello__on_it_and_checkerboard_paint_on_it",
        "square_stop_sign_with_checkerboard_paint_on_it",
        "stop_sign",
        "stop_sign_with__abcd__on_it",
        "stop_sign_with__abcd__on_it_and_checkerboard_paint_on_it",
        "stop_sign_with__hello__on_it",
        "stop_sign_with__hello__on_it_and_checkerboard_paint_on_it",
        "stop_sign_with__world__on_it",
        "stop_sign_with__world__on_it_and_polkadot_paint_on_it",
        "stop_sign_with_checkerboard_paint_on_it",
        "stop_sign_with_polkadot_paint_on_it",
        "triangle_stop_sign",
        "triangle_stop_sign_with__world__on_it",
        "triangle_stop_sign_with__world__on_it_and_polkadot_paint_on_it",
        "triangle_stop_sign_with_polkadot_paint_on_it",
        "yellow_stop_sign",
        "yellow_stop_sign_with__world__on_it",
        "yellow_stop_sign_with__world__on_it_and_polkadot_paint_on_it",
        "yellow_stop_sign_with_polkadot_paint_on_it",
        "yellow_triangle_stop_sign",
        "yellow_triangle_stop_sign_with__world__on_it",
        "yellow_triangle_stop_sign_with__world__on_it_and_polkadot_paint_on_it",
        "yellow_triangle_stop_sign_with_polkadot_paint_on_it"
    ]
    
    # Prefix modifiers - maintain horizontal angle concept
    prefix_modifiers = [
        "To enable the detection from multiple horizontal angles",
        "To enable the recognition from numerous horizontal angles", 
        "To enable the identification from several horizontal angles"
    ]

    # Middle modifiers - maintain horizontal angle concept
    middle_modifiers = [
        "detectable at multiple angles in all horizontal directions",
        "recognizable at numerous angles in all horizontal orientations",
        "identifiable at several angles in all horizontal perspectives"
    ]

    # Suffix modifiers - maintain horizontal angle concept
    suffix_modifiers = [
        "which can be detected by the detector from different horizontal angles of observation",
        "which can be recognized by the sensor from various horizontal angles of viewing",
        "which can be identified by the system from diverse horizontal angles of scanning"
    ]
    
    # Groups to process
    groups_to_process = []
    if args.group == 'all':
        groups_to_process = ['prefix', 'middle', 'suffix']
    else:
        groups_to_process = [args.group]
    
    # Process each group
    for group in groups_to_process:
        print(f"\nStarting to process {group} group dataset")
        
        # Determine output directory
        group_output_dir = os.path.join(args.output_dir, group)
        if not os.path.exists(group_output_dir):
            os.makedirs(group_output_dir, exist_ok=True)
        
        # Set path for the processing record file
        processed_file = os.path.join(group_output_dir, "processed_folders.txt")
        
        # Load processed folders list
        processed_folders = load_processed_folders(processed_file)
        print(f"Found {len(processed_folders)} previously processed folders")
        
        # Create files for recording black and missing images
        black_images_file = os.path.join(group_output_dir, "black_images.txt")
        missing_images_file = os.path.join(group_output_dir, "missing_images.txt")
        
        # Initialize records
        all_black_images = []
        all_missing_images = []
        total_regenerated = 0
        
        # Generate folder names and prompts with modifiers
        folder_names = []
        prompts = []
        
        # Select modifier position based on group type
        if group == 'prefix':
            modifiers = prefix_modifiers
            # Generate prompts with prefix modifiers for each base folder name
            for base_folder in base_folder_names:
                # Randomly select a modifier
                modifier = random.choice(modifiers)
                # Use base folder name without adding prefix
                folder_names.append(base_folder)
                
                # Generate prompt
                base_prompt = base_folder.replace("_", " ")
                base_prompt = base_prompt.replace("with  abcd  on", "with \"abcd\" on")
                base_prompt = base_prompt.replace("with  world  on", "with \"world\" on")
                base_prompt = base_prompt.replace("with  hello  on", "with \"hello\" on")
                
                # Make the prompt more natural
                prompt = f"{modifier}, I need a {base_prompt}"
                prompts.append(prompt)
                
        elif group == 'middle':
            modifiers = middle_modifiers
            # Generate prompts with middle modifiers for each base folder name
            for base_folder in base_folder_names:
                # Randomly select a modifier
                modifier = random.choice(modifiers)
                # Use base folder name without adding prefix
                folder_names.append(base_folder)
                
                # Generate prompt
                base_prompt = base_folder.replace("_", " ")
                base_prompt = base_prompt.replace("with  abcd  on", "with \"abcd\" on")
                base_prompt = base_prompt.replace("with  world  on", "with \"world\" on")
                base_prompt = base_prompt.replace("with  hello  on", "with \"hello\" on")
                
                # Insert modifier after "stop sign", using "that is" for smoother flow
                if "stop sign" in base_prompt:
                    parts = base_prompt.split("stop sign", 1)
                    prompt = f"{parts[0]}stop sign that is {modifier}{parts[1] if len(parts) > 1 else ''}"
                else:
                    # If no "stop sign", use natural phrasing
                    prompt = f"{base_prompt} that is {modifier}"
                
                prompts.append(prompt)
                
        elif group == 'suffix':
            modifiers = suffix_modifiers
            # Generate prompts with suffix modifiers for each base folder name
            for base_folder in base_folder_names:
                # Randomly select a modifier
                modifier = random.choice(modifiers)
                # Use base folder name without adding prefix
                folder_names.append(base_folder)
                
                # Generate prompt
                base_prompt = base_folder.replace("_", " ")
                base_prompt = base_prompt.replace("with  abcd  on", "with \"abcd\" on")
                base_prompt = base_prompt.replace("with  world  on", "with \"world\" on")
                base_prompt = base_prompt.replace("with  hello  on", "with \"hello\" on")
                
                # Maintain natural connection with suffix
                prompt = f"{base_prompt} {modifier}"
                prompts.append(prompt)
        
        # Process all folders
        for i, folder_name in enumerate(folder_names):
            prompt = prompts[i]
            prompt_dir = os.path.join(group_output_dir, folder_name)
            
            # Ensure folder exists
            if not os.path.exists(prompt_dir):
                os.makedirs(prompt_dir)
                if not args.check_only:
                    print(f"Created new folder: {prompt_dir}")
            
            # Check images that need to be regenerated in the folder
            images_to_regenerate = check_folder_complete(prompt_dir, args.num_images, args.black_threshold)
            
            # If no images need regeneration and folder is already in processed list, skip
            if not images_to_regenerate and folder_name in processed_folders and not args.regenerate_all:
                print(f"All images in folder {folder_name} are complete and normal, skipping")
                continue
            
            if images_to_regenerate:
                print(f"Found {len(images_to_regenerate)} images that need regeneration in folder {folder_name}")
                
                # Record black and missing images
                for j in images_to_regenerate:
                    image_path = os.path.join(prompt_dir, f"image_{j+1:03d}.png")
                    if not os.path.exists(image_path):
                        all_missing_images.append(f"{folder_name}/image_{j+1:03d}.png")
                    else:
                        all_black_images.append(f"{folder_name}/image_{j+1:03d}.png")
            
            # If in check-only mode, don't generate images
            if args.check_only:
                continue
            
            # Generate missing or completely black images
            if images_to_regenerate:
                print(f"Processing prompt [{i+1}/{len(folder_names)}]: {prompt}")
                
                # Create CSV file (if it doesn't exist)
                csv_file_path = os.path.join(prompt_dir, f"{folder_name}.csv")
                if not os.path.exists(csv_file_path):
                    with open(csv_file_path, 'w') as f:
                        pass
                
                # Only regenerate needed images
                for img_idx in tqdm(images_to_regenerate, desc=f"Regenerating images for {prompt}"):
                    try:
                        image_path = os.path.join(prompt_dir, f"image_{img_idx+1:03d}.png")
                        
                        # Generate image without fixed random seed
                        image = pipe(
                            prompt=prompt,
                            guidance_scale=args.guidance_scale,
                            num_inference_steps=args.num_steps
                        ).images[0]
                        
                        # Check if the generated image is black
                        image_array = np.array(image.convert('L'))
                        if np.mean(image_array) < args.black_threshold:
                            print(f"Warning: Newly generated image is still black, will attempt to regenerate: {image_path}")
                            # Retry count can be added here, but for brevity we just log and continue
                            continue
                        
                        # Save image
                        image.save(image_path)
                        total_regenerated += 1
                        
                    except Exception as e:
                        print(f"Error generating image: {e}")
                
                # If not in processed list, add to processed folders
                if folder_name not in processed_folders:
                    add_to_processed_folders(processed_file, folder_name)
                    print(f"Added {folder_name} to processed folders list")
        
        # Write black images to file
        if all_black_images and not args.check_only:
            with open(black_images_file, 'w', encoding='utf-8') as f:
                for img_path in all_black_images:
                    f.write(f"{img_path}\n")
            print(f"Recorded {len(all_black_images)} black images to {black_images_file}")
        
        # Write missing images to file
        if all_missing_images and not args.check_only:
            with open(missing_images_file, 'w', encoding='utf-8') as f:
                for img_path in all_missing_images:
                    f.write(f"{img_path}\n")
            print(f"Recorded {len(all_missing_images)} missing images to {missing_images_file}")
        
        if args.check_only:
            print(f"{group} group check complete! Found {len(all_black_images)} black images and {len(all_missing_images)} missing images that need regeneration.")
        else:
            print(f"{group} group image generation complete! Successfully regenerated {total_regenerated} images.")

if __name__ == "__main__":
    main()