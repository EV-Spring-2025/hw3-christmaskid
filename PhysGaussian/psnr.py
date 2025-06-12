import os
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
from argparse import ArgumentParser
from tqdm import tqdm

# Import from the gaussian splatting codebase
import sys
sys.path.append("gaussian-splatting")
from utils.image_utils import psnr

def calculate_average_psnr(predicted_dir, ground_truth_dir):
    """Calculate average PSNR between predicted and ground truth images"""
    predicted_images = sorted([f for f in os.listdir(predicted_dir) if f.endswith('.png')])
    gt_images = sorted([f for f in os.listdir(ground_truth_dir) if f.endswith('.png')])
    
    if len(predicted_images) != len(gt_images):
        print(f"Warning: Number of predicted images ({len(predicted_images)}) != ground truth images ({len(gt_images)})")
    
    psnr_values = []
    
    print(f"Calculating PSNR for {len(predicted_images)} image pairs...")
    for pred_img_name in tqdm(predicted_images):
        gt_img_name = pred_img_name  # Assuming same naming convention
        
        if gt_img_name not in gt_images:
            print(f"Warning: Ground truth image {gt_img_name} not found, skipping...")
            continue
        
        # Load images
        pred_path = os.path.join(predicted_dir, pred_img_name)
        gt_path = os.path.join(ground_truth_dir, gt_img_name)
        
        pred_img = Image.open(pred_path)
        gt_img = Image.open(gt_path)
        
        # Convert to tensors
        pred_tensor = tf.to_tensor(pred_img).unsqueeze(0)[:, :3, :, :].cuda()
        gt_tensor = tf.to_tensor(gt_img).unsqueeze(0)[:, :3, :, :].cuda()
        
        # Calculate PSNR
        psnr_value = psnr(pred_tensor, gt_tensor)
        psnr_values.append(psnr_value.item())
    
    if not psnr_values:
        print("No valid image pairs found!")
        return 0.0
    
    average_psnr = np.mean(psnr_values)
    return average_psnr, psnr_values

def main():
    parser = ArgumentParser(description="Calculate average PSNR between two folders of images")
    parser.add_argument("--predicted_dir", "-p", required=True, type=str, 
                       help="Directory containing predicted *.png images")
    parser.add_argument("--gt_output_dir", "-g", default="./ground_truth_renders", type=str,
                       help="Directory to save rendered ground truth images")
    
    args = parser.parse_args()
    
    # Calculate average PSNR
    avg_psnr, psnr_values = calculate_average_psnr(args.predicted_dir, args.gt_output_dir)
    
    print(f"\nResults:")
    print(f"Number of image pairs: {len(psnr_values)}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Min PSNR: {min(psnr_values):.4f}")
    print(f"Max PSNR: {max(psnr_values):.4f}")
    print(f"Std PSNR: {np.std(psnr_values):.4f}")

if __name__ == "__main__":
    main()