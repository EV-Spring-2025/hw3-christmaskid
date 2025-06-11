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
from utils.image_utils import psnr
from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

def load_camera_views_from_gs_simulation(model_path):
    """Load camera views used in gs_simulation"""
    # This would need to be adapted based on how camera views are stored
    # For now, we'll use the standard cameras from the scene
    scene = Scene(ModelParams(), model_path, shuffle=False)
    return scene.getTrainCameras() + scene.getTestCameras()

def render_ground_truth_images(model_path, output_dir):
    """Render ground truth images from PLY file using camera views"""
    print("Loading scene and cameras...")
    
    # Initialize Gaussian model from PLY
    gaussians = GaussianModel(3)  # sh_degree=3
    scene = Scene(ModelParams(), model_path, gaussians, shuffle=False)
    
    # Load pipeline parameters
    pipeline = PipelineParams(ArgumentParser(description="Testing script parameters"))
    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    
    # Get camera views
    cameras = scene.getTrainCameras() + scene.getTestCameras()
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Rendering {len(cameras)} ground truth images...")
    for idx, camera in enumerate(tqdm(cameras)):
        # Render the image
        rendering = render(camera, gaussians, pipeline, background)["render"]
        
        # Convert to numpy and save
        image = torch.clamp(rendering, 0.0, 1.0)
        image_np = image.permute(1, 2, 0).detach().cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        
        # Save with same naming convention as predicted images
        output_path = os.path.join(output_dir, f"{idx:08d}.png")
        cv2.imwrite(output_path, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    
    return len(cameras)

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
    parser = ArgumentParser(description="Calculate average PSNR between predicted and ground truth images")
    parser.add_argument("--predicted_dir", "-p", required=True, type=str, 
                       help="Directory containing predicted *.png images")
    parser.add_argument("--model_path", "-m", required=True, type=str,
                       help="Path to model directory containing input.ply")
    parser.add_argument("--gt_output_dir", "-g", default="./ground_truth_renders", type=str,
                       help="Directory to save rendered ground truth images")
    parser.add_argument("--skip_rendering", action="store_true",
                       help="Skip rendering ground truth images (use existing ones)")
    
    args = parser.parse_args()
    
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available! This script requires GPU.")
        return
    
    # Initialize system state
    safe_state(False)
    
    # Check if predicted directory exists
    if not os.path.exists(args.predicted_dir):
        print(f"Predicted directory {args.predicted_dir} does not exist!")
        return
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Model path {args.model_path} does not exist!")
        return
    
    # Render ground truth images if needed
    if not args.skip_rendering:
        print("Rendering ground truth images...")
        num_rendered = render_ground_truth_images(args.model_path, args.gt_output_dir)
        print(f"Rendered {num_rendered} ground truth images to {args.gt_output_dir}")
    else:
        print(f"Using existing ground truth images from {args.gt_output_dir}")
    
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