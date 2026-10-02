"""
Testing script for IllumiCurveNet.

This script implements a testing pipeline for our Low Light Image Enhancement model allowing configurations through command line arguments.
"""

import os
import argparse
import glob
import numpy as np
import torch
import torchvision
import torch.optim
import model
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def collect_images(input_path):
    """Collect image paths from a flat folder or nested subfolders."""
    images = []
    for name in sorted(os.listdir(input_path)):
        full_path = os.path.join(input_path, name)
        if os.path.isdir(full_path):
            images.extend(sorted(glob.glob(os.path.join(full_path, '*'))))
        elif os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
            images.append(full_path)
    return [path for path in images if os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS]


def run_output_dir(base_output_dir, input_path):
    """Build a run-specific output folder from the last three path parts."""
    parts = [part for part in os.path.normpath(os.path.abspath(input_path)).split(os.sep) if part]
    run_name = os.path.join(*parts[-3:]) if len(parts) >= 3 else os.path.join(*parts)
    return os.path.join(os.path.abspath(base_output_dir), run_name)


def test(image_path, config, ic_net):
    """
    Process and enhance a single low-light image using the IllumiCurveNet model.
    
    Args:
        image_path (str): Path to the input low-light image
        
    Returns:
        None: Saves the enhanced image to the result directory
    """
    # Load and preprocess the image
    data_lowlight = Image.open(image_path).convert('RGB')

    # Resize to dimensions divisible by 8 (to avoid dimension mismatch errors in the model's concatenations)
    width = (data_lowlight.size[0] // 8) * 8
    height = (data_lowlight.size[1] // 8) * 8
    data_lowlight = data_lowlight.resize((width, height))

    data_lowlight = (np.asarray(data_lowlight)/255.0)  # Normalize to [0,1]
    data_lowlight = torch.from_numpy(data_lowlight).float()
    data_lowlight = data_lowlight.permute(2,0,1)  # Change to channel-first format
    data_lowlight = data_lowlight.cuda().unsqueeze(0)  # Add batch dimension and move to GPU

    # Generate enhanced image
    enhanced_image,_ = ic_net(data_lowlight)

    # Create output path and save enhanced image
    input_root = os.path.abspath(config.lowlight_images_path)
    rel_path = os.path.relpath(os.path.abspath(image_path), input_root)
    result_path = os.path.join(os.path.abspath(config.output_dir), rel_path)
    result_dir = os.path.dirname(result_path)

    os.makedirs(result_dir, exist_ok=True)
    torchvision.utils.save_image(enhanced_image, result_path)
    return result_path


if __name__ == '__main__':
    # Setup command line argument parser
    parser = argparse.ArgumentParser(description='IllumiCurveNet Testing Script')
    
    # Define command line arguments
    parser.add_argument('--lowlight_images_path', type=str, default="data/test_data/", help='Path to low-light testing images')
    parser.add_argument('--pretrain_snapshot', type=str, default= "snapshots/model-best.pth", help='Pretrained model snapshot')
    parser.add_argument('--output_dir', type=str, default="result/", help='Base directory for run-specific enhanced image folders')

    config = parser.parse_args()

    image_list = collect_images(config.lowlight_images_path)
    if not image_list:
        print(f"No images found in {config.lowlight_images_path}")
        raise SystemExit(1)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    ic_net = model.illumi_curve_net().cuda()
    ic_net.load_state_dict(torch.load(config.pretrain_snapshot))
    ic_net.eval()

    output_dir = run_output_dir(config.output_dir, config.lowlight_images_path)
    config.output_dir = output_dir
    print(f"Processing {len(image_list)} image(s) from {config.lowlight_images_path}")
    print(f"Saving results to {output_dir}")

    with torch.no_grad():
        for image in image_list:
            result_path = test(image, config, ic_net)
            print(f"  saved {result_path}")

    print(f"Done. Enhanced {len(image_list)} image(s) in {output_dir}")
    
