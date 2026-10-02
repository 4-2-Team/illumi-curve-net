"""
This module provides functionality for loading and processing low-light images for training.
It includes utilities for dataset creation and image preprocessing.
"""

import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import random

# Set random seed for reproducibility
random.seed(1143)


def populate_train_list(lowlight_images_path):
    """
    Creates a list of file paths for all images in the given directory and its subdirectories.
    
    Args:
        lowlight_images_path (str): Path to the directory containing low-light images
        
    Returns:
        list: Shuffled list of file paths
    """
    file_paths_and_names = []

    # Walk through directory tree and collect all file paths
    for dirpath, dirnames, filenames in os.walk(lowlight_images_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_paths_and_names.append(file_path)

    # Shuffle the file paths for randomization
    random.shuffle(file_paths_and_names)

    return file_paths_and_names


def synthetic_low_light(img, gamma_range=(1.8, 3.2), exposure_range=(0.25, 0.6), noise_range=(0.005, 0.02)):
    """Darken a well-exposed image into a plausible synthetic low-light frame.

    IllumiCurveNet's unsupervised losses assume the input is genuinely
    underexposed; training it directly on already well-lit renders teaches it
    to enhance images that don't need it. This applies a random gamma raise
    (darkens midtones), a random global exposure cut, and sensor-like noise,
    so the network sees the same kind of dim, noisy input it will face at
    deployment. Runs on a single CHW tensor in [0, 1].
    """
    gamma = random.uniform(*gamma_range)
    exposure = random.uniform(*exposure_range)
    noise_sigma = random.uniform(*noise_range)

    img = img.clamp(min=1e-4).pow(gamma) * exposure
    img = img + torch.randn_like(img) * noise_sigma
    return img.clamp(0.0, 1.0)


class lowlight_loader(data.Dataset):
    """
    Dataset class for loading and preprocessing low-light images.

    Args:
        lowlight_images_path (str): Path to the directory containing low-light images
        max_samples (int, optional): Cap the dataset to this many images
        synthetic_degrade (bool): If True, apply synthetic_low_light() to each
            loaded image so training sees a dim/noisy frame instead of the
            source image as-is
    """
    def __init__(self, lowlight_images_path, max_samples=None, synthetic_degrade=False):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = 128  # Target size for image resizing
        self.synthetic_degrade = synthetic_degrade

        print("Total training examples:", len(self.train_list))

        # Optionally cap dataset size (e.g. for quick smoke tests). Off by default.
        if max_samples is not None and len(self.train_list) > max_samples:
            self.train_list = random.sample(self.train_list, max_samples)
        self.data_list = self.train_list

    def __getitem__(self, index):
        """
        Loads and preprocesses a single image from the dataset.

        Args:
            index (int): Index of the image to load

        Returns:
            torch.Tensor: Preprocessed image tensor in CHW format
        """
        data_lowlight_path = self.data_list[index]

        # Load and preprocess the image
        data_lowlight = Image.open(data_lowlight_path).convert('RGB')
        data_lowlight = data_lowlight.resize((self.size,self.size), Image.Resampling.LANCZOS)
        data_lowlight = (np.asarray(data_lowlight)/255.0)  # Normalize to [0,1]
        data_lowlight = torch.from_numpy(data_lowlight).float()
        data_lowlight = data_lowlight.permute(2,0,1)  # Convert to CHW format

        if self.synthetic_degrade:
            data_lowlight = synthetic_low_light(data_lowlight)

        return data_lowlight

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        
        Returns:
            int: Number of images
        """
        return len(self.data_list)





