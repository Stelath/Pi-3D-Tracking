"""
Transform functions for preparing images for Pi3 model input.
"""

import math
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class Pi3Transform:
    """
    Transform images for Pi3 model input.

    Resizes images to satisfy pixel limit while maintaining aspect ratio,
    ensures dimensions are multiples of 14 (patch size), and converts to tensor.

    Args:
        pixel_limit: Maximum number of pixels (width * height). Default: 255000

    Input:
        - Numpy array [H, W, 3] in RGB format, uint8, range [0, 255]

    Output:
        - PyTorch tensor [C, H, W], float32, range [0, 1]
    """

    def __init__(self, pixel_limit: int = 255000):
        self.pixel_limit = pixel_limit
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        """
        Apply transform to a single image.

        Args:
            img: Numpy array [H, W, 3] in RGB format

        Returns:
            PyTorch tensor [C, H, W] normalized to [0, 1]
        """
        # Convert numpy array to PIL Image
        if isinstance(img, np.ndarray):
            pil_img = Image.fromarray(img.astype(np.uint8))
        else:
            pil_img = img

        # Get original dimensions
        W_orig, H_orig = pil_img.size

        # Calculate target size maintaining aspect ratio with pixel limit
        scale = math.sqrt(self.pixel_limit / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
        W_target = W_orig * scale
        H_target = H_orig * scale

        # Round to multiples of 14 (patch size for Pi3)
        k = round(W_target / 14)
        m = round(H_target / 14)

        # Ensure we don't exceed pixel limit
        while (k * 14) * (m * 14) > self.pixel_limit:
            if k / m > W_target / H_target:
                k -= 1
            else:
                m -= 1

        # Final target dimensions (multiples of 14)
        TARGET_W = max(1, k) * 14
        TARGET_H = max(1, m) * 14

        # Resize using LANCZOS interpolation (high quality)
        resized_img = pil_img.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)

        # Convert to tensor [C, H, W] normalized to [0, 1]
        img_tensor = self.to_tensor(resized_img)

        return img_tensor