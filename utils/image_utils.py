# utils/image_utils.py
import numpy as np
import torch

def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize the image using ImageNet mean and std.
    Args:
        img: Input image as a NumPy array (H x W x 3) in uint8 format.
    Returns:
        Normalized image as a float32 NumPy array.
    """
    # Convert to float32 and scale to [0,1]
    img = img.astype(np.float32) / 255.0
    # ImageNet mean and std (RGB)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    # Normalize: (img - mean) / std (broadcasting over channels)
    img = (img - mean) / std
    return img

def to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Convert an image from HWC format to a PyTorch tensor in CHW format.
    Args:
        img: Input image as a NumPy array.
    Returns:
        Torch tensor in float32.
    """
    img = np.transpose(img, (2, 0, 1))  # convert to CHW
    return torch.from_numpy(img).float()
