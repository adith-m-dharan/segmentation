"""
utils/visualizer.py

This module contains helper functions for visualizing images,
segmentation masks, and overlays. You can use these functions to
display individual images, segmentation masks, or side-by-side comparisons.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_mask_on_image(image: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay the segmentation mask on the original image.
    
    Args:
        image (np.ndarray): Original image in RGB with shape (H, W, 3).
        seg_mask (np.ndarray): Segmentation mask (H_mask x W_mask) with integer class labels.
        alpha (float): Blending factor between 0 and 1.
    
    Returns:
        np.ndarray: Image with the segmentation overlay applied.
    """
    # Ensure seg_mask size matches image dimensions
    if seg_mask.shape != image.shape[:2]:
        seg_mask = cv2.resize(seg_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    unique_classes = np.unique(seg_mask)
    # Generate random color for each unique class.
    colors = {cls: np.random.randint(0, 255, size=3) for cls in unique_classes}
    
    # Create a color overlay image, same shape as the original image.
    color_overlay = np.zeros_like(image, dtype=np.uint8)
    for cls in unique_classes:
        color_overlay[seg_mask == cls] = colors[cls]
    
    overlay = cv2.addWeighted(image, 1 - alpha, color_overlay, alpha, 0)
    return overlay

def show_image(image: np.ndarray, title: str = "Image"):
    """
    Display an image using matplotlib.
    
    Args:
        image (np.ndarray): Image in RGB format.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_comparison(original: np.ndarray, mask: np.ndarray, overlay: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="jet")
    axes[1].set_title("Segmentation Mask (GT)")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Prediction)")
    axes[2].axis("off")

    plt.tight_layout()
    return fig
