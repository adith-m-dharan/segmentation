
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch

# Fixed RGBCMYBW color map for classes 0–7
COLOR_MAP = {
    0: [0, 0, 0],         # Black
    1: [255, 0, 0],       # Red
    2: [0, 255, 0],       # Green
    3: [0, 0, 255],       # Blue
    4: [0, 255, 255],     # Cyan
    5: [255, 0, 255],     # Magenta
    6: [255, 255, 0],     # Yellow
    7: [255, 255, 255],   # White
}

def label_to_rgb(mask: np.ndarray) -> np.ndarray:
    
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        rgb[mask == cls] = color
    return rgb

def overlay_mask(image: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    
    if seg_mask.shape != image.shape[:2]:
        seg_mask = cv2.resize(seg_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    overlay = label_to_rgb(seg_mask)
    blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
    return blended

def show_image(image: np.ndarray, title: str = "Image", save_path: str = None):
    
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def show_comparison(
    original: np.ndarray,
    gt_mask: np.ndarray,
    seg_mask: np.ndarray,
    save_path: str = None,
    class_info: str = None
):
    # Create visualization images.
    overlay = overlay_mask(original, seg_mask)
    gt_rgb = label_to_rgb(gt_mask)
    pred_rgb = label_to_rgb(seg_mask)

    # Create a 2x2 grid for the images.
    fig, axes = plt.subplots(2, 2, figsize=(10, 12))
    fig.patch.set_facecolor((0.5, 0.5, 0.5))

    axes[0, 0].imshow(original)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(gt_rgb)
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title("Overlay (Image + Prediction)")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(pred_rgb)
    axes[1, 1].set_title("Predicted Mask")
    axes[1, 1].axis("off")
    
    # If a valid class info file is provided, load the category names and add a legend.
    if class_info and os.path.exists(class_info):
        class_data = torch.load(class_info)
        legend_elements = []
        # Build legend entries for each class in the fixed color map.
        for class_id in sorted(COLOR_MAP.keys()):
            cat_name = class_data.get(class_id, {}).get("name", f"Class {class_id}")
            patch = Patch(facecolor=np.array(COLOR_MAP[class_id]) / 255.0, label=cat_name)
            legend_elements.append(patch)
        
        # Adjust the layout to leave room for the legend on the right side.
        plt.subplots_adjust(right=0.8)
        fig.legend(
            handles=legend_elements,
            loc='upper center',
            ncol=len(legend_elements),
            title='Categories'
        ).get_frame().set_facecolor((0.5, 0.5, 0.5))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    return fig

if __name__ == "__main__":
    # Example usage:
    # Create dummy data for demonstration.
    image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    gt_mask = np.random.randint(0, 8, (512, 512), dtype=np.uint8)
    seg_mask = np.random.randint(0, 8, (512, 512), dtype=np.uint8)
    
    # Adjust this path to point to your saved class info file.
    legend_file = "run/data/prepared/train/class_weights.pt"
    
    # Create the comparison figure with the legend inserted.
    fig = show_comparison(
        original=image,
        gt_mask=gt_mask,
        seg_mask=seg_mask,
        class_info=legend_file,
        save_path="run/logs/test.png"
    )
    # plt.show()
