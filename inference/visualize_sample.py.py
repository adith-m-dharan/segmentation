# demo_seg/visualize_sample.py

import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from utils.visualizer import overlay_mask_on_image

def visualize_sample(split="train", num_samples=3):
    base_dir = os.path.join("data", "prepared", split)
    image_dir = os.path.join(base_dir, "images")
    mask_dir = os.path.join(base_dir, "masks")

    # Match image-mask pairs by basename
    images = sorted([f for f in os.listdir(image_dir) if f.endswith((".jpg", ".png"))])
    masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    basenames = list(set(os.path.splitext(f)[0] for f in images) & set(os.path.splitext(f)[0] for f in masks))

    if not basenames:
        raise ValueError("No matching image-mask pairs found.")

    selected = random.sample(basenames, min(num_samples, len(basenames)))

    for name in selected:
        img_path = os.path.join(image_dir, name + ".jpg")  # fallback if .png
        if not os.path.exists(img_path):
            img_path = os.path.join(image_dir, name + ".png")
        mask_path = os.path.join(mask_dir, name + ".png")

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        overlay = overlay_mask_on_image(img, mask)

        # Plot
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap="jet")
        plt.title("Prepared Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(overlay)
        plt.title("Overlay")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    visualize_sample(split="train", num_samples=3)
