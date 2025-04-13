import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from models.vision_system import VisionSystem
from utils.visualizer import overlay_mask_on_image, show_comparison
from torchvision.transforms.functional import to_pil_image


def infer(image_path, checkpoint_path, save_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = VisionSystem(num_classes=num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    image_np = np.array(image_pil.resize((224, 224)))
    overlay = overlay_mask_on_image(image_np, pred)

    os.makedirs(save_path, exist_ok=True)
    Image.fromarray(image_np).save(os.path.join(save_path, "original_image.png"))
    Image.fromarray(pred.astype(np.uint8)).save(os.path.join(save_path, "segmented_output.png"))
    Image.fromarray(overlay).save(os.path.join(save_path, "overlay_output.png"))

    # Save only side-by-side comparison
    fig = show_comparison(image_np, np.zeros_like(pred), overlay)  # no ground truth in inference
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(os.path.join(save_path, "comparison.png"))
    plt.close(fig)

    print(f"Inference complete. Output saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/model_epoch_20.pth", help="Path to model checkpoint")
    parser.add_argument("--save_dir", default="results/inference", help="Directory to save output")
    parser.add_argument("--num_classes", type=int, default=6, help="Total number of classes (including background)")
    args = parser.parse_args()

    infer(args.image, args.checkpoint, args.save_dir, args.num_classes)
