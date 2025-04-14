import os, torch, wandb
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import yaml
import json
from PIL import Image

from utils.visualizer import show_comparison
import matplotlib.pyplot as plt
from collections import Counter
from torchvision.transforms.functional import to_pil_image

def convert_semantic_to_targets(masks, num_classes):
    """
    Convert semantic masks (B, H, W) to target format:
    [{'labels': Tensor[num_objects], 'masks': Tensor[num_objects, H, W]}, ...]
    """
    targets = []
    for mask in masks:
        unique_classes = mask.unique()
        labels = []
        binary_masks = []
        for cls in unique_classes:
            if int(cls) == 255:  # ignore class
                continue
            labels.append(cls)
            binary_masks.append((mask == cls).float())
        if len(binary_masks) == 0:
            binary_masks = [torch.zeros_like(mask).float()]
            labels = [torch.tensor(0)]
        target = {
            "labels": torch.stack(labels) if isinstance(labels[0], torch.Tensor) else torch.tensor(labels),
            "masks": torch.stack(binary_masks)
        }
        targets.append(target)
    return targets

def load_coco_class_names(dataset_yaml_path="configs/dataset.yaml", include_background=True):
    """
    Returns:
        dict[int, dict]: Mapping of contiguous class index to a dictionary with keys
                        "coco_id" and "name".  
                        If include_background is True, a background class is added at index 0.
    """
    with open(dataset_yaml_path, "r") as f:
        cfg = yaml.safe_load(f)

    annotation_json = os.path.join(cfg['data_dir'], cfg['annotations']['json_path'])
    with open(annotation_json, 'r') as f:
        coco_data = json.load(f)

    # Build a dictionary: COCO category ID -> name.
    coco_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Get the target category ids (assumed to exclude background).
    target_ids = sorted(cfg['target_categories'])

    class_idx_to_info = {}
    start_idx = 0
    if include_background:
        class_idx_to_info[0] = {"coco_id": 0, "name": "background"}
        start_idx = 1

    for idx, coco_id in enumerate(target_ids, start=start_idx):
        class_idx_to_info[idx] = {
            "coco_id": coco_id,
            "name": coco_id_to_name.get(coco_id, f"COCO ID {coco_id}")
        }

    print("[Info] Loaded COCO category names:")
    for idx, info in class_idx_to_info.items():
        print(f"  Class {idx} -> {info['name']} (COCO ID {info['coco_id']})")

    return class_idx_to_info

def compute_class_frequency(mask_folder, num_classes, save_path=None):
    """
    Computes inverse class weights based on pixel frequency in training masks.
    Returns:
        dict[int, dict]: Mapping from contiguous class index to a dict with keys:
                         "weight", "coco_id", "name"
    """
    if save_path and os.path.exists(save_path):
        print(f"[Info] Loading precomputed class weights from {save_path}")
        class_data = torch.load(save_path)
        df = pd.DataFrame.from_dict(class_data, orient="index")
        df.index.name = "class_id"
        df["weight"] = df["weight"].round(6)
        print(df)
        return class_data

    print(f"[Info] Computing class weights for {num_classes} total classes")

    # Load class names (with background included)
    class_names = load_coco_class_names(include_background=True)

    # Count pixel frequency in all mask images (assumes .png files)
    pixel_counter = Counter()
    for fname in os.listdir(mask_folder):
        if fname.endswith(".png"):
            mask = np.array(Image.open(os.path.join(mask_folder, fname)))
            pixel_counter.update(mask.flatten().tolist())

    total_pixels = sum(pixel_counter.values())
    raw_weights = []
    for c in range(num_classes):
        # To avoid division by zero, ensure a minimal count
        class_pixels = pixel_counter.get(c, 1)
        freq = class_pixels / total_pixels
        weight = 1.0 / freq
        raw_weights.append(weight)

    norm_weights = np.array(raw_weights) / np.sum(raw_weights)

    # Build the final dictionary.
    class_data = {}
    for i in range(num_classes):
        class_data[i] = {
            "weight": float(norm_weights[i]),
            "coco_id": class_names.get(i, {}).get("coco_id", -1),
            "name": class_names.get(i, {}).get("name", f"Class {i}")
        }

    if save_path:
        torch.save(class_data, save_path)
        print(f"[Info] Saved class info to {save_path}")

    df = pd.DataFrame.from_dict(class_data, orient="index")
    df.index.name = "class_id"
    df["weight"] = df["weight"].round(6)
    print(df)

    return class_data


def detailed_log(i, epoch, loss_dict, optimizer, model, wandb_log_freq):
    log_data = {
        "epoch": epoch,
        "iteration": i,
        "total_loss": loss_dict.get("total_loss"),
        "learning_rate": optimizer.param_groups[0]["lr"],
    }
    # Compute total gradient norm
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    log_data["grad_norm"] = total_norm

    if total_norm > 1e5:
        print(f"[Iteration {i}] Gradient norm exploded: {total_norm:.2e}")

    # Log to wandb at the specified frequency
    if i % wandb_log_freq == 0:
        wandb.log(log_data, commit=True)


def run_test_loop(model, test_loader, device, epoch, criterion, results_dir, epochs, class_info, inbuilt=False):
    model.eval()
    save_dir = os.path.join(results_dir, f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc=f"Evaluating {epoch}/{epochs}")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)

            # Choose target format based on model type
            if inbuilt:
                targets = masks
            else:
                targets = convert_semantic_to_targets(masks, num_classes=outputs["pred_logits"].shape[-1])
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss, _ = criterion(outputs, targets)
            total_loss += loss.item()
            count += 1

            if inbuilt:
                preds = torch.argmax(outputs["out"], dim=1)
            else:
                preds = torch.argmax(outputs["pred_masks"], dim=1)

            for j in range(0, images.size(0), 50):
                img_np = to_pil_image(images[j].detach().cpu()).convert("RGB")
                mask_np = masks[j].cpu().numpy()
                pred_np = preds[j].cpu().numpy()
                fig = show_comparison(original=np.array(img_np), gt_mask=mask_np, seg_mask=pred_np, class_info=class_info)
                fig.savefig(os.path.join(save_dir, f"comparison_{i}_{j}.png"))
                plt.close(fig)

    avg_test_loss = total_loss / count if count else 0.0
    print(f"Average Test Loss: {avg_test_loss:.4f}")
    return avg_test_loss


def save_checkpoint(model, optimizer=None, scheduler=None, epoch=None, path=None, inf_only=False):
    if inf_only:
        torch.save(model.state_dict(), path)
    else:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        }, path)

def load_checkpoint(path, model, optimizer, scheduler, device):
    print(f"[Resuming] Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint.get("epoch", 0)

