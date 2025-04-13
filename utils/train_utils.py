import os, torch, wandb
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import numpy as np

from utils.visualizer import show_comparison
import matplotlib.pyplot as plt
from collections import Counter
from torchvision.transforms.functional import to_pil_image


class ComboLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0, smooth=1.0, class_weights=None):
        """
        Args:
            weight_ce (float): Weight for CrossEntropyLoss.
            weight_dice (float): Weight for Dice Loss.
            smooth (float): Smoothing constant for Dice Loss.
            class_weights (torch.Tensor or None): Class weights for CrossEntropyLoss.
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.smooth = smooth

    def forward(self, preds, targets):
        ce_loss = self.ce(preds, targets)
        dice_loss = self._dice_loss(preds, targets)
        return self.weight_ce * ce_loss + self.weight_dice * dice_loss

    def _dice_loss(self, preds, targets):
        preds = torch.softmax(preds, dim=1)
        num_classes = preds.size(1)
        dice = 0.0
        for c in range(num_classes):
            pred_c = preds[:, c, :, :]
            target_c = (targets == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            denom = torch.clamp(union + self.smooth, min=1e-6)
            dice += 1 - (2. * intersection + self.smooth) / denom
        return dice / num_classes


def compute_class_frequency(mask_folder, num_classes, save_path=None):
    """
    Computes inverse class weights based on pixel frequency in the training masks.

    Args:
        mask_folder (str): Path to folder containing .png mask files.
        num_classes (int): Number of classes (including background as class 0).
        save_path (str): Path to save the computed weights (default: class_weights.pt).

    Returns:
        torch.Tensor: Normalized class weights of shape (num_classes,).
    """

    if save_path is not None and os.path.exists(save_path):
        print(f"[Info] Loading precomputed class weights from {save_path}")
        weights = torch.load(save_path).float()
        # weights = weights / weights.sum()  # Normalize to sum to 1

        print("Loaded Class Weights (including background):")
        for c, w in enumerate(weights.tolist()):
            print(f"  Class {c}: {w:.6f}")
        return weights

    print(f"[Info] Computing class weights for {num_classes} total classes (including background)")
    
    # Sanity check: print unique labels
    unique_labels = set()
    for fname in os.listdir(mask_folder):
        if fname.endswith('.png'):
            mask = np.array(Image.open(os.path.join(mask_folder, fname)))
            unique_labels.update(np.unique(mask).tolist())
    print(f"[Check] Unique labels found in masks: {sorted(unique_labels)}")

    # Count pixels
    pixel_counter = Counter()
    for fname in os.listdir(mask_folder):
        if fname.endswith('.png'):
            mask = np.array(Image.open(os.path.join(mask_folder, fname)))
            pixel_counter.update(mask.flatten().tolist())

    total_pixels = sum(pixel_counter.values())
    weights = []
    for c in range(num_classes):
        class_pixels = pixel_counter.get(c, 1)  # Avoid divide-by-zero
        freq = class_pixels / total_pixels
        weight = 1.0 / freq
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum()  # Normalize to sum to 1

    print("Class Weights (including background):")
    for c, w in enumerate(weights.tolist()):
        print(f"  Class {c}: {w:.6f}")

    torch.save(weights, save_path)
    print(f"[Info] Class weights saved to {save_path}")
    return weights

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


def run_test_loop(model, test_loader, device, epoch, criterion, results_dir, epochs):
    model.eval()
    save_dir = os.path.join(results_dir, f"epoch_{epoch}")
    os.makedirs(save_dir, exist_ok=True)
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc=f"Evaluating {epoch}/{epochs}")):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            count += 1

            preds = torch.argmax(outputs, dim=1)
            for j in range(0, images.size(0), 50):
                img_np = to_pil_image(images[j].detach().cpu()).convert("RGB")
                mask_np = masks[j].cpu().numpy()
                pred_np = preds[j].cpu().numpy()
                fig = show_comparison(original=np.array(img_np), gt_mask=mask_np, seg_mask=pred_np)
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

