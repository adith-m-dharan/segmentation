import os, sys, torch, wandb
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
import numpy as np
import multiprocessing
from omegaconf import OmegaConf
import argparse
import matplotlib.pyplot as plt
from collections import Counter

from datasets.lmdb_dataset import LmdbSegmentationDataset
from models.vision_system import VisionSystem
# from models.vision_system_import import VisionSystemImport
from utils.visualizer import overlay_mask_on_image, show_comparison
from utils.lmdb_utils import prepare_lmdb

torch.backends.cudnn.benchmark = True
num_workers = min(4, multiprocessing.cpu_count() // 2)

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)
warnings.filterwarnings("ignore", message="`torch.cuda.amp.*` is deprecated", category=FutureWarning)


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


def main():
    torch.autograd.set_detect_anomaly(True)
    # Parse CLI for config path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/train.yaml', help="Path to config YAML")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    # Set device from config
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Define transform for images
    image_transform = transforms.Compose([
        transforms.Resize(tuple(cfg.dataset.transform_size)),
        transforms.ToTensor()
    ])

    # Ensure LMDB is prepared (creates LMDB if missing)
    prepare_lmdb()

    # Create DataLoader for training and testing
    train_dataset = LmdbSegmentationDataset(
        cfg.dataset.train_image_lmdb,
        cfg.dataset.train_mask_lmdb,
        transform=image_transform,
        target_transform=lambda x: torch.from_numpy(
            np.array(x.resize(tuple(cfg.dataset.transform_size), resample=Image.NEAREST))
        ).long()
    )
    test_dataset = LmdbSegmentationDataset(
        cfg.dataset.test_image_lmdb,
        cfg.dataset.test_mask_lmdb,
        transform=image_transform,
        target_transform=lambda x: torch.from_numpy(
            np.array(x.resize(tuple(cfg.dataset.transform_size), resample=Image.NEAREST))
        ).long()
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True,
        num_workers=cfg.hardware.num_workers, 
        pin_memory=cfg.hardware.pin_memory, 
        persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=False,
        num_workers=cfg.hardware.num_workers, 
        pin_memory=cfg.hardware.pin_memory, 
        persistent_workers=True
    )

    mask_dir = os.path.join(cfg.paths.train_mask_dir)  # e.g., data/prepared/train/masks
    num_classes = cfg.model.num_classes  # include background if needed
    class_weights = compute_class_frequency(mask_dir, num_classes, cfg.paths.weights_path).to(device)

    # Build the model from the config
    # model = VisionSystemImport()
    model = VisionSystem(config_path=cfg.paths.model_config).to(device)
    criterion = ComboLoss(weight_ce=cfg.training.weight_ce, weight_dice=cfg.training.weight_dice, class_weights=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma)

    # Initialize WandB if enabled
    if cfg.logging.use_wandb:
        wandb.init(project=cfg.logging.wandb_project, config=OmegaConf.to_container(cfg))
        # Optional: use wandb.watch to log parameter histograms/gradients
        wandb.watch(model, log="all", log_freq=cfg.logging.wandb_log_freq)

    # Create checkpoint and results directories as per config
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.results_dir, exist_ok=True)

    # Optionally load checkpoint for resume training
    start_epoch = 0
    if os.path.exists(cfg.paths.output_checkpoint):
        start_epoch = load_checkpoint(cfg.paths.output_checkpoint, model, optimizer, scheduler, device)
        print(f"Resuming from epoch {start_epoch+1}")

    scaler = GradScaler() if cfg.hardware.use_amp else None

    # Begin training loop
    try:
        for epoch in range(start_epoch+1, cfg.training.epochs):
            model.train()
            epoch_loss = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}")
            for i, (images, masks) in enumerate(loop):
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()

                with autocast(enabled=cfg.hardware.use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                # ==== NaN/Inf Loss Recovery ====
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Warning] Loss exploded at Epoch {epoch}, Iteration {i}. Reloading checkpoint.")
                    torch.cuda.empty_cache()
                    load_checkpoint(cfg.paths.output_checkpoint, model, optimizer, scheduler, device)
                    scaler = GradScaler() if cfg.hardware.use_amp else None
                    epoch -= 1  # Re-run this epoch from the start
                    break  # Break out of batch loop to restart epoch

                # Continue normal training
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())

                # Detailed WandB logging at iteration level
                loss_dict = {"total_loss": loss.item()}
                if cfg.logging.use_wandb and (i % cfg.logging.wandb_log_freq == 0):
                    detailed_log(i, epoch, loss_dict, optimizer, model, cfg.logging.wandb_log_freq)

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
            if cfg.logging.use_wandb:
                wandb.log({"epoch": epoch, "train_loss": avg_loss}, commit=False)

            # Step the learning rate scheduler if applicable
            scheduler.step()

            # === Save full training checkpoint ===
            save_checkpoint(model, optimizer, scheduler, epoch, path=cfg.paths.output_checkpoint)

            # === Save inference-only model ===
            save_checkpoint(model, path=cfg.paths.output_inference_model, inf_only=True)

            # === Save legacy backup only every 10 epochs ===
            if (epoch) % 10 == 0:
                os.makedirs(os.path.join(cfg.paths.checkpoint_dir), exist_ok=True)
                backup_ckpt_path = os.path.join(cfg.paths.checkpoint_dir, f"latest_epoch_{epoch}.pth")
                backup_inf_path = os.path.join(cfg.paths.checkpoint_dir, f"model_epoch_{epoch}.pth")

                # Save full legacy checkpoint
                save_checkpoint(model, optimizer, scheduler, epoch, path=backup_ckpt_path)

                # Save legacy inference-only model
                save_checkpoint(model, path=backup_inf_path, inf_only=True)

            # Run test loop every eval_freq epochs
            if (epoch) % cfg.training.eval_freq == 0:
                test_loss = run_test_loop(model, test_loader, device, epoch, criterion, cfg.paths.results_dir, cfg.training.epochs)
                if cfg.logging.use_wandb:
                    wandb.log({"epoch": epoch, "test_loss": test_loss}, commit=False)

            # Log learning rate at epoch end
            if cfg.logging.use_wandb:
                wandb.log({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}, commit=True)

            # Optional: clear GPU cache at end of each epoch
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\nTraining interrupted. Exiting...")
        torch.cuda.empty_cache()
        if cfg.logging.use_wandb:
            wandb.finish()
        sys.exit(0)

    print("Training complete.")
    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segmentation Model with Detailed Logging")
    parser.add_argument('--config', type=str, default='configs/train.yaml', help="Path to the YAML config file")
    args = parser.parse_args()
    # Load configuration
    config = OmegaConf.load(args.config)
    # Call main training function with the config
    main()
