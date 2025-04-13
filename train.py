import os, sys, torch, wandb
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
import numpy as np
import multiprocessing
from omegaconf import OmegaConf
import argparse

from datasets.lmdb_dataset import LmdbSegmentationDataset
from models.vision_system import VisionSystem
from utils.lmdb_utils import prepare_lmdb
from utils.train_utils import *

torch.backends.cudnn.benchmark = True
num_workers = min(4, multiprocessing.cpu_count() // 2)

import warnings
warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)
warnings.filterwarnings("ignore", message="`torch.cuda.amp.*` is deprecated", category=FutureWarning)

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
