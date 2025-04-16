
import os, sys, torch, wandb
import torch.optim as optim
import torch.nn.functional as F
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
from models.segmentation import Segmentation
from models.loss import MaskLoss
from utils.lmdb_utils import prepare_lmdb
from utils.train_utils import *
import warnings
torch.backends.cudnn.benchmark = True
num_workers = min(4, multiprocessing.cpu_count() // 2)

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`", category=FutureWarning)
warnings.filterwarnings("ignore", message="`torch.cuda.amp.*` is deprecated", category=FutureWarning)


def setup_model(cfg, device):
    # Class Weights
    mask_dir = os.path.join(cfg.paths.train_mask_dir)
    num_classes = cfg.model.num_classes
    class_info = compute_class_frequency(mask_dir, num_classes, cfg.paths.weights_path)
    class_weights = torch.tensor([class_info[c]["weight"] for c in sorted(class_info)]).to(device)

    if cfg.model.inbuilt:
        from torchvision.models.segmentation import fcn_resnet50
        from torchvision.models import resnet50

        criterion = lambda outputs, targets: (
            F.cross_entropy(outputs["out"], targets.to(outputs["out"].device), weight=class_weights),
            {"cross_entropy": F.cross_entropy(outputs["out"], targets.to(outputs["out"].device), weight=class_weights).item()}
        )
        backbone = resnet50(weights=None)
        model = fcn_resnet50(weights=None, num_classes=num_classes).to(device)
        class_weights = torch.ones(num_classes).to(device)

    else:
        model = Segmentation(config_path=cfg.paths.model_config).to(device)
        criterion = MaskLoss(
            lambda_cls=cfg.training.weight_ce,
            lambda_mask=cfg.training.weight_mask_bce,
            lambda_dice=cfg.training.weight_dice,
            class_weights=class_weights,
            aux_weight=cfg.training.weight_aux
        )

    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.training.step_size, gamma=cfg.training.gamma)

    return model, criterion, optimizer, scheduler


def setup_data(cfg):

    # Image transform
    image_transform = transforms.Compose([
        transforms.Resize(tuple(cfg.dataset.transform_size)),
        transforms.ToTensor()
    ])

    # Ensure LMDB is prepared
    prepare_lmdb(cfg.dataset.base_input, cfg.dataset.base_output)

    # Dataset definitions
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

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.hardware.num_workers, pin_memory=cfg.hardware.pin_memory, persistent_workers=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.hardware.num_workers, pin_memory=cfg.hardware.pin_memory, persistent_workers=True
    )

    return train_loader, test_loader

def tracking_logs(epoch, avg_loss, model, optimizer, scheduler, cfg,
                  test_loader, device, criterion):
    if cfg.logging.use_wandb:
        wandb.log({"epoch": epoch, "train_loss": avg_loss}, commit=False)

    # Periodic backups
    if epoch % 10 == 0:
        backup_ckpt = os.path.join(cfg.paths.checkpoint_dir, f"latest_epoch_{epoch}.pth")
        backup_inf  = os.path.join(cfg.paths.checkpoint_dir, f"model_epoch_{epoch}.pth")
        save_checkpoint(model, optimizer, scheduler, epoch, path=backup_ckpt)
        save_checkpoint(model, path=backup_inf, inf_only=True)

    # Periodic evaluation
    if epoch % cfg.training.eval_freq == 0:
        test_loss = run_test_loop(
            model, test_loader, device, epoch,
            criterion, cfg.paths.results_dir,
            cfg.training.epochs, cfg.paths.weights_path,
            cfg.model.inbuilt
        )
        if cfg.logging.use_wandb:
            wandb.log({"epoch": epoch, "test_loss": test_loss}, commit=False)

    # Learning rate logging
    if cfg.logging.use_wandb:
        wandb.log({"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}, commit=True)

def main(cfg):
    torch.autograd.set_detect_anomaly(True)

    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Ensure LMDB is prepared
    prepare_lmdb(cfg.dataset.base_input, cfg.dataset.base_output)

    train_loader, test_loader = setup_data(cfg)
    model, criterion, optimizer, scheduler = setup_model(cfg, device)

    # WandB
    if cfg.logging.use_wandb:
        wandb.init(project=cfg.logging.wandb_project, config=OmegaConf.to_container(cfg), dir=cfg.logging.wandb_dir)
        wandb.watch(model, log="all", log_freq=cfg.logging.wandb_log_freq)

    # Resume checkpoint
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.results_dir, exist_ok=True)
    start_epoch = 0
    if os.path.exists(cfg.paths.output_checkpoint):
        start_epoch = load_checkpoint(cfg.paths.output_checkpoint, model, optimizer, scheduler, device)
        print(f"Resuming from epoch {start_epoch + 1}")

    scaler = GradScaler() if cfg.hardware.use_amp else None

    try:
        for epoch in range(start_epoch + 1, cfg.training.epochs + 1):
            model.train()
            epoch_loss = 0.0
            loop = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.training.epochs}")

            for i, (images, masks) in enumerate(loop):
                images = images.to(device)
                masks = masks.to(device)

                if cfg.model.inbuilt:
                    targets = masks  # Use raw masks for the inbuilt model
                else:
                    targets = convert_semantic_to_targets(masks, cfg.model.num_classes)
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()

                with autocast(enabled=cfg.hardware.use_amp):
                    outputs = model(images)
                    loss, loss_dict = criterion(outputs, targets)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[Warning] Loss exploded at Epoch {epoch}, Iteration {i}. Reloading checkpoint.")
                    torch.cuda.empty_cache()
                    load_checkpoint(cfg.paths.output_checkpoint, model, optimizer, scheduler, device)
                    scaler = GradScaler() if cfg.hardware.use_amp else None
                    epoch -= 1
                    break

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.training.max_norm)
                    optimizer.step()

                epoch_loss += loss.item()
                loop.set_postfix(loss=loss.item())

                if cfg.logging.use_wandb and (i % cfg.logging.wandb_log_freq == 0):
                    log_dict = {**loss_dict, "total_loss": loss.item()}
                    detailed_log(i, epoch, log_dict, optimizer, model, cfg.logging.wandb_log_freq)

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")

            scheduler.step()

            # Save model
            save_checkpoint(model, optimizer, scheduler, epoch, path=cfg.paths.output_checkpoint)
            save_checkpoint(model, path=cfg.paths.output_inference_model, inf_only=True)

            tracking_logs(epoch, avg_loss, model, optimizer, scheduler, cfg,
              test_loader, device, criterion)

            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("Training interrupted.")
        if cfg.logging.use_wandb:
            wandb.finish()
        sys.exit(0)

    print("Training complete.")
    if cfg.logging.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Segmentation Model")
    parser.add_argument('--config', type=str, default='configs/train.yaml', help="Path to YAML config")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    main(config)
