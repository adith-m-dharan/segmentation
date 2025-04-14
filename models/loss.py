
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import random
import numpy as np


class MaskLoss(nn.Module):
    def __init__(self, 
                 class_weights,
                 lambda_cls=1.0, 
                 lambda_mask=5.0, 
                 lambda_dice=5.0,
                 num_sample_points=125,
                 aux_weight=0.5,
                 epsilon=1e-6):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_mask = lambda_mask
        self.lambda_dice = lambda_dice
        self.num_sample_points = num_sample_points
        self.aux_weight = aux_weight
        self.epsilon = epsilon
        self.class_weights = class_weights

    def dice_loss(self, pred_mask, target_mask):
        pred_flat = pred_mask.view(-1)
        target_flat = target_mask.view(-1)
        intersection = (pred_flat * target_flat).sum()
        union = (pred_flat * pred_flat).sum() + (target_flat * target_flat).sum() + self.epsilon
        dice_coef = 2.0 * intersection / union
        return 1.0 - dice_coef

    def sample_points(self, mask_tensor, num_points):
        H, W = mask_tensor.shape[-2:]
        y = torch.randint(0, H, (num_points,), device=mask_tensor.device)
        x = torch.randint(0, W, (num_points,), device=mask_tensor.device)
        return y, x

    def compute_cost_matrix(self, pred_logits, pred_masks, target_labels, target_masks):
        Q, C = pred_logits.shape
        T, H, W = target_masks.shape

        log_probs = F.log_softmax(pred_logits, dim=1)
        target_labels_exp = target_labels.unsqueeze(0).expand(Q, T)
        cls_cost = -torch.gather(log_probs, 1, target_labels_exp)

        if self.num_sample_points is not None:
            y, x = self.sample_points(pred_masks, self.num_sample_points)
            pred_masks_sampled = pred_masks[:, y, x]
            target_masks_sampled = target_masks[:, y, x]
            pred_masks_exp = pred_masks_sampled[:, None, :].expand(Q, T, -1)
            target_masks_exp = target_masks_sampled[None, :, :].expand(Q, T, -1)
        else:
            pred_masks_exp = pred_masks.view(Q, 1, -1).expand(Q, T, -1)
            target_masks_exp = target_masks.view(1, T, -1).expand(Q, T, -1)

        bce_loss = F.binary_cross_entropy_with_logits(pred_masks_exp, target_masks_exp.float(), reduction='none')
        cost_mask = bce_loss.mean(dim=-1)

        pred_masks_sig = torch.sigmoid(pred_masks).clamp(min=1e-4, max=1-1e-4)
        if self.num_sample_points is not None:
            pred_masks_sampled = pred_masks_sig[:, y, x]
            target_masks_sampled = target_masks[:, y, x]
            pred_masks_sig_exp = pred_masks_sampled.unsqueeze(1)
            target_masks_exp = target_masks_sampled.unsqueeze(0).float()
        else:
            pred_masks_sig_exp = pred_masks_sig.view(Q, 1, -1)
            target_masks_exp = target_masks.view(1, T, -1).float()

        intersection = 2 * (pred_masks_sig_exp * target_masks_exp).sum(dim=-1)
        denominator = (pred_masks_sig_exp**2).sum(dim=-1) + (target_masks_exp**2).sum(dim=-1) + self.epsilon
        dice_loss_mat = 1.0 - (intersection + 1) / (denominator + 1)

        total_cost = self.lambda_cls * cls_cost + self.lambda_mask * cost_mask + self.lambda_dice * dice_loss_mat
        return total_cost.detach().cpu().numpy()

    def hungarian_matching(self, cost_matrix):
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        return row_ind, col_ind

    def loss_per_sample(self, pred_logits, pred_masks, target_labels, target_masks):
        cost_matrix = self.compute_cost_matrix(pred_logits, pred_masks, target_labels, target_masks)
        query_indices, target_indices = self.hungarian_matching(cost_matrix)

        cls_loss_total  = 0.0
        mask_loss_total = 0.0
        dice_loss_total = 0.0

        for q_idx, t_idx in zip(query_indices, target_indices):
            logits = pred_logits[q_idx].unsqueeze(0)
            cls_target = target_labels[t_idx].unsqueeze(0)
            cls_loss_total += F.cross_entropy(logits, cls_target, weight=self.class_weights)

            pred_mask = pred_masks[q_idx]
            tgt_mask  = target_masks[t_idx]
            if pred_mask.shape != tgt_mask.shape:
                pred_mask = F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0), size=tgt_mask.shape[-2:], mode="bilinear", align_corners=False).squeeze()

            mask_loss_total += F.binary_cross_entropy_with_logits(pred_mask, tgt_mask.float(), reduction='mean')
            dice_loss_total += self.dice_loss(torch.sigmoid(pred_mask).clamp(min=1e-4, max=1 - 1e-4), tgt_mask.float())

        num_matches = len(query_indices) if len(query_indices) > 0 else 1
        cls_loss_avg  = cls_loss_total  / num_matches
        mask_loss_avg = mask_loss_total / num_matches
        dice_loss_avg = dice_loss_total / num_matches

        total_loss = self.lambda_cls * cls_loss_avg + self.lambda_mask * mask_loss_avg + self.lambda_dice * dice_loss_avg
        loss_details = {
            "cls_loss": cls_loss_avg,
            "mask_loss": mask_loss_avg,
            "dice_loss": dice_loss_avg,
        }
        return total_loss, loss_details

    def forward(self, predictions, targets):
        batch_size = predictions["pred_logits"].shape[0]
        total_loss = 0.0
        loss_details = {"cls_loss": 0.0, "mask_loss": 0.0, "dice_loss": 0.0}

        # Main prediction
        for i in range(batch_size):
            loss, details = self.loss_per_sample(
                predictions["pred_logits"][i],
                predictions["pred_masks"][i],
                targets[i]["labels"],
                targets[i]["masks"]
            )
            total_loss += loss
            for k in loss_details:
                loss_details[k] += details[k]

        # Aux outputs (mask only)
        if "aux_outputs" in predictions:
            for aux_mask_pred in predictions["aux_outputs"]:
                for i in range(batch_size):
                    loss, _ = self.loss_per_sample(
                        predictions["pred_logits"][i],  # reuse classification
                        aux_mask_pred[i],
                        targets[i]["labels"],
                        targets[i]["masks"]
                    )
                    total_loss += self.aux_weight * loss

        # Final averaging
        total_loss = total_loss / batch_size
        loss_details = {k: v / batch_size for k, v in loss_details.items()}
        return total_loss, loss_details


# ------------------------------
# 🔍 Quick Test
# ------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    batch_size  = 2
    num_queries = 100
    num_classes = 9
    H, W        = 128, 128
    num_aux     = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")

    # Create dummy predictions
    pred_logits = torch.randn(batch_size, num_queries, num_classes, device=device)
    pred_masks  = torch.randn(batch_size, num_queries, H, W, device=device)
    aux_outputs = [torch.randn(batch_size, num_queries, H, W, device=device) for _ in range(num_aux)]
    predictions = {"pred_logits": pred_logits, "pred_masks": pred_masks, "aux_outputs": aux_outputs}

    # Create dummy targets
    targets = []
    for _ in range(batch_size):
        num_objects = random.randint(1, num_queries)
        labels = torch.randint(0, num_classes, (num_objects,), device=device)
        masks = torch.randint(0, 2, (num_objects, H, W), device=device)
        targets.append({"labels": labels, "masks": masks})

    # Instantiate loss
    class_weights = (1.0 / torch.randint(10, 1000, (num_classes,), dtype=torch.float32)).to(device)
    class_weights /= class_weights.sum()
    criterion = MaskLoss(lambda_cls=1.0, lambda_mask=5.0, lambda_dice=5.0, aux_weight=0.5, class_weights=class_weights)

    # Compute loss
    total_loss, loss_details = criterion(predictions, targets)

    print("\n--- Loss Output ---")
    print(f"Total Loss: {total_loss.item():.4f}")
    for k, v in loss_details.items():
        print(f"{k}: {v.item():.4f}")
