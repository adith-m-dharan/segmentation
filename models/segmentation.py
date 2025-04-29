
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.swin_backbone import SwinBackbone
from models.mask2former_head import Mask2FormerHead

class Segmentation(nn.Module):
    def __init__(self, config_path=None, config_dict=None):
        super(Segmentation, self).__init__()

        if config_path:
            with open(config_path, 'r') as f:
                cfg = yaml.safe_load(f)
        elif config_dict:
            cfg = config_dict
        else:
            raise ValueError("Provide either config_path or config_dict to Segmentation")

        self.backbone = SwinBackbone(**cfg["backbone"])
        self.head = Mask2FormerHead(**cfg["head"])


    def forward(self, x):

        backbone_features = self.backbone(x)
        head_features = backbone_features[::-1]  
        
        cls_logits, mask_logits, aux_masks = self.head(head_features)
        
        pred_masks = F.interpolate(mask_logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        
        return {
            "pred_masks": pred_masks,      # Expected shape: [B, Q, H, W]
            "pred_logits": cls_logits,     # Expected shape: [B, Q, num_classes+1]
            "aux_outputs": aux_masks       # A list of auxiliary mask predictions, each with shape [B, Q, H, W]
        }




# Quick test for the Segmentation model.
if __name__ == "__main__":
    from types import SimpleNamespace

    B = 1
    img_size = 512
    num_classes = 8
    num_queries = 100
    config_path = "configs/model.yaml"  # Set to None to enable the loop test

    if config_path is None:
        for num_stages in range(2, 6):  # P2 to P(num_stages+1)
            embed_dim = 16
            depth = 2
            num_heads = 2
            window_size = 8

            print(f"\n--- Testing Segmentation with {num_stages} stages ---")

            # Compute Swin output channels
            backbone_channels = [embed_dim * (2**i) for i in range(num_stages)]
            pixel_decoder_inputs = backbone_channels[::-1]  # [P5, P4, ..., P2]

            # Prepare config dynamically
            cfg = SimpleNamespace()
            cfg.backbone = {
                "img_size": img_size,
                "patch_size": 4,
                "in_chans": 3,
                "embed_dim": embed_dim,
                "depth": depth,
                "num_heads": num_heads,
                "window_size": window_size,
                "shift_size": 4,
                "num_stages": num_stages,
            }
            cfg.head = {
                "feature_project_dim": 128,
                "input_channels": pixel_decoder_inputs,
                "num_classes": num_classes,
                "d_model": 64,
                "num_queries": num_queries,
                "nhead": 2,
                "num_decoder_layers": 3,
                "dropout": 0.1
            }

            # Initialize model and run forward pass
            model = Segmentation(config_dict={"backbone": cfg.backbone, "head": cfg.head})

            dummy = torch.randn(B, 3, img_size, img_size)
            outputs = model(dummy)

            print("pred_masks:", outputs["pred_masks"].shape)
            print("pred_logits:", outputs["pred_logits"].shape)
            print("aux_outputs:", len(outputs["aux_outputs"]))

    else:
        print("\n--- Testing Segmentation with config file ---")
        model = Segmentation(config_path=config_path)
        dummy = torch.randn(B, 3, img_size, img_size)
        outputs = model(dummy)

        print("pred_masks:", outputs["pred_masks"].shape)
        print("pred_logits:", outputs["pred_logits"].shape)
        print("aux_outputs:", len(outputs["aux_outputs"]))