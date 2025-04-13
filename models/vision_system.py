import yaml
import torch.nn as nn
import torch.nn.functional as F
from models.swin_backbone import SwinBackbone
from models.mask2former_head import Mask2FormerHead


class VisionSystem(nn.Module):
    def __init__(self, config_path="configs/model.yaml"):
        super().__init__()

        # Load model config
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        # Backbone
        self.backbone = SwinBackbone(**cfg["backbone"])

        # Project each stage to the same feature dim expected by head
        self.feature_projectors = nn.ModuleList([
            nn.Conv2d(128, 512, kernel_size=1),  # C2
            nn.Conv2d(256, 512, kernel_size=1),  # C3
            nn.Conv2d(512, 512, kernel_size=1),  # C4
        ])

        # Mask2Former head
        self.head = Mask2FormerHead(**cfg["head"])

    def forward(self, x):
        # Extract features from Swin backbone
        features = self.backbone(x)  # returns [C2, C3, C4] i.e. [56x56, 28x28, 14x14]

        # Project each feature map to same channel dimension (512)
        projected_features = [
            self.feature_projectors[i](feat) for i, feat in enumerate(features)
        ]

        # Reverse the order to [C4, C3, C2] → [14x14, 28x28, 56x56]
        fused_features = projected_features[::-1]

        # Pass to head
        logits = self.head(fused_features)

        # Resize to match input
        logits = F.interpolate(logits, size=x.shape[2:], mode="bilinear", align_corners=False)
        return logits
