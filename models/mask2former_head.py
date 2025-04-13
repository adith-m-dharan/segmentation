# models/mask2former_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelDecoder(nn.Module):
    def __init__(self, input_channels, d_model):
        """
        Fuse multi-scale features from the backbone in a top-down manner.

        Args:
            input_channels (list): Channels of each input feature in order [C2, C3, C4].
                                   (e.g. [128, 256, 512] or [512, 512, 512])
            d_model (int): Target output channel dimension (e.g. 256 or 512).
        """
        super(PixelDecoder, self).__init__()
        # Lateral convolutions to match each feature to d_model dims
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, d_model, kernel_size=1) for in_ch in input_channels
        ])
        # Output convolutions for refinement after fusion
        self.output_convs = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1) for _ in input_channels
        ])

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): High-to-low resolution features, e.g. [C2, C3, C4]
                where shapes are like [(B, 512, 56, 56), (B, 512, 28, 28), (B, 512, 14, 14)]
        Returns:
            Tensor: Fused feature map of shape (B, d_model, H, W) where H, W = highest resolution (C2).
        """
        # Reverse feature order to low-to-high resolution: [C4, C3, C2]
        features = features[::-1]

        # Apply lateral projection
        laterals = [l_conv(f) for f, l_conv in zip(features, self.lateral_convs)]

        # Start from lowest resolution
        x = laterals[0]
        for i in range(1, len(laterals)):
            # Upsample and fuse with next resolution
            x = F.interpolate(x, size=laterals[i].shape[2:], mode='bilinear', align_corners=False)
            x = x + laterals[i]
            x = self.output_convs[i](x)
        return x


class Mask2FormerHead(nn.Module):
    def __init__(self, input_channels, num_classes,
                 d_model=256, num_queries=100, nhead=8, num_decoder_layers=6):
        super(Mask2FormerHead, self).__init__()
        # 1. Pixel Decoder: Fuse multi-scale features.
        self.pixel_decoder = PixelDecoder(input_channels, d_model)
        # 2. Transformer Decoder: Process learnable query embeddings.
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.query_embed = nn.Embedding(num_queries, d_model)
        # 3. Dynamic Kernel Generation: Map aggregated query features to a modulation kernel.
        self.kernel_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model)
        )
        # 4. Final Mask Head: Predict segmentation logits.
        self.mask_head = nn.Conv2d(d_model, num_classes, kernel_size=1)
    
    def forward(self, features):
        """
        Args:
            features (list[Tensor]): List of backbone features in order [C5, C4, C3, C2].
        Returns:
            Tensor: Segmentation logits of shape (B, num_classes, H, W), with H,W matching the fused feature map.
        """
        # Fuse features via pixel decoder.
        fused_features = self.pixel_decoder(features)  # (B, d_model, H, W)
        B, C, H, W = fused_features.shape
        
        # Flatten spatial dimensions to create memory for transformer decoding.
        memory = fused_features.flatten(2).permute(2, 0, 1)  # (H*W, B, d_model)
        
        # Prepare learnable queries: (num_queries, B, d_model)
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)
        
        # Apply transformer decoder.
        hs = self.transformer_decoder(tgt=queries, memory=memory)  # (num_queries, B, d_model)
        
        # Aggregate query features—here we use a simple mean across queries.
        query_feature = hs.mean(dim=0)  # (B, d_model)
        
        # Generate a dynamic modulation kernel.
        dynamic_kernel = self.kernel_head(query_feature).unsqueeze(2).unsqueeze(3)  # (B, d_model, 1, 1)
        
        # Modulate the fused pixel features.
        modulated_features = fused_features * dynamic_kernel  # (B, d_model, H, W)
        
        # Predict segmentation logits.
        logits = self.mask_head(modulated_features)  # (B, num_classes, H, W)
        return logits

# # Quick test for the Mask2Former head.
# if __name__ == "__main__":
#     # Create dummy features simulating backbone outputs.
#     # Order: [C5, C4, C3, C2] (from lowest to highest resolution).
#     dummy_features = [
#         torch.randn(1, 1024, 7, 7),   # C5
#         torch.randn(1, 768, 14, 14),   # C4
#         torch.randn(1, 512, 28, 28),   # C3
#         torch.randn(1, 256, 56, 56)    # C2
#     ]
#     head = Mask2FormerHead(input_channels=[1024, 768, 512, 256], num_classes=81,
#                                d_model=256, num_queries=100, nhead=8, num_decoder_layers=6)
#     logits = head(dummy_features)
#     print("Mask2FormerHead logits shape:", logits.shape)
