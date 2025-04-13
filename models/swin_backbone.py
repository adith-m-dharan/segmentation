# models/swin_backbone.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
    """
    Partition the input tensor x (B, C, H, W) into non-overlapping windows.
    Returns:
        windows: (num_windows*B, C, window_size, window_size)
        H_pad, W_pad: padded height and width
    """
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0)

    H_pad, W_pad = x.shape[2], x.shape[3]
    x = x.view(B, C, H_pad // window_size, window_size, W_pad // window_size, window_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    windows = x.view(-1, C, window_size, window_size)
    return windows, H_pad, W_pad

def window_reverse(windows: torch.Tensor, window_size: int, H_pad: int, W_pad: int, H: int, W: int) -> torch.Tensor:
    """
    Reverse the window partition back to the original tensor.
    Crops out the padding to get back to original (H, W).
    """
    B = int(windows.shape[0] / ((H_pad * W_pad) / (window_size ** 2)))
    x = windows.view(B, H_pad // window_size, W_pad // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    x = x.view(B, -1, H_pad, W_pad)
    return x[:, :, :H, :W]  # remove padding

class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96):
        """
        Image to Patch Embedding using a convolution.
        """
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        return x

class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int):
        """
        Minimal window-based multi-head self-attention with stability enhancements.
        """
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (num_windows * B, C, window_size, window_size)
        B_, C, H, W = x.shape
        N = H * W

        # Reshape input for linear projection
        x = x.view(B_, C, N).transpose(1, 2)  # (B_, N, C)

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B_, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B_, num_heads, N, head_dim)

        # Normalize q and k for stability
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, num_heads, N, N)

        # Softmax with max-subtraction for numerical stability
        attn = attn - attn.max(dim=-1, keepdim=True).values
        attn = attn.softmax(dim=-1)

        # Clamp to avoid very small or large probabilities
        attn = attn.clamp(min=1e-6, max=1.0)

        # Weighted sum
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)  # (B_, N, C)

        # Final linear projection
        x = self.proj(x)  # (B_, N, C)
        x = x.transpose(1, 2).view(B_, C, H, W)  # (B_, C, H, W)
        return x


class SwinBlock(nn.Module):
    def __init__(self, dim: int, input_resolution: Tuple[int, int], num_heads: int, window_size: int = 8, shift_size: int = 0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H*W, C)
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Mismatch in input resolution."
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Partition into windows
        x_windows, H_pad, W_pad = window_partition(x, self.window_size)
        attn_windows = self.attn(x_windows)
        x = window_reverse(attn_windows, self.window_size, H_pad, W_pad, H, W)
        x = x.flatten(2).transpose(1,2)  # (B, H*W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class SwinBackbone(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96,
                 depth: int = 4, num_heads: int = 4, window_size: int = 8):
        super(SwinBackbone, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.resolution = (img_size // patch_size, img_size // patch_size)
        self.stage1 = nn.ModuleList([
            SwinBlock(dim=embed_dim, input_resolution=self.resolution, num_heads=num_heads, window_size=window_size)
            for _ in range(depth)
        ])
        self.down1 = nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=2, stride=2)

        res2 = (self.resolution[0] // 2, self.resolution[1] // 2)
        self.stage2 = nn.ModuleList([
            SwinBlock(dim=embed_dim * 2, input_resolution=res2, num_heads=num_heads, window_size=window_size)
            for _ in range(depth)
        ])
        self.down2 = nn.Conv2d(embed_dim * 2, embed_dim * 4, kernel_size=2, stride=2)

        res3 = (res2[0] // 2, res2[1] // 2)
        self.stage3 = nn.ModuleList([
            SwinBlock(dim=embed_dim * 4, input_resolution=res3, num_heads=num_heads, window_size=window_size)
            for _ in range(depth)
        ])

        # Add layer norms per stage
        self.norms = nn.ModuleList([
            nn.LayerNorm(embed_dim),
            nn.LayerNorm(embed_dim * 2),
            nn.LayerNorm(embed_dim * 4),
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> list:
        # Initial patch embedding
        x = self.patch_embed(x)  # (B, C, H, W)
        B, C, H, W = x.shape
        features = []

        # Stage 1
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        for blk in self.stage1:
            x_flat = blk(x_flat)
        x = self.norms[0](x_flat).transpose(1, 2).view(B, C, H, W)
        features.append(x)

        # Downsample → Stage 2
        x = self.down1(x)  # (B, C*2, H/2, W/2)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        for blk in self.stage2:
            x_flat = blk(x_flat)
        x = self.norms[1](x_flat).transpose(1, 2).view(B, C, H, W)
        features.append(x)

        # Downsample → Stage 3
        x = self.down2(x)  # (B, C*2, H/2, W/2)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        for blk in self.stage3:
            x_flat = blk(x_flat)
        x = self.norms[2](x_flat).transpose(1, 2).view(B, C, H, W)
        features.append(x)

        return features  # [C2, C3, C4]
    

# # Quick test:
# if __name__ == "__main__":
#     model = SwinBackbone(img_size=224, patch_size=4, embed_dim=96, depth=4, num_heads=4, window_size=8)
#     dummy = torch.randn(1, 3, 224, 224)
#     output = model(dummy)
#     print("SwinBackbone output shape:", output[0].shape)
