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
        # Ensure shift_size is smaller than window_size
        self.shift_size = shift_size if shift_size < window_size else window_size // 2

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
        assert L == H * W, "Input feature has wrong size"
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # If shift_size > 0, apply cyclic shift (roll)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
        else:
            shifted_x = x

        # Partition into windows
        x_windows, H_pad, W_pad = window_partition(shifted_x, self.window_size)
        attn_windows = self.attn(x_windows)  # (num_windows*B, C, window_size, window_size)
        shifted_x = window_reverse(attn_windows, self.window_size, H_pad, W_pad, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
        else:
            x = shifted_x

        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

class SwinBackbone(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depth=2, num_heads=4,
                 window_size=8, shift_size=0, num_stages=4):
        super(SwinBackbone, self).__init__()
        print(f"[SwinBackbone] img_size={img_size}, patch_size={patch_size}, in_chans={in_chans}, "
              f"embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}, "
              f"window_size={window_size}, shift_size={shift_size}, num_stages={num_stages}")
        
        self.num_stages = num_stages
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.resolution = (img_size // patch_size, img_size // patch_size)

        # Containers for all stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Build each stage dynamically
        in_dim = embed_dim
        H, W = self.resolution
        for i in range(num_stages):
            stage_blocks = nn.ModuleList([
                SwinBlock(
                    dim=in_dim,
                    input_resolution=(H, W),
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=(shift_size if i > 0 else 0)
                )
                for _ in range(depth)
            ])
            self.stages.append(stage_blocks)
            self.norms.append(nn.LayerNorm(in_dim))

            # Downsample only if it's not the last stage
            if i < num_stages - 1:
                self.downsamples.append(nn.Conv2d(in_dim, in_dim * 2, kernel_size=2, stride=2))
                in_dim *= 2
                H //= 2
                W //= 2

    def forward(self, x: torch.Tensor) -> list:
        x = self.patch_embed(x)
        B, C, H, W = x.shape
        features = []

        in_dim = self.embed_dim
        for i in range(self.num_stages):
            x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)

            # SwinBlock expects correct input_resolution: update dynamically here
            for blk in self.stages[i]:
                blk.input_resolution = (H, W)  # dynamically correct resolution
                x_flat = blk(x_flat)

            x = self.norms[i](x_flat).transpose(1, 2).view(B, in_dim, H, W)
            features.append(x)

            # Downsample unless this is the last stage
            if i < self.num_stages - 1:
                x = self.downsamples[i](x)
                B, in_dim, H, W = x.shape  # update dims for next stage

        return features


# Quick test (optional):
if __name__ == "__main__":
    import yaml
    import os

    config_path = "configs/model.yaml"  # Set to None for looped test
    B = 1

    if config_path is not None and os.path.exists(config_path):
        print(f"\nLoading SwinBackbone config from: {config_path}")
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        backbone_cfg = cfg["scratch"]["backbone"]

        model = SwinBackbone(**backbone_cfg)
        dummy = torch.randn(B, backbone_cfg["in_chans"], backbone_cfg["img_size"], backbone_cfg["img_size"])
        outputs = model(dummy)

        print(f"Output count: {len(outputs)} (Expected: {backbone_cfg['num_stages']})")
        expected_channels = [backbone_cfg["embed_dim"] * (2 ** i) for i in range(backbone_cfg["num_stages"])]
        expected_res = backbone_cfg["img_size"] // backbone_cfg["patch_size"]

        for i, (feat, ch) in enumerate(zip(outputs, expected_channels), start=2):
            _, C, H, W = feat.shape
            expected_h = expected_res // (2 ** (i - 2))
            print(f"  P{i}: {C} channels, {H}x{W} resolution")
            assert C == ch, f"Channel mismatch at P{i}: got {C}, expected {ch}"
            assert H == expected_h and W == expected_h, f"Resolution mismatch at P{i}: got {H}x{W}, expected {expected_h}x{expected_h}"
            print("    Channels + Resolution OK")

    else:
        print("\nNo YAML config provided — running dynamic loop test")
        stage_list = [2, 3, 4, 5, 6, 7, 8]
        embed_dim = 16
        patch_size = 4
        img_size = 512

        for num_stages in stage_list:
            print(f"\n--- Testing SwinBackbone with {num_stages} stages ---")
            model = SwinBackbone(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
                depth=2,
                num_heads=2,
                window_size=8,
                shift_size=4,
                num_stages=num_stages
            )
            dummy = torch.randn(B, 3, img_size, img_size)
            outputs = model(dummy)

            assert len(outputs) == num_stages, f"Got {len(outputs)} outputs, expected {num_stages}"
            print(f"Output count: {len(outputs)}")

            expected_channels = [embed_dim * (2 ** i) for i in range(num_stages)]
            expected_res = img_size // patch_size

            for i, (feat, ch) in enumerate(zip(outputs, expected_channels), start=2):
                _, C, H, W = feat.shape
                expected_h = expected_res // (2 ** (i - 2))
                print(f"  P{i}: {C} channels, {H}x{W} resolution")
                assert C == ch, f"Channel mismatch at P{i}: got {C}, expected {ch}"
                assert H == expected_h and W == expected_h, f"Resolution mismatch at P{i}: got {H}x{W}, expected {expected_h}x{expected_h}"
                print("    Channels + Resolution OK")
