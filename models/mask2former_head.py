# # models/mask2former_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusionBlock(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Fuse two features using multi-head cross-attention.
        Query comes from an upsampled (lower-res) fused feature.
        Key and Value come from the corresponding lateral (higher-res) feature.
        """
        super(AttentionFusionBlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # 1x1 convs for Q, K, V projections.
        self.q_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.k_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.v_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.out_proj = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # An MLP for additional refinement.
        self.mlp = nn.Sequential(
            nn.Conv2d(d_model, d_model * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_model * 4, d_model, kernel_size=1)
        )

    def forward(self, query_feat, lateral_feat):
        """
        Args:
            query_feat: (B, d_model, H, W) from lower-res path (upsampled)
            lateral_feat: (B, d_model, H, W) from lateral (higher-res) branch.
        Returns:
            Fused feature: (B, d_model, H, W)
        """
        B, C, H, W = query_feat.shape
        Q = self.q_proj(query_feat)
        K = self.k_proj(lateral_feat)
        V = self.v_proj(lateral_feat)
        Q = Q.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
        K = K.flatten(2).transpose(1, 2)
        V = V.flatten(2).transpose(1, 2)
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, H*W, head_dim)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, H*W, H*W)
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = out.transpose(1, 2).view(B, self.d_model, H, W)
        out = self.out_proj(out)
        out = out + query_feat
        out = out + self.mlp(out)
        return out


class PixelDecoder(nn.Module):
    def __init__(self, input_channels, d_model, num_heads):
        """
        Args:
            input_channels (list): List of backbone channel dims in order [P5, P4, P3, P2].
            d_model (int): Common dimension.
            num_heads (int): Number of attention heads.
        """
        super(PixelDecoder, self).__init__()
        self.proj_convs = nn.ModuleList([nn.Conv2d(ch, d_model, kernel_size=1) for ch in input_channels])
        self.fusion_blocks = nn.ModuleList([
            AttentionFusionBlock(d_model, num_heads) for _ in range(len(input_channels) - 1)
        ])
        assert len(self.fusion_blocks) == len(self.proj_convs) - 1, "Mismatch in fusion blocks and projected features"
        self.refine_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.refine_norm = nn.GroupNorm(32, d_model)
    
    def forward(self, features):
        proj_feats = [conv(f) for conv, f in zip(self.proj_convs, features)]
        fused = proj_feats[0]
        multi_scale_features = [fused]
        for i in range(1, len(proj_feats)):
            fused = self.fusion_blocks[i - 1](fused, proj_feats[i])
            multi_scale_features.append(fused)
        fused_feature = self.refine_norm(self.refine_conv(fused))
        return fused_feature, multi_scale_features


class MaskedTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, ffn_ratio=4, dropout=0.1):
        """
        A single decoder layer with self-attention, masked cross-attention, and an FFN.
        """
        super(MaskedTransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * ffn_ratio)
        self.linear2 = nn.Linear(d_model * ffn_ratio, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, query, memory, spatial_shape, prev_mask=None):
        num_queries, B, d_model = query.shape
        H_mem, W_mem = spatial_shape
        q_norm = self.norm1(query)
        q2, _ = self.self_attn(q_norm, q_norm, q_norm)
        query = query + self.dropout(q2)
        Q = self.linear_q(self.norm2(query))
        Q = Q.permute(1, 0, 2)  # (B, num_queries, d_model)
        K = self.linear_k(memory).permute(1, 0, 2)  # (B, H_mem*W_mem, d_model)
        V = self.linear_v(memory).permute(1, 0, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_model ** 0.5)
        if prev_mask is not None:
            mask_bias = torch.log(prev_mask.flatten(2) + 1e-6)
            scores = scores + mask_bias
        attn = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn, V)
        attn_output = attn_output.permute(1, 0, 2)
        attn_output = self.out_proj(attn_output)
        query = query + self.dropout(attn_output)
        q_norm2 = self.norm3(query)
        ffn_output = self.linear2(self.dropout(F.gelu(self.linear1(q_norm2))))
        query = query + self.dropout(ffn_output)
        mask_predictor = nn.Linear(d_model, H_mem * W_mem).to(query.device)
        q_for_mask = query.permute(1, 0, 2)
        mask_logits = mask_predictor(q_for_mask)
        mask_logits = mask_logits.view(B, num_queries, H_mem, W_mem)
        pred_mask = torch.sigmoid(mask_logits)
        return query, pred_mask


class MaskedTransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead):
        """
        Stacks decoder layers. At each layer, the cross-attention is computed using one of the 
        multi-scale memory features, selected in a round-robin fashion.
        """
        super(MaskedTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            MaskedTransformerDecoderLayer(d_model, nhead)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, queries, multi_scale_features):
        intermediate_masks = []
        prev_mask = None
        for i, layer in enumerate(self.layers):
            level_idx = i % len(multi_scale_features)  # Round-robin over multi-scale features
            mem = multi_scale_features[level_idx]
            B, C, H_mem, W_mem = mem.shape
            memory = mem.flatten(2).permute(2, 0, 1)  # (H_mem * W_mem, B, d_model)
            queries, pred_mask = layer(queries, memory, (H_mem, W_mem), prev_mask)
            intermediate_masks.append(pred_mask)
            prev_mask = pred_mask
        return queries, intermediate_masks


class Mask2FormerHead(nn.Module):
    def __init__(self, input_channels, num_classes,
                 d_model=256, feature_project_dim=256,
                 num_queries=100, nhead=8, rounds=1):
        """
        Args:
            input_channels (list): Backbone channel dimensions in reverse order [P5, P4, P3, ...].
            rounds (int): The number of rounds (cycles). The total number of decoder layers is rounds * len(input_channels).
        """
        super(Mask2FormerHead, self).__init__()
        print(f"[Mask2FormerHead] input_channels={input_channels}, num_classes={num_classes}, "
              f"d_model={d_model}, feature_project_dim={feature_project_dim}, "
              f"num_queries={num_queries}, nhead={nhead}, rounds={rounds}")
        self.num_decoder_layers = rounds * len(input_channels)
        self.feature_project_dim = feature_project_dim

        self.pixel_decoder = PixelDecoder(input_channels, feature_project_dim, num_heads=nhead)

        # Project features to d_model if needed.
        if feature_project_dim != d_model:
            self.late_proj = nn.Conv2d(feature_project_dim, d_model, kernel_size=1)
        else:
            self.late_proj = nn.Identity()

        self.transformer_decoder = MaskedTransformerDecoder(
            num_layers=self.num_decoder_layers,
            d_model=d_model,
            nhead=nhead,
        )

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_norm = nn.LayerNorm(d_model)
        self.class_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_classes)
        )

        self.dynamic_mask_proj = nn.Linear(d_model, d_model)
    
    def forward(self, features):
        # Fuse features using the pixel decoder.
        fused_feature, multi_scale_features = self.pixel_decoder(features)
        fused_feature = self.late_proj(fused_feature)
        multi_scale_features = [self.late_proj(f) for f in multi_scale_features]
        B, C, H, W = fused_feature.shape

        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (num_queries, B, d_model)
        queries = self.query_norm(queries)
        queries, intermediate_masks = self.transformer_decoder(queries, multi_scale_features)

        cls_logits = self.class_head(queries.transpose(0,1))  # (B, num_queries, num_classes)
        dynamic_kernel = self.dynamic_mask_proj(queries.transpose(0,1))  # (B, num_queries, d_model)
        fused_flat = fused_feature.view(B, C, -1)  # (B, d_model, H*W)
        mask_logits = torch.einsum("bqd,bdn->bqn", dynamic_kernel, fused_flat)  # (B, num_queries, H*W)
        mask_logits = mask_logits.view(B, -1, H, W)
        return cls_logits, mask_logits, intermediate_masks



# Quick test (optional):
if __name__ == "__main__":
    B = 1
    # Define spatial sizes and corresponding channel dimensions for backbone features.
    spatial_sizes = [(128, 128), (64, 64), (32, 32), (16, 16), (8, 8)]
    channels_list = [128, 256, 512, 1024, 2048]

    config_path = None # "configs/model.yaml"  # Set to a path to load config; or use None for manual test.

    if config_path is None:
        # Manual test: loop over number of feature levels (simulate different backbone outputs).
        for num_feats in range(2, 6):
            print(f"\n--- Testing with {num_feats} feature levels ---")
            features = []
            input_channels = []
            for i in range(num_feats):
                C = channels_list[i]
                H, W = spatial_sizes[i]
                features.append(torch.randn(B, C, H, W))
                input_channels.append(C)

            # Use manual config: here, 'rounds' is used; for example, rounds=1 means total decoder layers = len(input_channels)
            head = Mask2FormerHead(
                input_channels=input_channels,
                num_classes=8,
                d_model=256,
                feature_project_dim=256,
                num_queries=100,
                nhead=4,
                rounds=1  # You can set rounds=3 to test a deeper decoder
            )

            import time
            start_time = time.time()
            cls_logits, mask_logits, inter_masks = head(features)
            end_time = time.time()  # End the timer
            print(f"Time taken: {end_time - start_time:.4f} seconds")

            print("Class logits shape:      ", cls_logits.shape)      # (B, Q, num_classes)
            print("Mask logits shape:       ", mask_logits.shape)     # (B, Q, H, W)
            print("# Intermediate masks:    ", len(inter_masks))        # Should equal rounds * len(input_channels)
    else:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        input_channels = cfg["head"]["input_channels"]
        features = []
        for i, C in enumerate(input_channels):
            H, W = spatial_sizes[i]
            features.append(torch.randn(B, C, H, W))
        head = Mask2FormerHead(**cfg["head"])
        cls_logits, mask_logits, inter_masks = head(features)
        print("Class logits shape:      ", cls_logits.shape)
        print("Mask logits shape:       ", mask_logits.shape)
        print("# Intermediate masks:    ", len(inter_masks))
        