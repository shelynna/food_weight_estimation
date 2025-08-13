import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from einops import rearrange

class FusionBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim)
        )
        self.ln2 = nn.LayerNorm(dim)
    def forward(self, x):
        # x: (B, N, D)
        attn_out,_ = self.attn(x,x,x,need_weights=False)
        x = x + attn_out
        x = self.ln1(x)
        m = self.mlp(x)
        x = self.ln2(x + m)
        return x

class MLLMCore(nn.Module):
    def __init__(self, clip_model_name="openai/clip-vit-large-patch14",
                 fusion_layers=4, fusion_heads=8, fusion_hidden=768,
                 feature_token_dim=512):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        self.visual_dim = self.clip.visual_projection.out_features
        self.feature_proj = nn.Linear(feature_token_dim, self.visual_dim)
        self.fusion_layers = nn.ModuleList([
            FusionBlock(self.visual_dim, heads=fusion_heads, mlp_ratio=4.0)
            for _ in range(fusion_layers)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, pixel_values, feature_tokens):
        """
        pixel_values: CLIP preprocessed images (B,3,H,W)
        feature_tokens: (B, T, Fdim) already numeric-tokenized
        """
        clip_out = self.clip.vision_model(pixel_values=pixel_values)
        # clip_out.pooler_output: (B,D)
        patch_embeds = clip_out.last_hidden_state  # (B, P+1, D)
        ft = self.feature_proj(feature_tokens)  # (B,T,D)
        x = torch.cat([patch_embeds, ft], dim=1)
        for blk in self.fusion_layers:
            x = blk(x)
        fused = x[:,0]  # CLS
        return fused  # (B,D)