import torch
import torch.nn as nn
from typing import List

class NumericFeatureTokenizer(nn.Module):
    """
    Converts continuous numeric feature vector into N tokens via learned projection.
    """
    def __init__(self, in_dim: int, token_dim: int, num_tokens: int=4):
        super().__init__()
        self.num_tokens = num_tokens
        self.proj = nn.Sequential(
            nn.Linear(in_dim, token_dim * num_tokens),
            nn.ReLU(),
            nn.Linear(token_dim * num_tokens, token_dim * num_tokens)
        )
        self.token_dim = token_dim
    def forward(self, x):
        # x: (B, F)
        out = self.proj(x)
        return out.view(x.shape[0], self.num_tokens, self.token_dim)  # (B, T, D)

class AdaptiveAlpha(nn.Module):
    def __init__(self, init_alpha=0.5):
        super().__init__()
        self.logit = nn.Parameter(torch.log(torch.tensor(init_alpha/(1-init_alpha))))
    def forward(self):
        return torch.sigmoid(self.logit)