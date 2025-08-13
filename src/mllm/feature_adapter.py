import torch, torch.nn as nn

class FeatureAdapter(nn.Module):
    """
    Projects numeric feature vector to same dimension as MLLM vision embeddings.
    Can be appended (concatenated) then fused via linear layer.
    """
    def __init__(self, in_dim, target_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, target_dim)
        )
    def forward(self, x):
        return self.net(x)