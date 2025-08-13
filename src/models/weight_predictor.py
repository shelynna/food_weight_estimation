import torch, torch.nn as nn

class FusionWeightPredictor(nn.Module):
    def __init__(self, feature_dim, fusion_dim, hidden_dims=(512,256), dropout=0.1):
        super().__init__()
        layers=[]
        in_dim = feature_dim + fusion_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(in_dim, hd))
            layers.append(nn.LayerNorm(hd))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hd
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, features, vision_embed):
        x = torch.cat([features, vision_embed], dim=1)
        return self.net(x).squeeze(1)