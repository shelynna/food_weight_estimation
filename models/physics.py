import torch
import torch.nn as nn
import json
import os

class PhysicsModule(nn.Module):
    """
    Combines shape features with density priors and height estimation.
    """
    def __init__(self, density_json_path:str, feature_input_dim:int, hidden=128):
        super().__init__()
        with open(density_json_path,'r') as f:
            self.density_map = json.load(f)
        # Map class index later
        self.height_net = nn.Sequential(
            nn.Linear(feature_input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, shape_features, class_names, pixel_area_cm2):
        """
        shape_features: (B,F)
        class_names: list of str length B
        pixel_area_cm2: (B,)
        Returns: physics_weight (B,), height_estimate (B,)
        """
        h_pred = self.height_net(shape_features).squeeze(1).clamp(0.1, 15.0)
        densities=[]
        mean_heights=[]
        for cn in class_names:
            d = self.density_map.get(cn, self.density_map["other"])
            densities.append(d["density_g_per_cm3"])
            mean_heights.append(d["mean_height_cm"])
        densities = torch.tensor(densities, device=shape_features.device, dtype=shape_features.dtype)
        mean_heights = torch.tensor(mean_heights, device=shape_features.device, dtype=shape_features.dtype)
        # Option to blend predicted height with prior mean
        height = 0.6*h_pred + 0.4*mean_heights
        volume_cm3 = pixel_area_cm2 * height  # approximate prism
        weight_physics = volume_cm3 * densities
        return weight_physics, height, h_pred