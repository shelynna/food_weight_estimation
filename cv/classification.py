import torch
import torch.nn as nn
import timm

class FoodClassifier(nn.Module):
    def __init__(self, model_name="efficientnet_b0", num_classes=20, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool='avg')
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)
    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

def build_classifier(model_name, num_classes):
    return FoodClassifier(model_name, num_classes)