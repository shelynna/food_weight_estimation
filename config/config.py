import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class Paths:
    data_root: str = "/content/data"
    output_dir: str = "/content/output"
    checkpoint_dir: str = "/content/output/checkpoints"
    log_dir: str = "/content/output/logs"

@dataclass
class SegmentationConfig:
    model_name: str = "unet"  # or deeplabv3
    in_channels: int = 3
    num_classes: int = 2
    lr: float = 1e-3
    epochs: int = 40
    batch_size: int = 8
    dice_weight: float = 0.7
    bce_weight: float = 0.3

@dataclass
class ClassificationConfig:
    model_name: str = "efficientnet_b0"
    num_classes: int = 20
    lr: float = 3e-4
    epochs: int = 30
    batch_size: int = 32
    label_smoothing: float = 0.05

@dataclass
class WeightModelConfig:
    clip_model: str = "openai/clip-vit-large-patch14"
    feature_dim: int = 512
    numeric_feature_dim: int = 64
    fusion_layers: int = 4
    fusion_heads: int = 8
    fusion_hidden: int = 768
    lr: float = 2e-4
    epochs: int = 60
    batch_size: int = 16
    alpha_init: float = 0.6  # initial weighting for learned vs physics
    reg_weight: float = 1.0
    physics_consistency_weight: float = 0.3
    height_reg_weight: float = 0.2

@dataclass
class TrainingConfig:
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    amp: bool = True

@dataclass
class FullConfig:
    paths: Paths = Paths()
    segmentation: SegmentationConfig = SegmentationConfig()
    classification: ClassificationConfig = ClassificationConfig()
    weight: WeightModelConfig = WeightModelConfig()
    train: TrainingConfig = TrainingConfig()

def ensure_dirs(cfg: FullConfig):
    for d in [cfg.paths.output_dir, cfg.paths.checkpoint_dir, cfg.paths.log_dir]:
        os.makedirs(d, exist_ok=True)