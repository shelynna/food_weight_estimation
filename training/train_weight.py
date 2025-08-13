import argparse, os, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

from data.dataset import FoodDataset, build_class_mapping
from data.transforms import get_val_transforms
from cv.segmentation import build_segmentation_model
from cv.classification import build_classifier
from cv.area import compute_pixel_area, estimate_pixel_to_cm, pixel_area_to_cm2
from cv.shape import shape_descriptors, descriptor_vector
from features.tokenizer import NumericFeatureTokenizer, AdaptiveAlpha
from models.mllm_core import MLLMCore
from models.physics import PhysicsModule
from models.weight_head import WeightHead
from config.config import FullConfig, ensure_dirs

import cv2

def extract_cv_features(batch, seg_model, class_model, class_to_idx, idx_to_class, device):
    """
    Returns numeric feature tensor (B,F), class names list, pixel_area_cm2 (B,)
    """
    images = batch["image"]
    bsz = images.size(0)
    images_np = (images.permute(0,2,3,1).cpu().numpy()*255).astype("uint8")
    with torch.no_grad():
        # segmentation
        seg_logits = seg_model(images.to(device))
        if isinstance(seg_logits, dict): seg_logits = seg_logits["out"]
        seg_mask = torch.argmax(seg_logits, dim=1).cpu().numpy().astype("uint8")
        # classification
        logits = class_model(images.to(device))
        pred_cls_idx = torch.argmax(logits,1).cpu().numpy()
    feats=[]
    class_names=[]
    pixel_area_cm2=[]
    for i in range(bsz):
        img = images_np[i]
        mask = seg_mask[i]
        desc = shape_descriptors(mask, img)
        vec, keys = descriptor_vector(desc)
        # scale estimation
        pixel_to_cm = estimate_pixel_to_cm(mask, img)
        area_cm2 = pixel_area_to_cm2(desc["area_px"], pixel_to_cm)
        vec.append(area_cm2)
        feats.append(vec)
        cn = idx_to_class[pred_cls_idx[i]]
        class_names.append(cn)
        pixel_area_cm2.append(area_cm2)
    feats = torch.tensor(feats, device=device).float()
    pixel_area_cm2 = torch.tensor(pixel_area_cm2, device=device).float()
    return feats, class_names, pixel_area_cm2

def build_models(cfg: FullConfig, density_json_path, class_to_idx):
    device = cfg.train.device
    seg_model = build_segmentation_model(cfg.segmentation.model_name,3,cfg.segmentation.num_classes).to(device)
    class_model = build_classifier(cfg.classification.model_name, len(class_to_idx)).to(device)
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    # numeric feature dimension:
    # shape_descriptors 21 + area_cm2 appended = 22
    numeric_dim = 22
    tokenizer = NumericFeatureTokenizer(numeric_dim, cfg.weight.feature_dim, num_tokens=4).to(device)
    mllm = MLLMCore(cfg.weight.clip_model, cfg.weight.fusion_layers, cfg.weight.fusion_heads,
                    cfg.weight.fusion_hidden, cfg.weight.feature_dim).to(device)
    physics = PhysicsModule(density_json_path, feature_input_dim=numeric_dim).to(device)
    weight_head = WeightHead(mllm.visual_dim).to(device)
    alpha = AdaptiveAlpha(cfg.weight.alpha_init).to(device)
    return seg_model, class_model, tokenizer, mllm, physics, weight_head, alpha, idx_to_class

def train(cfg: FullConfig):
    device = cfg.train.device
    train_csv = pd.read_csv(os.path.join(cfg.paths.data_root,"train","annotations.csv"))
    class_to_idx = build_class_mapping(train_csv)
    val_csv = pd.read_csv(os.path.join(cfg.paths.data_root,"val","annotations.csv"))
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    train_ds = FoodDataset(os.path.join(cfg.paths.data_root,"train"),"train",
                           class_to_idx, transforms=get_val_transforms(), seg=False, return_meta=True)
    val_ds = FoodDataset(os.path.join(cfg.paths.data_root,"val"),"val",
                         class_to_idx, transforms=get_val_transforms(), seg=False, return_meta=True)
    density_json_path = "data/density_priors.json"
    seg_model, class_model, tokenizer, mllm, physics, weight_head, alpha_module, idx_to_class = build_models(cfg, density_json_path, class_to_idx)
    # load pretrained segmentation & classification
    seg_ckpt = os.path.join(cfg.paths.checkpoint_dir,"segmentation_best.pt")
    cls_ckpt = os.path.join(cfg.paths.checkpoint_dir,"classifier_best.pt")
    if os.path.exists(seg_ckpt):
        seg_model.load_state_dict(torch.load(seg_ckpt,map_location=device))
    if os.path.exists(cls_ckpt):
        cc = torch.load(cls_ckpt,map_location=device)
        class_model.load_state_dict(cc["state_dict"])
    seg_model.eval()
    class_model.eval()
    opt = torch.optim.AdamW(list(tokenizer.parameters()) + list(mllm.parameters())
                            + list(physics.parameters()) + list(weight_head.parameters())
                            + list(alpha_module.parameters()), lr=cfg.weight.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp)
    best=1e9
    processor_required_size = 224  # CLIP expects 224
    # simple resize using torchvision transforms inside loop to avoid extra dependency
    import torch.nn.functional as TF
    for epoch in range(cfg.weight.epochs):
        mllm.train(); tokenizer.train(); physics.train(); weight_head.train()
        epoch_loss=0
        for batch in tqdm(DataLoader(train_ds,batch_size=cfg.weight.batch_size,shuffle=True,num_workers=cfg.train.num_workers)):
            imgs = batch["image"].to(device)
            # CLIP interpolation
            clip_imgs = TF.interpolate(imgs, size=(224,224), mode="bilinear", align_corners=False)
            target_weight = batch["weight_g"].to(device)
            with torch.no_grad():
                feats_numeric, class_names, area_cm2 = extract_cv_features(batch, seg_model, class_model,
                                                                           class_to_idx, idx_to_class, device)
            feature_tokens = tokenizer(feats_numeric)
            with torch.cuda.amp.autocast(enabled=cfg.train.amp):
                fused = mllm(clip_imgs, feature_tokens)
                learned_weight = weight_head(fused).clamp(0,5000)
                physics_weight, height, h_raw = physics(feats_numeric, class_names, area_cm2)
                a = alpha_module()
                pred_weight = a*learned_weight + (1-a)*physics_weight
                loss_reg = F.smooth_l1_loss(pred_weight, target_weight, beta=10.0)
                loss_phys_consistency = F.smooth_l1_loss(learned_weight, physics_weight.detach(), beta=10.0)
                # if ground-truth height present
                gt_height = batch["height_cm"].to(device)
                valid_height = gt_height>0
                if valid_height.any():
                    loss_height = F.smooth_l1_loss(height[valid_height], gt_height[valid_height], beta=2.0)
                else:
                    loss_height = torch.tensor(0.0, device=device)
                loss = cfg.weight.reg_weight*loss_reg + cfg.weight.physics_consistency_weight*loss_phys_consistency + cfg.weight.height_reg_weight*loss_height
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            epoch_loss += loss.item()*imgs.size(0)
        train_loss = epoch_loss/len(train_ds)
        # validation
        mllm.eval(); tokenizer.eval(); physics.eval(); weight_head.eval()
        vloss=0; preds=[]; gts=[]
        with torch.no_grad():
            for batch in DataLoader(val_ds,batch_size=cfg.weight.batch_size,num_workers=cfg.train.num_workers):
                imgs = batch["image"].to(device)
                clip_imgs = TF.interpolate(imgs, size=(224,224), mode="bilinear", align_corners=False)
                target_weight = batch["weight_g"].to(device)
                feats_numeric, class_names, area_cm2 = extract_cv_features(batch, seg_model, class_model,
                                                                           class_to_idx, idx_to_class, device)
                feature_tokens = tokenizer(feats_numeric)
                fused = mllm(clip_imgs, feature_tokens)
                learned_weight = weight_head(fused).clamp(0,5000)
                physics_weight, height, _ = physics(feats_numeric, class_names, area_cm2)
                a = alpha_module()
                pred_weight = a*learned_weight + (1-a)*physics_weight
                loss = F.l1_loss(pred_weight, target_weight)
                vloss += loss.item()*imgs.size(0)
                preds.append(pred_weight.cpu())
                gts.append(target_weight.cpu())
        val_mae = vloss/len(val_ds)
        print(f"Epoch {epoch} train_loss {train_loss:.3f} val_MAE {val_mae:.3f} alpha {a.item():.3f}")
        if val_mae < best:
            best = val_mae
            torch.save({
                "tokenizer": tokenizer.state_dict(),
                "mllm": mllm.state_dict(),
                "physics": physics.state_dict(),
                "weight_head": weight_head.state_dict(),
                "alpha": alpha_module.state_dict(),
                "class_to_idx": class_to_idx
            }, os.path.join(cfg.paths.checkpoint_dir,"weight_model_best.pt"))

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    cfg = FullConfig()
    cfg.paths.data_root = args.data_root
    cfg.paths.output_dir = args.output_dir
    ensure_dirs(cfg)
    train(cfg)