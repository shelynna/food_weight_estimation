import argparse, os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from data.dataset import FoodDataset
from data.transforms import get_val_transforms
from cv.segmentation import build_segmentation_model
from cv.classification import build_classifier
from training.train_weight import extract_cv_features
from features.tokenizer import NumericFeatureTokenizer, AdaptiveAlpha
from models.mllm_core import MLLMCore
from models.physics import PhysicsModule
from models.weight_head import WeightHead
import torch.nn.functional as TF
import json

def load_models(cfg, ckpt_dir, class_to_idx):
    device = cfg.train.device
    seg_model = build_segmentation_model(cfg.segmentation.model_name,3,cfg.segmentation.num_classes).to(device)
    seg_model.load_state_dict(torch.load(os.path.join(ckpt_dir,"segmentation_best.pt"), map_location=device))
    seg_model.eval()
    classifier_ckpt = torch.load(os.path.join(ckpt_dir,"classifier_best.pt"), map_location=device)
    class_model = build_classifier(cfg.classification.model_name, len(class_to_idx)).to(device)
    class_model.load_state_dict(classifier_ckpt["state_dict"])
    class_model.eval()
    weight_ckpt = torch.load(os.path.join(ckpt_dir,"weight_model_best.pt"), map_location=device)
    tokenizer = NumericFeatureTokenizer(22, cfg.weight.feature_dim, num_tokens=4).to(device)
    tokenizer.load_state_dict(weight_ckpt["tokenizer"])
    tokenizer.eval()
    mllm = MLLMCore(cfg.weight.clip_model, cfg.weight.fusion_layers, cfg.weight.fusion_heads,
                    cfg.weight.fusion_hidden, cfg.weight.feature_dim).to(device)
    mllm.load_state_dict(weight_ckpt["mllm"])
    mllm.eval()
    physics = PhysicsModule("data/density_priors.json", feature_input_dim=22).to(device)
    physics.load_state_dict(weight_ckpt["physics"])
    physics.eval()
    weight_head = WeightHead(mllm.visual_dim).to(device)
    weight_head.load_state_dict(weight_ckpt["weight_head"])
    weight_head.eval()
    alpha = AdaptiveAlpha()
    alpha.load_state_dict(weight_ckpt["alpha"])
    alpha.to(device)
    alpha.eval()
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    return seg_model, class_model, tokenizer, mllm, physics, weight_head, alpha, idx_to_class

def main(args):
    from config.config import FullConfig
    cfg = FullConfig()
    cfg.paths.data_root = args.data_root
    cfg.paths.output_dir = os.path.dirname(args.out_csv)
    import pandas as pd
    test_csv = pd.read_csv(os.path.join(args.data_root,"annotations.csv"))
    class_to_idx = {}
    # attempt to load from classifier ckpt
    ckpt_classifier = torch.load(os.path.join(args.checkpoint_dir,"classifier_best.pt"), map_location="cpu")
    class_to_idx = ckpt_classifier["class_to_idx"]
    ds = FoodDataset(args.data_root,"test",class_to_idx,transforms=get_val_transforms(),
                     seg=False, return_meta=True)
    dl = DataLoader(ds,batch_size=cfg.weight.batch_size,shuffle=False)
    seg_model, class_model, tokenizer, mllm, physics, weight_head, alpha, idx_to_class = load_models(cfg, args.checkpoint_dir, class_to_idx)
    device = cfg.train.device
    preds=[]
    metas=[]
    with torch.no_grad():
        for batch in dl:
            imgs = batch["image"].to(device)
            clip_imgs = TF.interpolate(imgs, size=(224,224), mode="bilinear", align_corners=False)
            feats_numeric, class_names, area_cm2 = extract_cv_features(batch, seg_model, class_model,
                                                                       class_to_idx, idx_to_class, device)
            feature_tokens = tokenizer(feats_numeric)
            fused = mllm(clip_imgs, feature_tokens)
            learned_weight = weight_head(fused)
            physics_weight, _, _ = physics(feats_numeric, class_names, area_cm2)
            a = alpha()
            pred_weight = a*learned_weight + (1-a)*physics_weight
            for i in range(pred_weight.size(0)):
                meta = batch["meta"][i]
                preds.append({
                    "image_id": meta["image_id"],
                    "item_id": meta["item_id"],
                    "group_id": meta["group_id"],
                    "pred_weight_g": float(pred_weight[i].cpu()),
                    "class_name": class_names[i]
                })
                metas.append(meta)
    df = pd.DataFrame(preds)
    # aggregate
    agg = df.groupby("group_id")["pred_weight_g"].sum().reset_index().rename(columns={"pred_weight_g":"group_total_weight_g"})
    out = df.merge(agg, on="group_id", how="left")
    out.to_csv(args.out_csv, index=False)
    print("Saved predictions to", args.out_csv)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()
    main(args)