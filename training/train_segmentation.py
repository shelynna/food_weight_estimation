import argparse, os, sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from cv.segmentation import build_segmentation_model, dice_loss
from data.dataset import FoodDataset, build_class_mapping, collate_fn
from data.transforms import get_segmentation_train, get_segmentation_val
from config.config.py import FullConfig, ensure_dirs

def train(cfg: FullConfig):
    device = cfg.train.device
    train_ds = FoodDataset(os.path.join(cfg.paths.data_root,"train"), "train",
                           class_to_idx={"other":0}, transforms=get_segmentation_train(), seg=True)
    val_ds = FoodDataset(os.path.join(cfg.paths.data_root,"val"), "val",
                         class_to_idx={"other":0}, transforms=get_segmentation_val(), seg=True)
    model = build_segmentation_model(cfg.segmentation.model_name,
                                     in_channels=3,
                                     num_classes=cfg.segmentation.num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.segmentation.lr)
    best = 1e9
    for epoch in range(cfg.segmentation.epochs):
        model.train()
        tloss=0
        for batch in tqdm(DataLoader(train_ds,batch_size=cfg.segmentation.batch_size,shuffle=True,num_workers=cfg.train.num_workers)):
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            opt.zero_grad()
            out = model(img)
            if isinstance(out, dict): out = out["out"]
            loss_dice = dice_loss(out, mask)
            loss_bce = F.cross_entropy(out, mask)
            loss = cfg.segmentation.dice_weight*loss_dice + cfg.segmentation.bce_weight*loss_bce
            loss.backward()
            opt.step()
            tloss += loss.item()
        model.eval()
        vloss=0
        with torch.no_grad():
            for batch in DataLoader(val_ds,batch_size=cfg.segmentation.batch_size,num_workers=cfg.train.num_workers):
                img = batch["image"].to(device)
                mask = batch["mask"].to(device)
                out = model(img)
                if isinstance(out, dict): out = out["out"]
                loss = dice_loss(out, mask)
                vloss += loss.item()
        print(f"Epoch {epoch}: train {tloss/len(train_ds):.4f} val {vloss/len(val_ds):.4f}")
        if vloss < best:
            best = vloss
            torch.save(model.state_dict(), os.path.join(cfg.paths.checkpoint_dir,"segmentation_best.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    from config.config import FullConfig, ensure_dirs
    cfg = FullConfig()
    cfg.paths.data_root = args.data_root
    cfg.paths.output_dir = args.output_dir
    ensure_dirs(cfg)
    train(cfg)