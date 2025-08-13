import argparse, os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
from data.dataset import FoodDataset, collate_fn, build_class_mapping
from data.transforms import get_train_transforms, get_val_transforms
from cv.classification import build_classifier
from config.config import FullConfig, ensure_dirs
import pandas as pd

def train(cfg: FullConfig):
    device = cfg.train.device
    train_csv = pd.read_csv(os.path.join(cfg.paths.data_root,"train","annotations.csv"))
    class_to_idx = build_class_mapping(train_csv)
    train_ds = FoodDataset(os.path.join(cfg.paths.data_root,"train"), "train",
                           class_to_idx, transforms=get_train_transforms(), seg=False)
    val_ds = FoodDataset(os.path.join(cfg.paths.data_root,"val"), "val",
                         class_to_idx, transforms=get_val_transforms(), seg=False)
    model = build_classifier(cfg.classification.model_name, len(class_to_idx)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.classification.lr)
    best_acc=0
    for epoch in range(cfg.classification.epochs):
        model.train()
        correct=0
        total=0
        for batch in tqdm(DataLoader(train_ds,batch_size=cfg.classification.batch_size,shuffle=True,num_workers=cfg.train.num_workers)):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=cfg.classification.label_smoothing)
            loss.backward()
            opt.step()
            pred = logits.argmax(1)
            correct += (pred==y).sum().item()
            total += y.numel()
        train_acc = correct/total
        model.eval()
        vcorrect=0; vtotal=0
        with torch.no_grad():
            for batch in DataLoader(val_ds,batch_size=cfg.classification.batch_size,num_workers=cfg.train.num_workers):
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                logits = model(x)
                pred = logits.argmax(1)
                vcorrect += (pred==y).sum().item()
                vtotal += y.numel()
        val_acc = vcorrect/vtotal
        print(f"Epoch {epoch} train_acc {train_acc:.3f} val_acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "class_to_idx": class_to_idx
            }, os.path.join(cfg.paths.checkpoint_dir,"classifier_best.pt"))

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