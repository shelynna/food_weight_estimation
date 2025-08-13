import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Optional, Dict, Any

class FoodDataset(Dataset):
    """
    Provides image, segmentation mask (optional), classification label, ground-truth weight.
    annotations.csv columns: image_id,class_name,weight_g,(optional)height_cm,item_id,group_id
    """
    def __init__(self, root: str, split: str, class_to_idx: Dict[str,int],
                 transforms=None, seg: bool=True, mask_dir="masks", img_dir="images",
                 return_meta=False):
        self.root = root
        self.split = split
        self.transforms = transforms
        self.seg = seg
        self.return_meta = return_meta
        csv_path = os.path.join(root, "annotations.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing annotations at {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.class_to_idx = class_to_idx
        self.img_dir = os.path.join(root, img_dir)
        self.mask_dir = os.path.join(root, mask_dir)
        self.samples = self.df.to_dict("records")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        rec = self.samples[idx]
        image_id = rec["image_id"]
        img_path = os.path.join(self.img_dir, image_id)
        if not os.path.exists(img_path):
            # assume add .jpg
            if os.path.exists(img_path + ".jpg"):
                img_path += ".jpg"
            elif os.path.exists(img_path + ".png"):
                img_path += ".png"
            else:
                raise FileNotFoundError(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = None
        if self.seg:
            mpath = os.path.join(self.mask_dir, os.path.splitext(os.path.basename(img_path))[0] + ".png")
            if os.path.exists(mpath):
                mask = cv2.imread(mpath, 0)
                mask = (mask > 0).astype("uint8")
            else:
                mask = np.zeros(img.shape[:2], dtype="uint8")
        label = self.class_to_idx.get(rec["class_name"], self.class_to_idx["other"])
        weight = float(rec["weight_g"])
        height_cm = float(rec["height_cm"]) if "height_cm" in rec and not np.isnan(rec["height_cm"]) else -1.0
        item_id = rec.get("item_id", image_id)
        group_id = rec.get("group_id", image_id)

        aug = self.transforms(image=img, mask=mask) if self.transforms else {"image":img, "mask":mask}
        img_t = aug["image"]
        mask_t = torch.tensor(aug["mask"]).long() if mask is not None else torch.zeros((img.shape[0], img.shape[1])).long()
        sample = {
            "image": img_t,
            "mask": mask_t,
            "label": torch.tensor(label).long(),
            "weight_g": torch.tensor(weight).float(),
            "height_cm": torch.tensor(height_cm).float(),
        }
        if self.return_meta:
            sample["meta"] = {"image_id": image_id, "item_id": item_id, "group_id": group_id, "class_name": rec["class_name"]}
        return sample

def build_class_mapping(train_csv):
    classes = sorted(train_csv["class_name"].unique().tolist())
    if "other" not in classes:
        classes.append("other")
    return {c:i for i,c in enumerate(classes)}

def collate_fn(batch):
    out = {k: [] for k in batch[0].keys()}
    for b in batch:
        for k,v in b.items():
            out[k].append(v)
    out["image"] = torch.stack(out["image"])
    out["mask"] = torch.stack(out["mask"])
    out["label"] = torch.stack(out["label"])
    out["weight_g"] = torch.stack(out["weight_g"])
    out["height_cm"] = torch.stack(out["height_cm"])
    if "meta" in out:
        pass
    return out