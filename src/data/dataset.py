import json, os, cv2, numpy as np, torch
from torch.utils.data import Dataset
from .augmentations import get_train_transforms, get_val_transforms

class FoodInstanceDataset(Dataset):
    def __init__(self, root, split="train", img_size=448, train=True):
        self.root=root; self.split=split
        ann_path=os.path.join(root, split,"annotations.json")
        with open(ann_path,'r') as f:
            self.annotations=json.load(f)
        self.images_dir=os.path.join(root, split,"images")
        self.items=[]
        for img_ann in self.annotations:
            for inst in img_ann["segments_info"]:
                self.items.append({
                    "image_file": img_ann["file_name"],
                    "image_id": img_ann["file_name"],
                    "instance": inst,
                    "width": img_ann["width"],
                    "height": img_ann["height"]
                })
        self.train=train
        self.tf = get_train_transforms(img_size) if train else get_val_transforms(img_size)
        # class mapping
        with open(os.path.join(root,"class_mapping.json")) as f:
            self.class_map=json.load(f)
        self.rev_class = {v:k for k,v in self.class_map.items()}

    def __len__(self): return len(self.items)

    def _decode_rle(self, rle, h, w):
        # simple RLE: counts
        counts = [int(x) for x in rle.split()]
        mask = np.zeros(h*w, dtype=np.uint8)
        idx=0; val=0
        for c in counts:
            if val==1:
                mask[idx:idx+c]=1
            idx += c
            val=1-val
        return mask.reshape(h,w)

    def __getitem__(self, idx):
        it=self.items[idx]
        path=os.path.join(self.images_dir, it["image_file"])
        img=cv2.imread(path); img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h,w = it["height"], it["width"]
        inst=it["instance"]
        if "mask_rle" in inst:
            mask=self._decode_rle(inst["mask_rle"], h, w)
        else:
            # fallback to bbox
            mask=np.zeros((h,w), dtype=np.uint8)
            x,y,bw,bh=inst["bbox"]
            mask[int(y):int(y+bh), int(x):int(x+bw)]=1
        crop=img
        augmented=self.tf(image=crop, masks=[mask])
        img_t=augmented["image"]
        mask_t=augmented["masks"][0].unsqueeze(0)
        class_id=self.class_map.get(inst["category"], -1)
        weight=inst.get("weight_g", float("nan"))
        return {
            "image": img_t,
            "mask": mask_t.float(),
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "weight": torch.tensor(weight if weight==weight else -1.0),
            "image_id": it["image_id"],
            "instance_id": inst["id"]
        }