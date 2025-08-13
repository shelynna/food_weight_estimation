import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from ..data.dataset import FoodInstanceDataset
from ..cv.classification import build_classifier
from ..utils.seed import set_seed
from ..utils.logging import console
from ..utils.config import load_config
from ..utils.metrics import classification_accuracy
import argparse, os, json, numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args=ap.parse_args()
    cfg=load_config(args.config)
    set_seed(cfg["seed"])
    device=cfg["device"]

    train_ds=FoodInstanceDataset(cfg["data"]["root"], cfg["data"]["train_split"],
                                 img_size=cfg["data"]["img_size"], train=True)
    val_ds=FoodInstanceDataset(cfg["data"]["root"], cfg["data"]["val_split"],
                               img_size=cfg["data"]["img_size"], train=False)
    train_loader=DataLoader(train_ds, batch_size=cfg["data"]["batch_size"],
                            shuffle=True, num_workers=cfg["data"]["num_workers"])
    val_loader=DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False)

    model=build_classifier(cfg).to(device)
    crit=nn.CrossEntropyLoss()
    opt=optim.AdamW(model.parameters(), lr=cfg["model"]["classifier"]["lr"],
                    weight_decay=cfg["model"]["classifier"]["weight_decay"])
    epochs=cfg["model"]["classifier"]["epochs"]

    best=0
    for ep in range(1, epochs+1):
        model.train()
        losses=[]
        for batch in train_loader:
            x=batch["image"].to(device)
            y=batch["class_id"].to(device)
            logits=model(x)
            loss=crit(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval()
        preds=[]; trues=[]
        with torch.no_grad():
            for batch in val_loader:
                x=batch["image"].to(device)
                y=batch["class_id"].to(device)
                logits=model(x)
                p=logits.argmax(1)
                preds.extend(p.cpu().tolist())
                trues.extend(y.cpu().tolist())
        acc=classification_accuracy(trues,preds)
        console.log(f"Epoch {ep}: loss={np.mean(losses):.4f} val_acc={acc:.4f}")
        if acc>best:
            best=acc
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/food_classifier.pt")
            console.log("Saved best classifier.")
    console.log(f"Training finished. Best Acc={best:.4f}")

if __name__=="__main__":
    main()