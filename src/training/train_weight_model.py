import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from ..data.dataset import FoodInstanceDataset
from ..cv.classification import build_classifier
from ..cv.feature_tokenizer import NumericFeatureTokenizer
from ..models.weight_predictor import FusionWeightPredictor
from ..mllm.llava_wrapper import MLLMWrapper
from ..mllm.feature_adapter import FeatureAdapter
from ..physics.reasoner import physics_weight
from ..cv.area_shape import compute_all_features
from ..utils.seed import set_seed
from ..utils.metrics import regression_metrics
from ..utils.config import load_config
from ..utils.logging import console
import torchvision.transforms.functional as TF
import numpy as np, argparse, os, json, cv2

FEATURE_KEYS=["area_cm2","circularity","solidity","extent","aspect_ratio",
              "hu1","hu2","hu3","hu4","relative_plate_coverage",
              "mean_r","mean_g","mean_b","std_r","std_g","std_b"]

def crop_from_mask(image, mask):
    ys, xs = (mask>0).nonzero()
    if len(xs)==0: return image
    x1,x2=xs.min(), xs.max()
    y1,y2=ys.min(), ys.max()
    return image[y1:y2+1, x1:x2+1]

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
    train_loader=DataLoader(train_ds, batch_size=1, shuffle=True)  # batch=1 for per-instance feature building
    val_loader=DataLoader(val_ds, batch_size=1, shuffle=False)

    classifier = build_classifier(cfg).to(device)
    classifier.load_state_dict(torch.load("checkpoints/food_classifier.pt", map_location=device))
    classifier.eval()

    mllm = MLLMWrapper(cfg["mllm"]["name"], device=device, max_new_tokens=cfg["mllm"]["max_new_tokens"])
    tokenizer = NumericFeatureTokenizer(FEATURE_KEYS)

    # Estimate feature dimension
    feature_dim=len(FEATURE_KEYS)+cfg["model"]["classifier"]["num_classes"]+3  # +3 physics derived scalars
    fusion_dim=768  # LLaVA hidden size
    adapter=None
    if cfg["model"]["weight_predictor"]["use_feature_adapter"]:
        adapter=FeatureAdapter(feature_dim, fusion_dim).to(device)

    predictor=FusionWeightPredictor(
        feature_dim=feature_dim if adapter is None else fusion_dim,
        fusion_dim=fusion_dim,
        hidden_dims=cfg["model"]["weight_predictor"]["hidden_dims"],
        dropout=cfg["model"]["weight_predictor"]["dropout"]
    ).to(device)

    opt=optim.AdamW(
        list(predictor.parameters()) + ([] if adapter is None else list(adapter.parameters())),
        lr=cfg["model"]["weight_predictor"]["lr"]
    )
    crit=nn.SmoothL1Loss()

    def process_batch(batch, train=True):
        image=batch["image"][0].permute(1,2,0).cpu().numpy()
        mask=batch["mask"][0,0].cpu().numpy()>0.5
        class_id=batch["class_id"].item()
        # Convert CHW back to PIL for MLLM
        image_pil = TF.to_pil_image(batch["image"][0])
        # Derive scale heuristics: assume plate diameter of 25cm if not provided
        plate_diam_cm=25
        plate_area_cm2=3.1416*(plate_diam_cm/2)**2
        # naive px per cm using inscribed square assumption
        h,w=image.shape[:2]
        px_per_cm = max(h,w)/plate_diam_cm
        feats = compute_all_features(image, mask.astype('uint8'), px_per_cm, plate_area_cm2)
        feats_vec = tokenizer.dict_to_tensor(feats, device=device).unsqueeze(0)
        # classifier logits
        with torch.no_grad():
            logits = classifier(batch["image"].to(device))
            probs = torch.softmax(logits,1)
        physics_w, phx = physics_weight(feats["area_cm2"], list(train_ds.class_map.keys())[class_id])
        phx_vec = torch.tensor([physics_w, phx["height_cm"], phx["density"]], device=device).unsqueeze(0)
        full_features = torch.cat([feats_vec, probs, phx_vec], dim=1)

        # MLLM embedding
        feature_text = tokenizer.to_text(feats)
        prompt = f"Analyze the food item and consider features: {feature_text}. Provide any preparation adjectives."
        mllm_text = mllm.generate_with_image(image_pil, prompt)[:512]
        vision_embed = mllm.image_embedding(image_pil)  # [1,768]

        if adapter is not None:
            full_features = adapter(full_features)

        pred = predictor(full_features, vision_embed)
        true_w = batch["weight"].item()
        # skip if no ground truth
        loss=None
        if true_w>=0:
            target=torch.tensor([true_w], device=device, dtype=torch.float32)
            loss=crit(pred, target)
        return pred, true_w, loss

    epochs=cfg["model"]["weight_predictor"]["epochs"]
    for ep in range(1, epochs+1):
        predictor.train(); 
        if adapter: adapter.train()
        train_losses=[]
        for batch in train_loader:
            pred,true_w,loss=process_batch(batch, train=True)
            if loss is None: continue
            opt.zero_grad(); loss.backward(); opt.step()
            train_losses.append(loss.item())
        predictor.eval(); 
        if adapter: adapter.eval()
        y_true=[]; y_pred=[]
        with torch.no_grad():
            for batch in val_loader:
                pred,true_w,loss=process_batch(batch, train=False)
                if true_w>=0:
                    y_true.append(true_w); y_pred.append(pred.item())
        if y_true:
            metrics=regression_metrics(y_true,y_pred)
            console.log(f"Epoch {ep}: train_loss={np.mean(train_losses):.4f} Val MAE={metrics['MAE']:.2f}g RMSE={metrics['RMSE']:.2f}g")
        else:
            console.log(f"Epoch {ep}: No validation weights available.")
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            "predictor": predictor.state_dict(),
            "adapter": adapter.state_dict() if adapter else None
        }, "checkpoints/weight_predictor.pt")
    console.log("Weight predictor training complete.")

if __name__=="__main__":
    main()