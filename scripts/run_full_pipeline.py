import os, glob, argparse, json, cv2, torch, csv
from PIL import Image
from tqdm import tqdm
from src.cv.segmentation import Segmenter
from src.cv.area_shape import compute_all_features
from src.cv.feature_tokenizer import NumericFeatureTokenizer
from src.cv.classification import build_classifier
from src.mllm.llava_wrapper import MLLMWrapper
from src.models.weight_predictor import FusionWeightPredictor
from src.mllm.feature_adapter import FeatureAdapter
from src.physics.reasoner import physics_weight
from src.utils.config import load_config
from src.utils.logging import console
from src.training.train_weight_model import FEATURE_KEYS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--images", required=True)
    args=ap.parse_args()
    cfg=load_config(args.config)
    device=cfg["device"]
    seg=Segmenter(device=device)
    classifier=build_classifier(cfg).to(device)
    classifier.load_state_dict(torch.load("checkpoints/food_classifier.pt", map_location=device))
    classifier.eval()
    mllm=MLLMWrapper(cfg["mllm"]["name"], device=device, max_new_tokens=cfg["mllm"]["max_new_tokens"])
    tokenizer=NumericFeatureTokenizer(FEATURE_KEYS)
    ckpt=torch.load("checkpoints/weight_predictor.pt", map_location=device)
    adapter=None
    if cfg["model"]["weight_predictor"]["use_feature_adapter"]:
        feature_dim=len(FEATURE_KEYS)+cfg["model"]["classifier"]["num_classes"]+3
        adapter=FeatureAdapter(feature_dim,768).to(device)
        adapter.load_state_dict(ckpt["adapter"])
        adapter.eval()
        pred_model=FusionWeightPredictor(768,768,
                                         hidden_dims=cfg["model"]["weight_predictor"]["hidden_dims"],
                                         dropout=cfg["model"]["weight_predictor"]["dropout"]).to(device)
    else:
        pred_model=FusionWeightPredictor(len(FEATURE_KEYS)+cfg["model"]["classifier"]["num_classes"]+3,
                                         768,
                                         hidden_dims=cfg["model"]["weight_predictor"]["hidden_dims"],
                                         dropout=cfg["model"]["weight_predictor"]["dropout"]).to(device)
    pred_model.load_state_dict(ckpt["predictor"]); pred_model.eval()

    os.makedirs("outputs", exist_ok=True)
    outfile="outputs/predictions.csv"
    f=open(outfile,"w", newline="")
    writer=csv.writer(f)
    writer.writerow(["image_id","instance_idx","pred_weight","class_pred","physics_weight"])
    imgs=glob.glob(os.path.join(args.images,"*.jpg"))+glob.glob(os.path.join(args.images,"*.png"))
    for img_path in tqdm(imgs):
        image_bgr=cv2.imread(img_path)
        results=seg.predict(image_bgr)
        if not results: continue
        img_rgb=cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        plate_diam_cm=25; plate_area_cm2=3.1416*(plate_diam_cm/2)**2
        h,w=img_rgb.shape[:2]; px_per_cm=max(h,w)/plate_diam_cm
        pil_img=Image.open(img_path).convert("RGB")
        for i,res in enumerate(results):
            feats=compute_all_features(img_rgb, res["mask"], px_per_cm, plate_area_cm2)
            feats_vec=tokenizer.dict_to_tensor(feats, device=device).unsqueeze(0)
            # classification on full image (simplified, ideally crop)
            with torch.no_grad():
                logits=classifier(torch.nn.functional.interpolate(
                    torch.tensor(img_rgb).permute(2,0,1).unsqueeze(0).float()/255.,
                    size=(cfg["data"]["img_size"], cfg["data"]["img_size"])
                ).to(device))
                probs=torch.softmax(logits,1)
                class_pred=probs.argmax(1).item()
            physics_w,_=physics_weight(feats["area_cm2"], list(load_json(os.path.join(cfg["data"]["root"],"class_mapping.json")).keys())[class_pred])
            phx_vec=torch.tensor([[physics_w,1.0,1.0]], device=device)
            full_features=torch.cat([feats_vec, probs, phx_vec],1)
            if adapter:
                full_features=adapter(full_features)
            vision_embed=mllm.image_embedding(pil_img)
            with torch.no_grad():
                pred_w=pred_model(full_features, vision_embed).item()
            writer.writerow([os.path.basename(img_path), i, pred_w, class_pred, physics_w])
    f.close()
    console.log(f"Saved predictions to {outfile}")

def load_json(p): 
    with open(p,'r') as f: return json.load(f)

if __name__=="__main__":
    main()