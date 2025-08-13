# Improving MLLM-Based Food Weight Estimation Through Targeted Visual Feature Augmentation

This project implements an end-to-end pipeline:

1. CV Frontend
   - Segmentation (instance masks)
   - Area calculation + pixel-to-real scale estimation
   - Shape descriptors (circularity, elongation, convexity, Hu moments, etc.)
   - Food classification (category + density prior)
   - Feature Tokenization (numeric -> textual or learned embedding tokens)

2. MLLM Core
   - Loads a small open-source Multimodal LLM (default: LLaVA 1.5 7B or Qwen2-VL 2B for Colab)
   - Injects structured features via:
     a. Prompt-level textual augmentation ("FEATURES: area=..., circularity=...")
     b. Optional learned Feature Adapter projecting numeric features into the model's embedding space as pseudo-image tokens

3. Physics-Constrained Reasoning
   - Uses priors on density and plausible thickness heuristics
   - Computes a constrained volume estimate
   - Combines MLLM semantic embedding + CV numeric features via fusion network
   - Outputs final per-item weights and aggregated plate weight

4. Evaluation
   - Weight prediction metrics: MAE, RMSE, MAPE
   - Classification accuracy
   - Ablations for: no features, +shape, +physics, +adapter

## Quick Start (Colab)

```bash
!git clone <your_fork_url> food-weight-estimation
%cd food-weight-estimation
!pip install -r requirements.txt
```

Prepare data (expects images + annotations in a simple JSON or COCO-like format):
```bash
python scripts/prepare_data.py --raw_dir /content/drive/MyDrive/ghana --out_dir data/processed
```

Train classifier:
```bash
python -m src.training.train_classifier --config configs/base.yaml
```

Run full pipeline (segmentation->features->MLLM weight estimation):
```bash
python scripts/run_full_pipeline.py --config configs/base.yaml --images data/processed/val/images
```

## Data Assumptions

Directory layout after preparation:
```
data/processed/
  train/
    images/*.jpg
    annotations.json
  val/
    images/*.jpg
    annotations.json
  densities.json        # mapping food_label -> density (g/cm^3)
  class_mapping.json
```

Each annotation entry (COCO-like):
```json
{
 "file_name": "img_001.jpg",
 "height": 1024,
 "width": 1024,
 "segments_info": [
   {
     "id": 1,
     "bbox": [x,y,w,h],
     "category": "plantain",
     "mask_rle": "...",
     "weight_g": 120.0    // optional ground truth
   }
 ]
}
```

If no per-instance weight, but total plate weight known, you can still train fusion model with aggregated constraints later.

## Feature List

- Pixel area, convex area, area_ratio
- Contour perimeter
- Circularity, solidity, extent, aspect_ratio
- Major/minor axis (ellipse fit)
- Hu moments (first 4)
- Mean/Std color (RGB or Lab)
- Relative plate coverage
- Classification logits / probability vector

## Physics Constraints

Estimated volume = (projected_area * estimated_height)
estimated_height derived from:
- Food class thickness prior in priors.py
- Shape heuristic (elongation vs thickness)
Weight = volume * density_prior(class)
Clamp & adjust with MLLM semantic cues (e.g., "sliced", "mashed" -> modifies thickness factor)

## MLLM Integration Modes

1. Prompt Augmentation (zero extra training):
   "Describe the portion weights. FEATURES: item_1: class=plantain, area=34.2cm2, circularity=0.71, thickness=1.8cm ..."

2. Feature Adapter:
   - Numeric feature vector -> Linear/MLP -> embedding dimension -> appended to vision tokens -> forward pass -> pooled
   - Trained with lightweight LoRA on MLLM (optional; disabled by default for Colab memory limits).

## Evaluation

Outputs a CSV:
```
image_id, instance_id, true_weight, pred_weight, class_true, class_pred, abs_error_g
```

And prints aggregated metrics.

## Reproducibility

All hyperparameters in configs/base.yaml. Set random seeds in seed.py.

## Extending

- Swap segmentation with SAM fine-tuning
- Add depth estimation to refine height
- Introduce mixture density networks for weight uncertainty

See detailed run instructions in the bottom of this README after cloning.

## License
Choose an appropriate license (MIT suggested) before release.