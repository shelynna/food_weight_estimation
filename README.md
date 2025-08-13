# Improving MLLM-Based Food Weight Estimation Through Targeted Visual Feature Augmentation

End-to-end pipeline:
1. Segmentation (produces masks)
2. Area + shape descriptors from masks
3. Food classification (for density prior + class-specific features)
4. Feature tokenization (numeric → embedding tokens)
5. Image encoder (CLIP ViT-L/14 by default)
6. Fusion + reasoning transformer (augments image embedding with feature tokens)
7. Physics-constrained reasoning: combines learned regression with physics formula (volume ≈ area * height_estimate, weight = volume * density_prior)
8. Multi-item aggregation (plate-level / meal-level weight)

## Dataset Assumptions

Folder layout (example):
```
data/
  train/
    images/
      img_0001.jpg
      ...
    masks/
      img_0001.png        # single-channel mask (0 background, >0 foreground) OR multi-class mask
    annotations.csv        # columns: image_id, class_name, weight_g, (optional) height_cm, item_id, group_id
  val/
    images/...
    masks/...
    annotations.csv
  test/
    images/...
    masks/...
    annotations.csv
```

`group_id` groups multiple items (e.g., meal). `item_id` distinguishes instances on the same plate (if you have per-item masks). If instance-level segmentation not available, item_id can be same as image_id.

If using Ghana dataset, adapt a preprocessing script to export into this structure.

## Quick Start (Colab)

```bash
!git clone <your_repo_url> food_weight_estimation
%cd food_weight_estimation
!pip install -r requirements.txt
```

Then:
```bash
python scripts/run_full_pipeline.py \
  --data_root /content/data \
  --output_dir /content/output \
  --train_all
```

Or open `colab_notebook.py` for a step-by-step interactive run.

## Key Components

- Segmentation: UNet (fast) or `torchvision` DeepLabV3; optional integration of Segment Anything (SAM) stub.
- Classification: EfficientNet-B0 (timm).
- Shape Descriptors:
  - Area (pixels)
  - Perimeter
  - Convex area ratio
  - Eccentricity
  - Aspect ratio
  - Circularity
  - Extent
  - Hu Moments (log transformed)
  - Mean / std / entropy of masked RGB
- Feature Tokenization: Numeric features → MLP → tokens appended to visual embedding (or cross-attended).
- MLLM Core: CLIP visual encoder + lightweight multimodal fusion transformer (custom).
- Physics Module:
  - Density prior (configurable JSON)
  - Height estimation network (regresses plausible height_cm from shape features)
  - Physics weight = area_cm2 * height_cm * density (with scaling from pixel-to-cm via reference estimation or learned scale)
  - Final weight = α * learned_weight + (1-α) * physics_weight (α learned / adaptive per sample)
- Losses: Smooth L1 for weight, classification CE, segmentation Dice + BCE, auxiliary physics consistency loss.
- Evaluation: MAE, MAPE, RMSE, R^2, per-class metrics, confusion matrix for classification, weight calibration plots.

## Extending Density Priors

Edit `data/density_priors.json` with `{"class_name": {"density_g_per_cm3": float, "mean_height_cm": float}}`.

## Multi-Item Aggregation

If multiple items (same group_id) exist: aggregated weight = sum(predicted_item_weights). Optionally apply plate-level correction via linear calibration.

## Training Order

You can:
1. Train segmentation: `python training/train_segmentation.py`
2. Train classification: `python training/train_classifier.py`
3. Train weight model (loads frozen encoders by default): `python training/train_weight.py`

Or `--train_all` in script orchestrates.

## Model Checkpoints

Default: stored in `output/checkpoints/`. Names:
- segmentation_best.pt
- classifier_best.pt
- weight_model_best.pt

## Inference

```bash
python inference/predict.py \
  --data_root /content/data/test \
  --checkpoint_dir /content/output/checkpoints \
  --out_csv /content/output/predictions.csv
```

## Colab Notebook

`colab_notebook.py` contains a linear runnable script to mount drive, install deps, train, evaluate, and visualize.

## Reproducibility

Set seeds in `config/config.py`. Determinism may still vary due to CuDNN nondeterministic ops.

## License

Provided as academic reference. Verify compliance with any external model (CLIP) licenses for redistribution.

## Notes

- If you have a known reference object (e.g., checker card, standard plate diameter), implement pixel-to-cm scale in `cv/area.py` (placeholder included).
- If segmentation masks are not available, enable `--weak_segmentation` to approximate using threshold + GrabCut.
- MLLM customizing: You can swap CLIP with LLaVA style model; would require adjusting tokenizer alignment.

## Roadmap Ideas

- Integrate depth estimation for improved height inference.
- Few-shot density adaptation via Bayesian updating.
- Uncertainty estimation (MC dropout).