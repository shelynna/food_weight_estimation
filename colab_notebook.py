# This linear script can be run inside a Colab cell by %run colab_notebook.py
import os, sys, json, subprocess, shlex

DATA_ROOT = "/content/data"
OUTPUT_DIR = "/content/output"

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Installing requirements...")
subprocess.run(shlex.split("pip install -q -r requirements.txt"))

print("Stage 1: Train segmentation")
subprocess.run(shlex.split(f"python training/train_segmentation.py --data_root {DATA_ROOT} --output_dir {OUTPUT_DIR}"))

print("Stage 2: Train classification")
subprocess.run(shlex.split(f"python training/train_classifier.py --data_root {DATA_ROOT} --output_dir {OUTPUT_DIR}"))

print("Stage 3: Train weight estimation")
subprocess.run(shlex.split(f"python training/train_weight.py --data_root {DATA_ROOT} --output_dir {OUTPUT_DIR}"))

print("Inference example (using validation set as placeholder):")
VAL_ROOT = os.path.join(DATA_ROOT,"val")
subprocess.run(shlex.split(f"python inference/predict.py --data_root {VAL_ROOT} --checkpoint_dir {OUTPUT_DIR}/checkpoints --out_csv {OUTPUT_DIR}/val_predictions.csv"))
print("Done.")