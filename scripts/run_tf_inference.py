"""
TensorFlow 1 Inference Script - CV Backend for Weight Estimation
Implements correct logic:
  - Background pixel = 254
  - Weight calculation = pixels * 0.015
"""
import tensorflow.compat.v1 as tf
import numpy as np
import sys
import os
import argparse
import json
import pandas as pd
import cv2
from tqdm import tqdm

# Force TF1 compatibility
tf.disable_eager_execution()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_graph(frozen_graph_filename):
    """Load frozen TensorFlow graph"""
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

def preprocess_image(image_path, width=513, height=513):
    """Loads image using OpenCV to match training conditions."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Resize
        img_resized = cv2.resize(img_rgb, (width, height))
        # Add batch dimension: [1, 513, 513, 3]
        return np.expand_dims(img_resized, axis=0)
    except Exception as e:
        print(f"Error preprocessing {image_path}: {e}", file=sys.stderr)
        return None

def get_gram_weight(sess, image_tensor, output_tensor, image_path):
    """
    1. Runs the TF model to get the segmentation map.
    2. Counts pixels that are NOT background (254).
    3. Multiplies by 0.015 to get grams.
    """
    image_np = preprocess_image(image_path)
    if image_np is None:
        return None

    # Run inference
    seg_map = sess.run(output_tensor, feed_dict={image_tensor: image_np})
    seg_map = np.squeeze(seg_map)  # Remove batch dim

    # --- THE CORRECT LOGIC ---
    # Count pixels that are NOT background (254)
    # Based on your previous success, we use != 254
    pixel_count = np.sum(seg_map != 254)

    # Apply the linear regression factor
    gram_weight = float(pixel_count * 0.015)
    # -------------------------

    return gram_weight

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--filename_col", default="Filename")
    args = parser.parse_args()

    # Load Graph
    graph = load_graph(args.model_path)
    image_tensor = graph.get_tensor_by_name("ImageTensor:0")
    output_tensor = graph.get_tensor_by_name("SemanticPredictions:0")

    # Load Data
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    results = {}

    # Start Session
    with tf.Session(graph=graph) as sess:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="CV Inference", file=sys.stderr):
            fname = str(row[args.filename_col])
            img_path = os.path.join(args.image_dir, fname)

            if os.path.exists(img_path):
                w = get_gram_weight(sess, image_tensor, output_tensor, img_path)
                results[fname] = w
            else:
                results[fname] = None

    # Output JSON to stdout
    print(json.dumps(results))

if __name__ == "__main__":
    main()
