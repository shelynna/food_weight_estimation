"""
Stage 1: Generate Training Data
- Runs CV model (TensorFlow 1) in batch
- Formats data for LLaVA training
"""
import subprocess
import pandas as pd
import json
from tqdm import tqdm
import os
import sys

def stage_1_generate_data(
    root_dir,
    master_csv_path,
    image_dir,
    tf_model_file,
    tf_script_path,
    data_dir,
    train_subset_csv,
    dataset_json_path,
    subset_size=150
):
    """
    Generate training dataset by:
    1. Sampling from master CSV
    2. Running CV inference
    3. Formatting for LLaVA
    """
    
    print("\n" + "="*60)
    print("STAGE 1: GENERATE TRAINING DATA")
    print("="*60)
    
    # 1. Load Master CSV
    print(f"\n📄 Loading Master CSV: {master_csv_path}")
    try:
        df_master = pd.read_csv(master_csv_path)
        print(f"   ✓ Found {len(df_master)} total images")
    except Exception as e:
        print(f"   ✗ Error loading CSV: {e}")
        return False

    # 2. Create subset
    print(f"\n📊 Creating training subset (n={subset_size})...")
    df_subset = df_master.sample(n=min(subset_size, len(df_master)), random_state=42)
    df_subset.to_csv(train_subset_csv, index=False)
    print(f"   ✓ Saved subset to {train_subset_csv}")

    # 3. Run CV Model in Batch (Subprocess)
    print("\n🚀 Running CV Model (TensorFlow 1)...")
    print("   ⏳ This takes ~1 minute...")
    
    command = [
        "python", tf_script_path,
        "--model_path", tf_model_file,
        "--csv_path", train_subset_csv,
        "--image_dir", image_dir,
        "--filename_col", "Filename"
    ]

    cv_results = {}
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        cv_results = json.loads(result.stdout)
        print(f"   ✓ CV Model finished. Processed {len(cv_results)} images.")
    except subprocess.CalledProcessError as e:
        print("   ✗ CV Script Failed!")
        print(f"   STDERR: {e.stderr}")
        return False
    except json.JSONDecodeError:
        print("   ✗ CV output could not be parsed as JSON")
        return False

    # 4. Format for LLaVA
    print("\n📋 Formatting data for LLaVA...")
    dataset_data = []
    df_train = pd.read_csv(train_subset_csv)

    for _, row in tqdm(df_train.iterrows(), total=len(df_train), desc="   Building Dataset"):
        fname = str(row['Filename'])
        cv_weight = cv_results.get(fname)

        # Skip if CV failed or image missing
        if cv_weight is None:
            continue

        # Get GT data
        # Using 'GT Food name' based on the CSV structure
        food_name = str(row.get('GT Food name', 'Food'))
        try:
            real_weight = float(row['weight'])
        except:
            continue

        # Construct Prompt
        user_text = (
            f"<image>\n"
            f"Based on a computer vision analysis suggesting a weight of "
            f"around {cv_weight:.0f}g, identify the food and its exact weight."
        )
        assistant_text = f"Here is the breakdown:\n- {food_name}: {real_weight:.1f}g"

        dataset_data.append({
            "id": fname,
            "image": fname,
            "conversations": [
                {"from": "human", "value": user_text},
                {"from": "gpt", "value": assistant_text}
            ]
        })

    # 5. Save JSON
    os.makedirs(data_dir, exist_ok=True)
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_data, f, indent=2)

    print(f"   ✓ Created {len(dataset_data)} training samples")
    print(f"   ✓ Saved to {dataset_json_path}")
    
    return True

if __name__ == "__main__":
    # Example usage
    root_dir = "/path/to/project"
    stage_1_generate_data(
        root_dir,
        master_csv_path=os.path.join(root_dir, "ghana_gt_weights_w_filenames_images.csv"),
        image_dir=os.path.join(root_dir, "images_with_gt_weights"),
        tf_model_file=os.path.join(root_dir, "tf_portion_model", "ghana_frozen_graph_9.0_489ksteps.pb"),
        tf_script_path=os.path.join(root_dir, "scripts", "run_tf_inference.py"),
        data_dir=os.path.join(root_dir, "data"),
        train_subset_csv=os.path.join(root_dir, "ghana_train_subset.csv"),
        dataset_json_path=os.path.join(root_dir, "data", "dataset.json"),
        subset_size=150
    )
