import torch
import os
import sys
import pandas as pd
import json
import subprocess
import cv2
from peft import PeftModel
from transformers import LlavaForCausalLM, AutoProcessor, BitsAndBytesConfig
from PIL import Image
from tqdm import tqdm

def get_cv_hint(
    img_path,
    tf_script_path,
    tf_model_file,
    temp_csv="temp_test.csv"
):
    """Run CV helper for single image"""
    try:
        # Create temp CSV for single item
        img_dir = os.path.dirname(img_path)
        img_name = os.path.basename(img_path)
        
        temp_df = pd.DataFrame({'Filename': [img_name]})
        temp_df.to_csv(temp_csv, index=False)
        
        cmd = [
            "python", tf_script_path,
            "--model_path", tf_model_file,
            "--csv_path", temp_csv,
            "--image_dir", img_dir,
            "--filename_col", "Filename"
        ]
        
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        
        data = json.loads(res.stdout)
        return data.get(img_name, None)
    except Exception as e:
        print(f"Warning: CV hint failed: {e}")
        if os.path.exists(temp_csv):
            os.remove(temp_csv)
        return None

def stage_3_verify_test(
    train_subset_csv,
    image_dir,
    tf_script_path,
    tf_model_file,
    final_adapter_dir,
    model_base_id="llava-hf/llava-1.5-7b-hf",
    num_tests=3
):
    """
    Verify model works by testing on random images.
    """
    
    print("\n" + "="*60)
    print("STAGE 3: VERIFY & TEST")
    print("="*60)
    
    # 1. Load Trained Model
    print(f"\n Loading Fine-Tuned Model...")
    try:
        torch.cuda.empty_cache()
        
        model_base = LlavaForCausalLM.from_pretrained(
            model_base_id,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, 
                bnb_4bit_compute_dtype=torch.float16
            ),
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = PeftModel.from_pretrained(model_base, final_adapter_dir)
        processor = AutoProcessor.from_pretrained(final_adapter_dir)
        model.eval()
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return False

    # 2. Load test data
    print(f"\n Loading test data...")
    try:
        df_test = pd.read_csv(train_subset_csv)
        print(f"   ✓ Loaded {len(df_test)} test samples")
    except Exception as e:
        print(f"   ✗ Error loading test data: {e}")
        return False

    # 3. Run tests
    print(f"\n Running {num_tests} test(s)...")
    results = []
    
    test_samples = df_test.sample(n=min(num_tests, len(df_test)))
    
    for idx, (_, sample) in enumerate(test_samples.iterrows(), 1):
        print(f"\n   Test {idx}/{num_tests}")
        print("   " + "-"*50)
        
        fname = str(sample['Filename'])
        img_path = os.path.join(image_dir, fname)
        
        if not os.path.exists(img_path):
            print(f"   ✗ Image not found: {img_path}")
            continue
        
        # Get CV hint
        print(f" Getting CV hint...")
        cv_weight = get_cv_hint(img_path, tf_script_path, tf_model_file)
        if cv_weight is None:
            print(f"   ⚠ CV hint unavailable")
            cv_weight = 0
        else:
            print(f"   ✓ CV Weight: {cv_weight:.1f}g")
        
        # Generate LLaVA Output
        print(f" Generating model output...")
        try:
            prompt = (
                f"USER: <image>\n"
                f"Based on a computer vision analysis suggesting a weight of "
                f"around {cv_weight:.0f}g, identify the food and its exact weight.\n"
                f"ASSISTANT:"
            )
            
            image = Image.open(img_path).convert('RGB')
            inputs = processor(
                prompt, 
                images=image, 
                return_tensors="pt"
            ).to("cuda" if torch.cuda.is_available() else "cpu")
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=60)
            
            decoded = processor.decode(out[0], skip_special_tokens=True)
            response = decoded.split("ASSISTANT:")[-1].strip()
            
            print(f"   ✓ Model Output: {response[:100]}...")
        except Exception as e:
            print(f"   ✗ Model generation failed: {e}")
            response = "Error"
        
        # Ground truth
        try:
            food_name = str(sample.get('GT Food name', 'Food'))
            gt_weight = float(sample['weight'])
            gt_text = f"{food_name} ({gt_weight}g)"
            print(f" Ground Truth: {gt_text}")
        except:
            gt_text = "Unknown"
            print(f"   ⚠ Ground truth unavailable")
        
        # Store result
        results.append({
            'image': fname,
            'cv_weight': cv_weight,
            'model_output': response,
            'ground_truth': gt_text
        })
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['image']}")
        print(f"   CV:       {result['cv_weight']:.1f}g")
        print(f"   Model:    {result['model_output'][:80]}...")
        print(f"   GT:       {result['ground_truth']}")
    
    print("\n Verification complete!")
    return True

if __name__ == "__main__":
    # Example usage
    root_dir = "/path/to/project"
    stage_3_verify_test(
        train_subset_csv=os.path.join(root_dir, "ghana_train_subset.csv"),
        image_dir=os.path.join(root_dir, "images_with_gt_weights"),
        tf_script_path=os.path.join(root_dir, "scripts", "run_tf_inference.py"),
        tf_model_file=os.path.join(root_dir, "tf_portion_model", "ghana_frozen_graph_9.0_489ksteps.pb"),
        final_adapter_dir=os.path.join(root_dir, "final_adapter"),
        num_tests=3
    )
