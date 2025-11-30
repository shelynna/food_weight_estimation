import os
import sys
import argparse
from pathlib import Path

# Import stage scripts
from stage_1_generate_data import stage_1_generate_data
from stage_2_train_model import stage_2_train_model
from stage_3_verify_test import stage_3_verify_test

def setup_environment(root_dir):
    """Set up directory structure and paths"""
    print("\n" + "="*60)
    print("PROJECT SETUP")
    print("="*60)
    
    # Define all paths
    paths = {
        'root': root_dir,
        'images': os.path.join(root_dir, "images_with_gt_weights"),
        'master_csv': os.path.join(root_dir, "ghana_gt_weights_w_filenames_images.csv"),
        'tf_model_dir': os.path.join(root_dir, "tf_portion_model"),
        'tf_model_file': os.path.join(root_dir, "tf_portion_model", "ghana_frozen_graph_9.0_489ksteps.pb"),
        'data_dir': os.path.join(root_dir, "data"),
        'checkpoint_dir': os.path.join(root_dir, "food_llm_v1"),
        'final_adapter_dir': os.path.join(root_dir, "final_adapter"),
        'train_subset_csv': os.path.join(root_dir, "ghana_train_subset.csv"),
        'dataset_json': os.path.join(root_dir, "data", "dataset.json"),
        'tf_script': os.path.join(root_dir, "scripts", "run_tf_inference.py"),
    }
    
    # Verify required input paths
    print("\nChecking input paths...")
    required_inputs = [
        paths['images'],
        paths['master_csv'],
        paths['tf_model_file'],
    ]
    
    for path in required_inputs:
        if os.path.exists(path):
            print(f"   OK {os.path.basename(path)}")
        else:
            print(f"   MISSING: {path}")
            return None
    
    # Create output directories
    print("\nCreating output directories...")
    os.makedirs(paths['data_dir'], exist_ok=True)
    os.makedirs(paths['checkpoint_dir'], exist_ok=True)
    os.makedirs(paths['final_adapter_dir'], exist_ok=True)
    print("   Output directories ready")
    
    return paths

def main():
    parser = argparse.ArgumentParser(
        description="Complete Project Reconstruction Pipeline"
    )
    parser.add_argument(
        "--root_dir",
        required=True,
        help="Root project directory (where all data lives)"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=150,
        help="Number of images for training (default: 150)"
    )
    parser.add_argument(
        "--stages",
        nargs='+',
        type=int,
        choices=[1, 2, 3],
        default=[1, 2, 3],
        help="Stages to run (default: 1 2 3)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size (default: 2)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--num_tests",
        type=int,
        default=3,
        help="Number of test samples (default: 3)"
    )
    
    args = parser.parse_args()
    
    # Setup
    print("\n" + "#"*60)
    print("# FOOD WEIGHT ESTIMATION - COMPLETE RECONSTRUCTION")
    print("#"*60)
    
    paths = setup_environment(args.root_dir)
    if paths is None:
        print("\n❌ Setup failed. Please check input paths.")
        return 1
    
    # Stage 1: Generate Data
    if 1 in args.stages:
        print("\n\n" + "#"*60)
        print("# STAGE 1: GENERATE DATA")
        print("#"*60)
        
        success = stage_1_generate_data(
            root_dir=paths['root'],
            master_csv_path=paths['master_csv'],
            image_dir=paths['images'],
            tf_model_file=paths['tf_model_file'],
            tf_script_path=paths['tf_script'],
            data_dir=paths['data_dir'],
            train_subset_csv=paths['train_subset_csv'],
            dataset_json_path=paths['dataset_json'],
            subset_size=args.subset_size
        )
        
        if not success:
            print("\n❌ Stage 1 failed!")
            return 1
        print("\nStage 1 complete.")
    
    # Stage 2: Train Model
    if 2 in args.stages:
        print("\n\n" + "#"*60)
        print("# STAGE 2: TRAIN MODEL")
        print("#"*60)
        
        success = stage_2_train_model(
            dataset_json_path=paths['dataset_json'],
            image_dir=paths['images'],
            checkpoint_dir=paths['checkpoint_dir'],
            final_adapter_dir=paths['final_adapter_dir'],
            batch_size=args.batch_size,
            num_epochs=args.num_epochs
        )
        
        if not success:
            print("\n❌ Stage 2 failed!")
            return 1
        print("\nStage 2 complete.")
    
    # Stage 3: Verify & Test
    if 3 in args.stages:
        print("\n\n" + "#"*60)
        print("# STAGE 3: VERIFY & TEST")
        print("#"*60)
        
        success = stage_3_verify_test(
            train_subset_csv=paths['train_subset_csv'],
            image_dir=paths['images'],
            tf_script_path=paths['tf_script'],
            tf_model_file=paths['tf_model_file'],
            final_adapter_dir=paths['final_adapter_dir'],
            num_tests=args.num_tests
        )
        
        if not success:
            print("\n❌ Stage 3 failed!")
            return 1
        print("\nStage 3 complete.")
    
    # Final summary
    print("\n\n" + "#"*60)
    print("# ALL STAGES COMPLETED SUCCESSFULLY!")
    print("#"*60)
    print(f"\nOutput locations:")
    print(f"   Adapter: {paths['final_adapter_dir']}")
    print(f"   Checkpoints: {paths['checkpoint_dir']}")
    print(f"   Data: {paths['dataset_json']}")
    print("\nReady for inference.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())