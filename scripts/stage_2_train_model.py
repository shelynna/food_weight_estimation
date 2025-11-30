"""
Stage 2: Train LLaVA Model
- Fine-tunes LLaVA on generated dataset with LoRA
"""
import torch
import os
import sys
from transformers import (
    AutoProcessor, 
    LlavaForCausalLM, 
    TrainingArguments, 
    Trainer, 
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

def stage_2_train_model(
    dataset_json_path,
    image_dir,
    checkpoint_dir,
    final_adapter_dir,
    model_id="llava-hf/llava-1.5-7b-hf",
    batch_size=2,
    grad_accum_steps=8,
    num_epochs=1,
    learning_rate=2e-4
):
    """
    Fine-tunes LLaVA model using LoRA on the generated dataset.
    """
    
    print("\n" + "="*60)
    print("STAGE 2: TRAIN MODEL")
    print("="*60)
    
    # 1. Config
    print(f"\n🔧 Configuration:")
    print(f"   Model: {model_id}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Gradient Accumulation: {grad_accum_steps}")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning Rate: {learning_rate}")
    
    torch.cuda.empty_cache()

    # 2. Load Model (4-bit)
    print("\n⏳ Loading LLaVA (4-bit quantization)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    try:
        model = LlavaForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config, 
            device_map="auto",
            torch_dtype=torch.float16
        )
        processor = AutoProcessor.from_pretrained(model_id)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return False

    # 3. Add LoRA
    print("\n⚡ Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Data Collator
    class LlavaCollator:
        def __init__(self, processor, image_dir):
            self.processor = processor
            self.image_dir = image_dir
        
        def __call__(self, batch):
            images = []
            texts = []
            
            for item in batch:
                # Load Image
                img_path = os.path.join(self.image_dir, item['image'])
                try:
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
                    images.append(Image.new('RGB', (336, 336)))

                # Format Text
                convs = item['conversations']
                text = self.processor.tokenizer.apply_chat_template(
                    convs, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                texts.append(text)

            batch_out = self.processor(
                text=texts, 
                images=images, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            )
            batch_out['labels'] = batch_out['input_ids'].clone()
            return batch_out

    # 5. Load Dataset
    print("\n📊 Loading dataset...")
    try:
        dataset = load_dataset(
            "json", 
            data_files=dataset_json_path, 
            split="train"
        )
        print(f"   ✓ Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"   ✗ Error loading dataset: {e}")
        return False

    # 6. Trainer
    print("\n🎯 Setting up trainer...")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    args = TrainingArguments(
        output_dir=checkpoint_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="steps",
        save_steps=50,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=LlavaCollator(processor, image_dir)
    )

    print("\n🚀 Starting Training...")
    try:
        trainer.train()
        print("   ✓ Training completed successfully")
    except Exception as e:
        print(f"   ✗ Training failed: {e}")
        return False

    # 7. Save Adapter
    print(f"\n💾 Saving adapter to {final_adapter_dir}...")
    os.makedirs(final_adapter_dir, exist_ok=True)
    try:
        model.save_pretrained(final_adapter_dir)
        processor.save_pretrained(final_adapter_dir)
        print("   ✓ Adapter saved successfully")
    except Exception as e:
        print(f"   ✗ Error saving adapter: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Example usage
    root_dir = "/path/to/project"
    stage_2_train_model(
        dataset_json_path=os.path.join(root_dir, "data", "dataset.json"),
        image_dir=os.path.join(root_dir, "images_with_gt_weights"),
        checkpoint_dir=os.path.join(root_dir, "food_llm_v1"),
        final_adapter_dir=os.path.join(root_dir, "final_adapter")
    )
