import torch
from transformers import AutoProcessor, AutoModelForCausalLM

class MLLMWrapper:
    def __init__(self, model_name, device="cuda", max_new_tokens=128):
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        self.max_new_tokens=max_new_tokens
        self.device=device

    @torch.no_grad()
    def generate_with_image(self, image_pil, prompt: str):
        inputs = self.processor(prompt, images=image_pil, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        text = self.processor.decode(out[0], skip_special_tokens=True)
        return text

    @torch.no_grad()
    def image_embedding(self, image_pil):
        # Extract vision tower embeddings (approx method)
        vision_tower = self.model.get_vision_tower()
        image_inputs = self.processor(images=image_pil, return_tensors="pt").to(self.device)
        feats = vision_tower(image_inputs["pixel_values"], output_hidden_states=True)
        pooled = feats.hidden_states[-1].mean(1)  # [1, dim]
        return pooled.float()