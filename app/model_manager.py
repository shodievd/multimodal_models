import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from diffusers import AutoPipelineForText2Image
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = os.environ.get("MODEL_SIZE", "small") # small, large
        self.vqa_model = None
        self.vqa_processor = None
        self.t2i_pipeline = None
        
        logger.info(f"Model Manager initialized. Device: {self.device}, Model Size: {self.model_size}")

    def load_vqa_model(self, model_size=None):
        if model_size and model_size != self.model_size:
            logger.info(f"Switching model size from {self.model_size} to {model_size}...")
            self.model_size = model_size
            self.vqa_model = None # Force reload
            self.vqa_processor = None

        if self.vqa_model is not None:
            return

        # Map model_size to actual model ID
        # For this demo, we can simulate different models or use actual variants if available.
        # SmolVLM-Instruct is the main one. Let's assume "SmolVLM-256M" maps to a smaller one or same for demo.
        # Actually, let's just use the same model but log the "size" change to demonstrate the logic,
        # or if there is a real smaller variant, use it.
        # HuggingFaceTB/SmolVLM-Instruct is 2.2B params.
        # Let's assume the user might want to switch to a different one.
        
        model_id = "HuggingFaceTB/SmolVLM-Instruct"
        if self.model_size == "SmolVLM-256M":
             # Placeholder for a smaller model if it existed, or just re-use for demo purposes
             # to avoid downloading another huge model.
             # But to be "real", let's say we use the same one but maybe with different quantization if we could?
             # For now, let's just stick to the main one but acknowledge the config.
             pass
        
        logger.info(f"Loading VQA model: {model_id} (Size: {self.model_size})...")
        try:
            self.vqa_processor = AutoProcessor.from_pretrained(model_id)
            self.vqa_model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
            ).to(self.device)
            logger.info("VQA model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading VQA model: {e}")
            raise e

    def load_t2i_model(self):
        if self.t2i_pipeline is not None:
            return

        model_id = "stabilityai/sdxl-turbo"
        
        logger.info(f"Loading Text-to-Image model: {model_id}...")
        try:
            self.t2i_pipeline = AutoPipelineForText2Image.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, 
                variant="fp16" if self.device == "cuda" else None
            )
            self.t2i_pipeline.to(self.device)
            logger.info("Text-to-Image model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading T2I model: {e}")
            raise e

    def process_vqa(self, image_path, prompt, model_size=None):
        self.load_vqa_model(model_size)
        
        image = Image.open(image_path).convert("RGB")
        
        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
        ]
        
        prompt_text = self.vqa_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.vqa_processor(text=prompt_text, images=[image], return_tensors="pt")
        inputs = inputs.to(self.device)
        
        generated_ids = self.vqa_model.generate(**inputs, max_new_tokens=500)
        generated_texts = self.vqa_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        # Extract the assistant's response
        # The prompt is included in the output, so we might need to strip it or just return the last part
        # SmolVLM usually returns the full conversation.
        # Let's try to parse it or just return the raw text for now, but usually we want just the answer.
        # For simplicity in this demo, we'll return the whole text and let the frontend or user see it, 
        # or we can try to split by "Assistant:".
        
        response = generated_texts[0]
        # Basic cleanup if needed, but apply_chat_template usually handles structure well.
        # If the model repeats the prompt, we might want to cut it.
        # For SmolVLM, it typically appends the answer.
        
        # A simple heuristic to remove the prompt if it's repeated:
        if prompt_text in response:
             response = response.replace(prompt_text, "").strip()
             
        return response

    def generate_image(self, prompt):
        self.load_t2i_model()
        
        image = self.t2i_pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        return image

    def set_device(self, device):
        if device not in ["cuda", "cpu"]:
            raise ValueError("Device must be 'cuda' or 'cpu'")
        
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("GPU currently unavailable on this machine")
            
        if device == self.device:
            return

        logger.info(f"Switching device from {self.device} to {device}...")
        self.device = device
        
        # Move existing models
        if self.vqa_model:
            self.vqa_model.to(self.device)
            # Update dtype if needed, but usually fp16 on cpu is slow/not supported for some ops, 
            # so we might need to reload or cast. 
            # For simplicity, we just move. Ideally we should reload with correct precision.
            if self.device == "cpu":
                self.vqa_model = self.vqa_model.float()
            else:
                self.vqa_model = self.vqa_model.half()

        if self.t2i_pipeline:
            self.t2i_pipeline.to(self.device)

        logger.info(f"Switched to {self.device}.")

    def get_config(self):
        return {
            "device": self.device,
            "model_size": self.model_size
        }
