import os
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText
from diffusers import AutoPipelineForText2Image
from PIL import Image
import logging
import io

# Configure logging to capture to string
log_capture_string = io.StringIO()
ch = logging.StreamHandler(log_capture_string)
ch.setLevel(logging.INFO)

logging.basicConfig(level=logging.INFO, handlers=[
    logging.StreamHandler(), # Output to console
    ch # Output to string buffer
])
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.device = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = os.environ.get("MODEL_SIZE", "small") # small, large
        self.vqa_model = None
        self.vqa_processor = None
        self.t2i_pipeline = None
        self.model_map = {
           "large": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
           "small": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        }
        
        logger.info(f"Model Manager initialized. Device: {self.device}, Model Version: {self.model_size}")

    def load_vqa_model(self, model_size=None):
        if model_size and model_size != self.model_size:
            logger.info(f"Switching model size from {self.model_size} to {model_size}...")
            self.model_size = model_size
            self.vqa_model = None # Force reload
            self.vqa_processor = None

        if self.vqa_model is not None:
            return
        

        model_id = self.model_map[self.model_size]
        
        logger.info(f"Loading VQA model: {model_id} (Size: {self.model_size})...")
        try:
            self.vqa_processor = AutoProcessor.from_pretrained(model_id)
            # Check if flash_attn is available
            attn_implementation = "eager"
            if self.device == "cuda":
                try:
                    import flash_attn
                    attn_implementation = "flash_attention_2"
                except ImportError:
                    logger.warning("FlashAttention2 not installed. Falling back to default attention.")
                    attn_implementation = "sdpa" # Scaled Dot Product Attention (PyTorch 2.0+)

            self.vqa_model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                _attn_implementation=attn_implementation,
            )
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

            logger.info("Text-to-Image model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading T2I model: {e}")
            raise e

    def process_vqa(self, image_path, prompt, model_size=None):
        self.load_vqa_model(model_size)
        self.vqa_model.to(self.device)
        
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
                
        response = generated_texts[0].split("\nAssistant:")[1]
        # logger.info(f"Generated response: {generated_texts}")
        # logger.info(f"VQA response: {response}, response_type {type(response)}")
        if prompt_text in response:
             response = response.replace(prompt_text, "").strip()

        self.vqa_model.to("cpu")
        torch.cuda.empty_cache()
             
        return response

    def generate_image(self, prompt):
        self.load_t2i_model()
        self.t2i_pipeline.to(self.device)
        
        image = self.t2i_pipeline(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
        
        self.t2i_pipeline.to("cpu")
        torch.cuda.empty_cache()
        
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
        
        if self.device == "cpu":
            self.vqa_model = self.vqa_model.float()
        else:
            self.vqa_model = self.vqa_model.half()

        logger.info(f"Switched to {self.device}.")

    def get_config(self):
        return {
            "device": self.device,
            "model_size": self.model_size
        }

    def get_logs(self):
        return log_capture_string.getvalue()
