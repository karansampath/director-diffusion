#!/usr/bin/env python

from pathlib import Path
from typing import Optional
import io
import logging

import modal
import torch
from PIL import Image
from diffusers import FluxPipeline

# Import from training module
from src.train import MODEL_DIR, DIRECTORS_DIR, volume, huggingface_secret
from src.config import InferenceConfig, LoRAInfo, GradioConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Modal image with Gradio support
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "accelerate",
        "diffusers",
        "torch>=2.1.0",
        "torchvision",
        "peft>=0.8.0",
        "transformers",
        "sentencepiece",
        "huggingface-hub[hf_transfer]",
        "pillow",
        "numpy",
        "fastapi[standard]",
        "gradio~=5.7.1",
        "pydantic==2.10.6",
        "pillow",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCH_CUDNN_V8_API_ENABLED": "1",
        }
    )
)



# Create Modal app
app = modal.App(
    name="flux-lora-gradio",
    image=image,
    volumes={"/volume": volume},
    secrets=[huggingface_secret],
)

@app.cls(
    gpu="H100",
    timeout=300,
    scaledown_window=240,
    volumes={"/volume": volume},
    secrets=[huggingface_secret],
)
class FluxServeModel:
    """Modal class for serving Flux model with LoRA support."""
    
    base_pipeline: Optional[FluxPipeline] = None
    available_loras: dict[str, LoRAInfo] = {}
    device: str = "cuda"
    
    @modal.enter()
    def setup_model(self):
        """Initialize the model and discover LoRAs."""
        logger.info("Setting up Flux inference model...")
        
        # Reload volume to get latest state
        volume.reload()
        
        # Load base model
        logger.info("Loading base Flux model...")
        self.base_pipeline = FluxPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
        )
        self.base_pipeline.to(self.device)
        self.base_pipeline.enable_model_cpu_offload()
        logger.info("Base model loaded successfully")
        
        # Discover LoRAs
        self.discover_loras()
        
        logger.info("Model setup complete")
    
    def discover_loras(self) -> None:
        """Discover available LoRA adapters."""
        logger.info("Discovering LoRA adapters...")
        
        loras_path = Path(DIRECTORS_DIR)
        if not loras_path.exists():
            logger.warning(f"LoRA directory not found: {loras_path}")
            return
            
        for lora_dir in loras_path.iterdir():
            if lora_dir.is_dir():
                # Check if it contains LoRA files
                if any(lora_dir.glob("*.safetensors")) or any(lora_dir.glob("adapter_config.json")):
                    # Try to infer trigger phrase and description
                    trigger_phrase = f"<{lora_dir.name}-style>"
                    description = f"{lora_dir.name.title()} style adaptation"
                    
                    self.available_loras[lora_dir.name] = LoRAInfo(
                        name=lora_dir.name,
                        path=str(lora_dir),
                        trigger_phrase=trigger_phrase,
                        description=description,
                    )
                    
        logger.info(f"Found {len(self.available_loras)} LoRA adapters")
    
    @modal.method()
    def get_available_loras(self) -> dict[str, dict[str, str]]:
        """Get list of available LoRA adapters."""
        return {
            name: {
                "description": lora.description,
                "trigger_phrase": lora.trigger_phrase,
            }
            for name, lora in self.available_loras.items()
        }
    
    @modal.method()
    def generate_image(
        self, 
        prompt: str, 
        lora_name: Optional[str] = None,
        seed: Optional[int] = None
    ) -> bytes:
        """Generate an image with optional LoRA adapter."""
        
        if seed is not None:
            torch.manual_seed(seed)
            
        pipeline = self.base_pipeline
        
        # Load LoRA if specified
        if lora_name and lora_name in self.available_loras:
            lora_info = self.available_loras[lora_name]
            logger.info(f"Loading LoRA: {lora_name}")
            
            # Load LoRA weights
            pipeline.load_lora_weights(lora_info.path)
            
            # Add trigger phrase if not already in prompt
            if lora_info.trigger_phrase not in prompt:
                prompt = f"{lora_info.trigger_phrase} {prompt}"
        
        # Load inference config
        config = InferenceConfig()
        
        # Generate image
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                height=config.height,
                width=config.width,
                num_inference_steps=config.num_inference_steps,
                guidance_scale=config.guidance_scale,
                max_sequence_length=config.max_sequence_length,
            )
        
        # Convert to bytes
        buffer = io.BytesIO()
        result.images[0].save(buffer, format="PNG")
        return buffer.getvalue()

    @modal.method()
    def generate_comparison(
        self, 
        prompt: str, 
        lora_name: str,
        seed: Optional[int] = None
    ) -> tuple[bytes, bytes]:
        """Generate side-by-side comparison images."""
        
        logger.info(f"Generating comparison for prompt: {prompt}")
        
        # Generate base model image
        base_image = self.generate_image.remote(prompt, lora_name=None, seed=seed)
        
        # Generate LoRA image  
        lora_image = self.generate_image.remote(prompt, lora_name=lora_name, seed=seed)
        
        return base_image, lora_image


@app.function(
    image=image,
    min_containers=1,
    scaledown_window=60 * 20,
    max_containers=1,
    volumes={"/volume": volume},
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def gradio_app():
    """Create and serve the Gradio interface."""
    
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    
    # Initialize model reference (will be called remotely)
    model = FluxServeModel
    
    def generate_side_by_side(
        prompt: str,
        lora_selection: str,
        seed: Optional[int] = None,
    ) -> tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Generate side-by-side comparison."""
        
        if not prompt.strip():
            return None, None
            
        if lora_selection == "None":
            return None, None
            
        try:
            # Get comparison images as bytes
            base_bytes, lora_bytes = model().generate_comparison.remote(
                prompt=prompt,
                lora_name=lora_selection,
                seed=seed,
            )
            
            # Convert bytes to PIL Images
            base_img = Image.open(io.BytesIO(base_bytes))
            lora_img = Image.open(io.BytesIO(lora_bytes))
            
            return base_img, lora_img
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None, None
    
    def get_lora_options() -> list[str]:
        """Get available LoRA options."""
        try:
            loras = model().get_available_loras.remote()
            return ["None"] + list(loras.keys())
        except Exception as e:
            logger.error(f"Failed to get LoRAs: {e}")
            return ["None"]
    
    # Load Gradio configuration
    try:
        gradio_config = GradioConfig()
        title = gradio_config.title
        description = gradio_config.description
        example_prompts = gradio_config.example_prompts
        theme = gradio_config.theme
    except:
        # Fallback config if GradioConfig fails
        title = "FLUX LoRA Playground"
        description = "Compare base FLUX model with LoRA fine-tuned versions"
        example_prompts = [
            "A beautiful landscape painting",
            "Portrait of a person in cinematic lighting", 
            "Abstract art with vibrant colors",
            "Futuristic cityscape at sunset"
        ]
        theme = "soft"
    
    # Create Gradio interface using gr.Interface (simpler and more reliable)
    demo = gr.Interface(
        fn=generate_side_by_side,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Enter your image generation prompt...", lines=3),
            gr.Dropdown(label="Select LoRA", choices=get_lora_options(), value="None"),
            gr.Number(label="Seed (optional)", value=None, precision=0),
        ],
        outputs=[
            gr.Image(label="Base Model Output", height=512),
            gr.Image(label="LoRA Model Output", height=512),
        ],
        title=title,
        description=description,
        theme=theme,
        allow_flagging="never",
    )
    
    # Enable queue - this is crucial for handling remote Modal calls
    demo.queue(max_size=10)
    
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
