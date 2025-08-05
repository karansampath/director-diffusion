#!/usr/bin/env python

from pathlib import Path
from typing import Optional
import io
import logging
import random
import time

import modal
import torch
from PIL import Image
from diffusers import FluxPipeline
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe

# Import from training module
from src.train import MODEL_DIR, DIRECTORS_DIR, volume, huggingface_secret
from src.config import InferenceConfig, LoRAInfo, GradioConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common configuration
CONTAINER_CACHE_DIR = Path("/cache")
CONTAINER_CACHE_VOLUME = modal.Volume.from_name("flux_lora_cache", create_if_missing=True)

# Optimized Modal image
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "accelerate==1.6.0",
        "diffusers==0.33.1", 
        "torch>=2.7.0",
        "torchvision",
        "peft>=0.8.0",
        "transformers==4.51.3",
        "sentencepiece==0.2.0",
        "huggingface-hub[hf_transfer]==0.30.2",
        "pillow",
        "numpy==2.2.4",
        "fastapi[standard]",
        "gradio~=5.7.1",
        "pydantic==2.10.6",
        "para-attn==0.3.32",
        "safetensors==0.5.3",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TORCH_CUDNN_V8_API_ENABLED": "1",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "CUDA_CACHE_PATH": str(CONTAINER_CACHE_DIR / ".nv_cache"),
        "HF_HUB_CACHE": str(CONTAINER_CACHE_DIR / ".hf_hub_cache"),
        "TORCHINDUCTOR_CACHE_DIR": str(CONTAINER_CACHE_DIR / ".inductor_cache"),
        "TRITON_CACHE_DIR": str(CONTAINER_CACHE_DIR / ".triton_cache"),
    })
)

app = modal.App(
    name="flux-lora-gradio",
    image=image,
    volumes={
        "/volume": volume,
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME,
    },
    secrets=[huggingface_secret],
)

@app.cls(
    gpu="H100",
    timeout=2000,
    scaledown_window=300,
    min_containers=1,
    volumes={
        "/volume": volume,
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME,
    },
    secrets=[huggingface_secret],
    enable_memory_snapshot=True,
)
class FluxServeModel:
    """Optimized Modal class for serving Flux model with LoRA support."""
    
    def _optimize_pipeline(self, pipeline: FluxPipeline, compile_pipeline: bool = True) -> None:
        """Apply optimization techniques from the reference implementation."""
        # Apply first block cache
        apply_cache_on_pipe(pipeline, residual_diff_threshold=0.12)
        
        # Fuse QKV projections
        pipeline.transformer.fuse_qkv_projections()
        pipeline.vae.fuse_qkv_projections()
        
        # Use channels last memory format
        pipeline.transformer.to(memory_format=torch.channels_last)
        pipeline.vae.to(memory_format=torch.channels_last)
        
        if compile_pipeline:
            # Torch compile configuration
            config = torch._inductor.config
            config.conv_1x1_as_mm = True
            config.coordinate_descent_check_all_directions = True
            config.coordinate_descent_tuning = True
            config.disable_progress = False
            config.epilogue_fusion = False
            config.shape_padding = True
            
            # Compile critical components
            pipeline.transformer = torch.compile(
                pipeline.transformer, mode="max-autotune-no-cudagraphs", dynamic=True
            )
            pipeline.vae.decode = torch.compile(
                pipeline.vae.decode, mode="max-autotune-no-cudagraphs", dynamic=True
            )
    
    def _load_mega_cache(self) -> None:
        """Load torch mega-cache if available."""
        try:
            if self.mega_cache_path.exists():
                with open(self.mega_cache_path, "rb") as f:
                    artifacts = f.read()
                if artifacts:
                    torch.compiler.load_cache_artifacts(artifacts)
                logger.info("Loaded torch mega-cache")
        except Exception as e:
            logger.warning(f"Could not load mega-cache: {e}")
    
    def _save_mega_cache(self) -> None:
        """Save torch mega-cache."""
        try:
            artifacts = torch.compiler.save_cache_artifacts()
            with open(self.mega_cache_path, "wb") as f:
                f.write(artifacts[0])
            CONTAINER_CACHE_VOLUME.commit()
            logger.info("Saved torch mega-cache")
        except Exception as e:
            logger.warning(f"Could not save mega-cache: {e}")
    
    def _compile_pipeline(self) -> None:
        """Trigger compilation with dummy prompts."""
        logger.info("Compiling pipeline...")
        
        # Monkey-patch for para-attn compatibility
        from torch._inductor.fx_passes import post_grad
        if not hasattr(post_grad, "_orig_same_meta"):
            post_grad._orig_same_meta = post_grad.same_meta
            def _safe_same_meta(node1, node2):
                try:
                    return post_grad._orig_same_meta(node1, node2)
                except AttributeError as e:
                    if "SymFloat" in str(e) and "size" in str(e):
                        return False
                    raise
            post_grad.same_meta = _safe_same_meta
        
        # Trigger compilation with short dummy run
        dummy_result = self.base_pipeline(
            "test", 
            height=512,  # Smaller size for compilation
            width=512, 
            num_inference_steps=4,  # Much fewer steps for compilation
            guidance_scale=3.5,
            max_sequence_length=256,
        )
        logger.info("Pipeline compiled")
    
    def _discover_loras(self) -> None:
        """Discover available LoRA adapters."""
        logger.info("Discovering LoRA adapters...")
        
        loras_path = Path(DIRECTORS_DIR)
        if not loras_path.exists():
            logger.warning(f"LoRA directory not found: {loras_path}")
            return
            
        for lora_dir in loras_path.iterdir():
            if lora_dir.is_dir() and (
                any(lora_dir.glob("*.safetensors")) or 
                any(lora_dir.glob("adapter_config.json"))
            ):
                self.available_loras[lora_dir.name] = LoRAInfo(
                    name=lora_dir.name,
                    path=str(lora_dir),
                    trigger_phrase=f"<{lora_dir.name}-style>",
                    description=f"{lora_dir.name.title()} style adaptation",
                )
                    
        logger.info(f"Found {len(self.available_loras)} LoRA adapters")
    
    @modal.enter(snap=True)
    def load_model(self):
        """Load base model (snapshotted)."""
        logger.info("Loading base Flux model...")
        volume.reload()
        
        self.base_pipeline = FluxPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cpu")
        
        self.available_loras = {}
        self.current_lora = None  # Track currently loaded LoRA
        
        # Set up cache paths
        cache_dir = CONTAINER_CACHE_DIR / ".mega_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.mega_cache_path = cache_dir / "flux_lora_mega"
        
        logger.info("Base model loaded")
    
    @modal.enter(snap=False)
    def setup_model(self):
        """Setup and optimize model (not snapshotted)."""
        self.base_pipeline.to("cuda")
        
        # Create separate pipeline for LoRA inference (uncompiled for faster switching)
        logger.info("Creating LoRA pipeline (uncompiled)...")
        self.lora_pipeline = FluxPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to("cuda")
        
        self._load_mega_cache()
        
        # Optimize base pipeline with compilation
        logger.info("Optimizing base pipeline (compiled)...")
        self._optimize_pipeline(self.base_pipeline, compile_pipeline=True)
        self._compile_pipeline()
        self._save_mega_cache()
        
        # Optimize LoRA pipeline without compilation
        logger.info("Optimizing LoRA pipeline (uncompiled)...")
        self._optimize_pipeline(self.lora_pipeline, compile_pipeline=False)
        
        self._discover_loras()
        self.config = InferenceConfig()
        
        logger.info("Model setup complete")
    
    @modal.method()
    def get_available_loras(self) -> dict[str, dict[str, str]]:
        """Get available LoRA adapters."""
        return {
            name: {
                "description": lora.description,
                "trigger_phrase": lora.trigger_phrase,
            }
            for name, lora in self.available_loras.items()
        }
    
    def _load_lora_if_needed(self, lora_name: Optional[str]) -> bool:
        """Load LoRA only if different from current."""
        if lora_name == self.current_lora:
            return self.current_lora is not None
        
        # Unload current LoRA if any
        if self.current_lora is not None:
            self.lora_pipeline.unload_lora_weights()
            self.current_lora = None
        
        # Load new LoRA if specified
        if lora_name and lora_name in self.available_loras:
            lora_info = self.available_loras[lora_name]
            self.lora_pipeline.load_lora_weights(lora_info.path)
            self.current_lora = lora_name
            return True
        
        return False
    
    def generate_image(
        self, 
        prompt: str, 
        lora_name: Optional[str] = None,
        seed: Optional[int] = None
    ) -> bytes:
        """Generate image with optional LoRA."""
        if seed is not None:
            torch.manual_seed(seed)
        
        # Determine which pipeline to use
        if lora_name is None:
            # Use compiled base pipeline for base model inference
            pipeline = self.base_pipeline
            lora_loaded = False
        else:
            # Use uncompiled LoRA pipeline for LoRA inference
            pipeline = self.lora_pipeline
            lora_loaded = self._load_lora_if_needed(lora_name)
        
        # Add trigger phrase if LoRA is loaded
        if lora_loaded and lora_name in self.available_loras:
            lora_info = self.available_loras[lora_name]
            if lora_info.trigger_phrase not in prompt:
                prompt = f"{lora_info.trigger_phrase} {prompt}"
        
        # Time inference
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # Generate image
        with torch.no_grad():
            result = pipeline(
                prompt=prompt,
                height=self.config.height,
                width=self.config.width,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                max_sequence_length=self.config.max_sequence_length,
            )
        
        torch.cuda.synchronize()
        inference_time = time.perf_counter() - start_time
        pipeline_type = "compiled base" if lora_name is None else "uncompiled LoRA"
        logger.info(f"Inference time ({pipeline_type}): {inference_time:.2f}s")
        
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
        """Generate comparison images efficiently."""
        # Generate base image first
        base_image = self.generate_image(prompt, lora_name=None, seed=seed)
        
        # Generate LoRA image second (LoRA will be loaded only once)
        lora_image = self.generate_image(prompt, lora_name=lora_name, seed=seed)
        
        return base_image, lora_image

    def generate_single(
        self, 
        prompt: str, 
        lora_name: Optional[str] = None,
        seed: Optional[int] = None
    ) -> bytes:
        """Generate single image - faster for non-comparison use."""
        return self.generate_image(prompt, lora_name=lora_name, seed=seed)


@app.function(
    image=image,
    min_containers=1,
    scaledown_window=300,
    timeout=30000,
    max_containers=1,
    volumes={"/volume": volume},
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def gradio_app():
    """Create Gradio interface with comparison and blind voting tabs."""
    
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    
    model = FluxServeModel()
    
    # Common generation function
    def generate_images(
        prompt: str,
        lora_selection: str,
        seed: Optional[int] = None,
        *args
    ) -> tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Generate comparison images."""
        logger.info(f"generate_images called with prompt='{prompt}', lora_selection='{lora_selection}', seed={seed}")
        
        if not prompt.strip():
            logger.warning("Empty prompt provided")
            return None, None
            
        # For comparison, we need at least one LoRA to compare against base
        if lora_selection == "None":
            logger.warning("No LoRA selected for comparison - need a LoRA to compare against base model")
            return None, None
            
        try:
            logger.info("Starting model generation...")
            base_bytes, lora_bytes = model.generate_comparison.remote(
                prompt=prompt,
                lora_name=lora_selection,
                seed=seed,
            )
            logger.info("Model generation completed successfully")
            
            return (
                Image.open(io.BytesIO(base_bytes)),
                Image.open(io.BytesIO(lora_bytes))
            )
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None, None
    
    # Single image generation function
    def generate_single_image(
        prompt: str,
        lora_selection: str,
        seed: Optional[int] = None,
    ) -> Optional[Image.Image]:
        """Generate single image - faster for testing."""
        if not prompt.strip():
            return None
            
        try:
            lora_name = None if lora_selection == "None" else lora_selection
            image_bytes = model.generate_single.remote(
                prompt=prompt,
                lora_name=lora_name,
                seed=seed,
            )
            return Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Single generation failed: {e}")
            return None
    
    def generate_blind_comparison(
        prompt: str,
        lora_selection: str,
        seed: Optional[int] = None,
    ) -> tuple[Optional[Image.Image], Optional[Image.Image], str]:
        """Generate images for blind comparison with randomized order."""
        base_img, lora_img = generate_images(prompt, lora_selection, seed)
        
        if base_img is None or lora_img is None:
            return None, None, ""
        
        # Randomize order
        if random.choice([True, False]):
            return base_img, lora_img, "left=base,right=lora"
        else:
            return lora_img, base_img, "left=lora,right=base"
    
    def get_lora_options() -> list[str]:
        """Get available LoRA options."""
        try:
            loras = model.get_available_loras.remote()
            logger.info(f"Available LoRAs: {list(loras.keys())}")
            return ["None"] + list(loras.keys())
        except Exception as e:
            logger.error(f"Failed to get LoRAs: {e}")
            return ["None"]
    
    # Load configuration
    try:
        gradio_config = GradioConfig()
        title = gradio_config.title
        description = gradio_config.description
        example_prompts = gradio_config.example_prompts
    except Exception:
        title = "FLUX LoRA Playground"
        description = "Compare base FLUX model with LoRA fine-tuned versions"
        example_prompts = [
            "A beautiful landscape painting",
            "Portrait of a person in cinematic lighting", 
            "Abstract art with vibrant colors",
            "Futuristic cityscape at sunset"
        ]
    
    # Custom CSS for clean blue theme and Garamond-like font
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;600&display=swap');
    
    .gradio-container {
        font-family: 'EB Garamond', 'Times New Roman', serif !important;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .gr-button-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        font-family: 'EB Garamond', serif !important;
        font-weight: 500 !important;
    }
    
    .gr-button-primary:hover {
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    
    .gr-panel {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(102, 126, 234, 0.1) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
    }
    
    .gr-textbox, .gr-dropdown, .gr-number {
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 8px !important;
        font-family: 'EB Garamond', serif !important;
    }
    
    .gr-textbox:focus, .gr-dropdown:focus, .gr-number:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    h1, h2, h3 {
        font-family: 'EB Garamond', serif !important;
        color: #2c3e50 !important;
    }
    
    .gr-tab-nav .gr-tab.selected {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-family: 'EB Garamond', serif !important;
        font-weight: 500 !important;
    }
    """
    
    # Common input components
    def create_inputs():
        return [
            gr.Textbox(
                label="Prompt", 
                placeholder="Enter your image generation prompt...", 
                lines=3
            ),
            gr.Dropdown(
                label="Select LoRA", 
                choices=get_lora_options(), 
                value="None"
            ),
            gr.Number(
                label="Seed (optional)", 
                value=None, 
                precision=0
            ),
        ]
    
    # Create tabbed interface
    with gr.Blocks(css=custom_css, title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        
        with gr.Tabs():
            # Quick Test Tab - for faster single image generation
            with gr.Tab("Quick Test"):
                with gr.Row():
                    with gr.Column():
                        quick_inputs = create_inputs()
                        quick_button = gr.Button("Generate Single Image", variant="primary")
                    
                with gr.Row():
                    quick_output = gr.Image(label="Generated Image", height=512)
                
                quick_button.click(
                    fn=generate_single_image,
                    inputs=quick_inputs,
                    outputs=[quick_output]
                )
            
            # Comparison Tab
            with gr.Tab("Comparison"):
                with gr.Row():
                    with gr.Column():
                        comp_inputs = create_inputs()
                        comp_button = gr.Button("Generate Comparison", variant="primary")
                    
                with gr.Row():
                    comp_base_out = gr.Image(label="Base Model", height=512)
                    comp_lora_out = gr.Image(label="LoRA Model", height=512)
                
                comp_button.click(
                    fn=generate_images,
                    inputs=comp_inputs,
                    outputs=[comp_base_out, comp_lora_out]
                )
            
            # Blind Voting Tab
            with gr.Tab("Blind Vote"):
                with gr.Row():
                    with gr.Column():
                        blind_inputs = create_inputs()
                        blind_button = gr.Button("Generate Blind Comparison", variant="primary")
                        blind_reveal = gr.Textbox(
                            label="Reveal (after voting)", 
                            visible=False,
                            interactive=False
                        )
                    
                with gr.Row():
                    blind_left = gr.Image(label="Image A", height=512)
                    blind_right = gr.Image(label="Image B", height=512)
                
                with gr.Row():
                    vote_left = gr.Button("Vote for A", variant="secondary")
                    vote_right = gr.Button("Vote for B", variant="secondary")
                
                def generate_and_hide_reveal(*args):
                    img_a, img_b, reveal_text = generate_blind_comparison(*args)
                    return img_a, img_b, gr.update(value=reveal_text, visible=False)
                
                def show_reveal(*args):
                    return gr.update(visible=True)
                
                blind_button.click(
                    fn=generate_and_hide_reveal,
                    inputs=blind_inputs,
                    outputs=[blind_left, blind_right, blind_reveal]
                )
                
                vote_left.click(fn=show_reveal, outputs=[blind_reveal])
                vote_right.click(fn=show_reveal, outputs=[blind_reveal])
        
        # Examples
        gr.Examples(
            examples=[[prompt, "None", None] for prompt in example_prompts],
            inputs=comp_inputs,
        )
    
    demo.queue(max_size=20)
    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
