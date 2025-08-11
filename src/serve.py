#!/usr/bin/env python

import io
import logging
import random
import time
from pathlib import Path
from typing import Optional

import modal
import torch
from diffusers import FluxPipeline
from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
from PIL import Image

from src.config import (
    DIRECTOR_MAP,
    DIRECTORS_DIR,
    MODEL_DIR,
    InferenceConfig,
    LoRAInfo,
    huggingface_secret,
    volume,
)
from src.images import serving_image
from src.utils import CUSTOM_GRADIO_THEME

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Common configuration
CONTAINER_CACHE_DIR = Path("/cache")
CONTAINER_CACHE_VOLUME = modal.Volume.from_name(
    "flux_lora_cache", create_if_missing=True
)
ENABLE_COMPILATION = False
# Optimized Modal image
image = serving_image(CONTAINER_CACHE_DIR)

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
    gpu="A100-80GB",
    timeout=2000,
    scaledown_window=300,
    max_containers=1,
    volumes={
        "/volume": volume,
        CONTAINER_CACHE_DIR: CONTAINER_CACHE_VOLUME,
    },
    secrets=[huggingface_secret],
    enable_memory_snapshot=True,
)
class FluxServeModel:
    """Optimized Modal class for serving Flux model with LoRA support."""

    enable_compilation: bool = modal.parameter(default=ENABLE_COMPILATION)

    def _optimize_pipeline(
        self, pipeline: FluxPipeline, compile_pipeline: bool = True
    ) -> None:
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
        _ = self.base_pipeline(
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
                any(lora_dir.glob("*.safetensors"))
                or any(lora_dir.glob("adapter_config.json"))
            ):
                self.available_loras[lora_dir.name] = LoRAInfo(
                    name=lora_dir.name,
                    path=str(lora_dir),
                    trigger_phrase=f"<{lora_dir.name}-style>",
                    description=f"{lora_dir.name.title()} style adaptation",
                )

        logger.info(f"Found {len(self.available_loras)} LoRA adapters")

    def _create_shared_component_pipeline(
        self, base_pipeline: FluxPipeline
    ) -> FluxPipeline:
        """Create a pipeline that shares components with the base pipeline.

        This optimization reduces memory usage by sharing the underlying model weights
        between the base and LoRA pipelines, while maintaining separate pipeline
        instances for independent LoRA operations.
        """
        logger.info("Creating pipeline with shared components to optimize memory usage")

        # Create a new pipeline instance that shares the same model components
        # This avoids loading model weights twice while allowing independent LoRA operations
        shared_pipeline = FluxPipeline(
            transformer=base_pipeline.transformer,
            vae=base_pipeline.vae,
            text_encoder=base_pipeline.text_encoder,
            text_encoder_2=base_pipeline.text_encoder_2,
            tokenizer=base_pipeline.tokenizer,
            tokenizer_2=base_pipeline.tokenizer_2,
            scheduler=base_pipeline.scheduler,
        )
        shared_pipeline.to(base_pipeline.device, dtype=base_pipeline.dtype)

        # Verify the shared components are identical (same memory references)
        assert (
            shared_pipeline.transformer is base_pipeline.transformer
        ), "Transformer should be shared"
        assert shared_pipeline.vae is base_pipeline.vae, "VAE should be shared"
        logger.info("Successfully created pipeline with shared components")

        return shared_pipeline

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

        # Create LoRA pipeline sharing components with base pipeline (uncompiled for faster switching)
        logger.info("Creating LoRA pipeline with shared components...")
        self.lora_pipeline = self._create_shared_component_pipeline(self.base_pipeline)

        self._load_mega_cache()

        # Optimize base pipeline with conditional compilation
        if self.enable_compilation:
            logger.info("Optimizing base pipeline (compiled)...")
            self._optimize_pipeline(self.base_pipeline, compile_pipeline=True)
            self._compile_pipeline()
            self._save_mega_cache()
        else:
            logger.info(
                "Optimizing base pipeline (uncompiled - compilation disabled)..."
            )
            self._optimize_pipeline(self.base_pipeline, compile_pipeline=False)

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
            try:
                self.lora_pipeline.unload_lora_weights()
                self.current_lora = None
            except Exception as e:
                logger.warning(f"Failed to unload LoRA {self.current_lora}: {e}")
                self.current_lora = None

        # Load new LoRA if specified
        if lora_name and lora_name in self.available_loras:
            lora_info = self.available_loras[lora_name]
            try:
                self.lora_pipeline.load_lora_weights(lora_info.path)
                self.current_lora = lora_name
                return True
            except Exception as e:
                logger.error(f"Failed to load LoRA {lora_name}: {e}")
                return False

        return False

    def generate_image(
        self, prompt: str, lora_name: Optional[str] = None, seed: Optional[int] = None
    ) -> bytes:
        """Generate image with optional LoRA."""
        if seed is not None:
            torch.manual_seed(seed)

        if lora_name is None:
            pipeline = self.base_pipeline
            lora_loaded = False
        else:
            pipeline = self.lora_pipeline
            lora_loaded = self._load_lora_if_needed(lora_name)

        if lora_loaded and lora_name in self.available_loras:
            lora_info = self.available_loras[lora_name]
            if lora_info.trigger_phrase not in prompt:
                prompt = f"{lora_info.trigger_phrase} {prompt}"

        torch.cuda.synchronize()
        start_time = time.perf_counter()

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

        buffer = io.BytesIO()
        result.images[0].save(buffer, format="PNG")
        return buffer.getvalue()

    @modal.method()
    def generate_comparison(
        self, prompt: str, lora_name: str, seed: Optional[int] = None
    ) -> tuple[bytes, bytes]:
        """Generate comparison images efficiently."""

        base_prompt = prompt
        if lora_name in DIRECTOR_MAP:
            director_name = DIRECTOR_MAP[lora_name]
            style_instruction = f"in the style of {director_name}"

            # Only add if not already mentioned
            if style_instruction.lower() not in prompt.lower():
                base_prompt = f"{prompt}, {style_instruction}"

        # Generate base image first with style instruction
        base_image = self.generate_image(base_prompt, lora_name=None, seed=seed)

        # Generate LoRA image second (LoRA will be loaded only once)
        lora_image = self.generate_image(prompt, lora_name=lora_name, seed=seed)

        return base_image, lora_image

    @modal.method()
    def generate_single(
        self, prompt: str, lora_name: Optional[str] = None, seed: Optional[int] = None
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
    """Create Gradio interface with comparison and blind voting tabs.

    Args:
        enable_compilation: Whether to enable torch compilation for faster inference.
    """

    import gradio as gr
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from gradio.routes import mount_gradio_app

    model = FluxServeModel(enable_compilation=ENABLE_COMPILATION)

    # Director mapping
    DIRECTOR_MAP = {
        "anderson": "Wes Anderson",
        "fincher": "David Fincher",
        "nolan": "Christopher Nolan",
        "scorsese": "Martin Scorsese",
        "villeneuve": "Denis Villeneuve",
    }

    # In-memory leaderboard for votes
    leaderboard = {
        "base_model": 0,
        "anderson": 0,
        "fincher": 0,
        "nolan": 0,
        "scorsese": 0,
        "villeneuve": 0,
    }

    # Common generation function
    def generate_images(
        prompt: str, lora_selection: str, seed: Optional[int] = None
    ) -> tuple[Optional[Image.Image], Optional[Image.Image]]:
        """Generate comparison images."""
        logger.info(
            f"generate_images called with prompt='{prompt}', lora_selection='{lora_selection}', seed={seed}"
        )

        if not prompt.strip():
            logger.warning("Empty prompt provided")
            return None, None

        # For comparison, we always compare against base model
        if not lora_selection or lora_selection not in DIRECTOR_MAP:
            logger.warning("Invalid LoRA selected for comparison")
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
                Image.open(io.BytesIO(lora_bytes)),
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
            lora_name = lora_selection if lora_selection in DIRECTOR_MAP else None
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
        """Get available LoRA options with director names."""
        try:
            loras = model.get_available_loras.remote()
            logger.info(f"Available LoRAs: {list(loras.keys())}")
            # Return director names mapped to their lora keys
            options = []
            for lora_key in DIRECTOR_MAP.keys():
                if lora_key in loras:
                    options.append(lora_key)
            return options
        except Exception as e:
            logger.error(f"Failed to get LoRAs: {e}")
            return list(DIRECTOR_MAP.keys())

    def vote_for_image(choice: str, reveal_text: str) -> tuple[str, str]:
        """Record vote and return updated leaderboard."""
        if not reveal_text:
            return "", ""

        # Parse the reveal text to determine what was voted for
        parts = reveal_text.split(",")
        if len(parts) != 2:
            return "", ""

        left_info = parts[0].split("=")[1]  # "base" or "lora"
        right_info = parts[1].split("=")[1]  # "lora" or "base"

        # Determine what was voted for
        if choice == "left":
            voted_for = left_info
        else:  # choice == "right"
            voted_for = right_info

        # Update leaderboard
        if voted_for == "base":
            leaderboard["base_model"] += 1
        elif voted_for == "lora":
            # Need to know which LoRA was used - this is a limitation
            # For now, we'll assume the last selected LoRA
            pass  # Will be handled in the UI

        return (
            get_leaderboard_display(),
            "Vote recorded! Scroll down to see the leaderboard.",
        )

    def vote_for_lora(lora_name: str) -> None:
        """Record vote for specific LoRA."""
        if lora_name in leaderboard:
            leaderboard[lora_name] += 1

    def get_leaderboard_display() -> str:
        """Get formatted leaderboard display."""
        sorted_scores = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)

        display = "## 🏆 Leaderboard\n\n"
        for i, (model, votes) in enumerate(sorted_scores, 1):
            if model == "base_model":
                name = "Flux Base Model"
            else:
                name = DIRECTOR_MAP.get(model, model.title())

            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🎯"
            display += f"{medal} **{name}**: {votes} votes\n"

        return display

    # Load configuration
    title = "Director-Diffusion: Cinematic Style Generator"

    # Better examples showcasing different directors' styles
    example_prompts = [
        ["A symmetrical hotel lobby with pastel colors", "anderson"],
        ["A dark urban scene with rain and neon reflections", "fincher"],
        ["An epic space battle with dramatic lighting", "nolan"],
        ["A gritty street scene with dynamic camera angles", "scorsese"],
        ["A vast desert landscape with mysterious structures", "villeneuve"],
        ["A cozy cafe interior with warm lighting", "anderson"],
        ["A tense interrogation room with harsh shadows", "fincher"],
        ["A complex maze-like architecture", "nolan"],
    ]

    # Common input components
    def create_inputs():
        lora_options = get_lora_options()
        director_choices = [(DIRECTOR_MAP.get(key, key), key) for key in lora_options]

        return [
            gr.Textbox(
                label="🎬 Prompt",
                placeholder="Describe the scene you want to generate...",
                lines=3,
            ),
            gr.Dropdown(
                label="🎭 Director Style",
                choices=director_choices,
                value="anderson",
                info="Choose a director's visual style for your image",
            ),
        ]

    def create_inputs_with_seed():
        """Create inputs with collapsible seed option."""
        base_inputs = create_inputs()

        with gr.Accordion("⚙️ Advanced Options", open=False):
            seed_input = gr.Number(
                label="Seed (optional)",
                value=None,
                precision=0,
                info="Set a specific seed for reproducible results",
            )

        return base_inputs + [seed_input]

    web_app = FastAPI()

    @web_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("/assets/favicon.ico")

    @web_app.get("/assets/background.svg", include_in_schema=False)
    async def background():
        return FileResponse("/assets/background.svg")

    # Create main interface with favicon support
    with gr.Blocks(
        css=CUSTOM_GRADIO_THEME,
        title=title,
        head='<link rel="icon" href="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHJlY3Qgd2lkdGg9IjMyIiBoZWlnaHQ9IjMyIiByeD0iNCIgZmlsbD0iIzFhMWEyZSIvPgo8Y2lyY2xlIGN4PSIxNiIgY3k9IjE2IiByPSI2IiBzdHJva2U9IiNkNGFmMzciIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0ibm9uZSIvPgo8Y2lyY2xlIGN4PSIxNiIgY3k9IjE2IiByPSIyIiBmaWxsPSIjZDRhZjM3Ii8+CjwvHN2Zz4K">',
    ) as demo:
        # Header
        gr.Markdown(f"# {title}")

        with gr.Tabs():
            # Introduction Tab
            with gr.Tab("🎬 About"):
                gr.Markdown(
                    """
                <div class="director-info">

                ## Welcome to Director-Diffusion!

                *by [Karan Sampath](https://www.karansampath.com) | [Contact](mailto:me@karansampath.com)*

                I built **Director-Diffusion** as an AI image generation tool that creates images in the distinctive visual styles of famous film directors. Using fine-tuned LoRA adapters on the FLUX.1-Krea-dev model, I aimed to capture the unique cinematographic essence of five legendary directors:

                ### 🎭 Available Directors

                - **🎨 Wes Anderson**: Symmetric compositions, pastel colors, whimsical framing, large character groups
                - **🌃 David Fincher**: Dark, moody atmosphere, noir like vignettes, and sharp contrasts
                - **🌀 Christopher Nolan**: Epic scale, complex lighting, scientific realism, and dramatic architectural elements
                - **🎪 Martin Scorsese**: Dynamic camera angles, gritty realism, and vibrant urban scenes
                - **🏜️ Denis Villeneuve**: Vast landscapes, mysterious atmospheres, gritty realism, and epic sci-fi aesthetics

                ### 🚀 How It Works

                1. **Quick Test**: Generate single images to experiment with different director styles
                2. **Comparison**: See side-by-side comparisons of base model vs. director-styled results
                3. **Blind Vote**: Participate in blind comparisons and help build the community leaderboard!

                ### 📊 Dataset
                I trained the models on carefully curated datasets:
                - **Anderson**: 201 images
                - **Fincher**: 214 images
                - **Nolan**: 232 images
                - **Scorsese**: 215 images
                - **Villeneuve**: 197 images

                *Start by choosing a tab above to begin generating cinematic masterpieces!*

                </div>
                """
                )

            # Quick Test Tab
            with gr.Tab("⚡ Quick Test"):
                gr.Markdown(
                    "### Generate single images to test different director styles"
                )

                with gr.Row():
                    with gr.Column():
                        quick_prompt = gr.Textbox(
                            label="🎬 Prompt",
                            placeholder="Describe the scene you want to generate...",
                            lines=3,
                        )
                        lora_options = get_lora_options()
                        director_choices = [
                            (DIRECTOR_MAP.get(key, key), key) for key in lora_options
                        ]
                        quick_director = gr.Dropdown(
                            label="🎭 Director Style",
                            choices=director_choices,
                            value="anderson",
                            info="Choose a director's visual style for your image",
                        )

                        with gr.Accordion("⚙️ Advanced Options", open=False):
                            quick_seed = gr.Number(
                                label="Seed (optional)",
                                value=None,
                                precision=0,
                                info="Set a specific seed for reproducible results",
                            )

                        quick_button = gr.Button(
                            "🎬 Generate Image", variant="primary", size="lg"
                        )

                with gr.Row():
                    quick_output = gr.Image(label="Generated Image", height=512)

                quick_button.click(
                    fn=generate_single_image,
                    inputs=[quick_prompt, quick_director, quick_seed],
                    outputs=[quick_output],
                )

                # Examples for Quick Test tab
                gr.Examples(
                    examples=example_prompts,
                    inputs=[quick_prompt, quick_director],
                    label="🎨 Click any example to try it out!",
                )

            # Comparison Tab
            with gr.Tab("⚖️ Comparison"):
                gr.Markdown("### Compare base FLUX model with director-styled versions")

                with gr.Row():
                    with gr.Column():
                        comp_prompt = gr.Textbox(
                            label="🎬 Prompt",
                            placeholder="Describe the scene you want to generate...",
                            lines=3,
                        )
                        comp_director = gr.Dropdown(
                            label="🎭 Director Style",
                            choices=director_choices,
                            value="anderson",
                            info="Choose a director's visual style to compare against base model",
                        )

                        with gr.Accordion("⚙️ Advanced Options", open=False):
                            comp_seed = gr.Number(
                                label="Seed (optional)",
                                value=None,
                                precision=0,
                                info="Set a specific seed for reproducible results",
                            )

                        comp_button = gr.Button(
                            "⚖️ Generate Comparison", variant="primary", size="lg"
                        )

                with gr.Row():
                    comp_base_out = gr.Image(label="🔹 Base FLUX Model", height=512)
                    comp_lora_out = gr.Image(label="🎭 Director Style", height=512)

                comp_button.click(
                    fn=generate_images,
                    inputs=[comp_prompt, comp_director, comp_seed],
                    outputs=[comp_base_out, comp_lora_out],
                )

                # Examples for Comparison tab
                gr.Examples(
                    examples=example_prompts,
                    inputs=[comp_prompt, comp_director],
                    label="⚖️ Click any example to compare styles!",
                )

            # Blind Voting Tab
            with gr.Tab("🗳️ Blind Vote"):
                gr.Markdown(
                    "### Help us rank the directors! Vote for your preferred image without knowing which is which."
                )

                # State to track current comparison
                current_lora = gr.State("anderson")

                with gr.Row():
                    with gr.Column():
                        blind_prompt = gr.Textbox(
                            label="🎬 Prompt",
                            placeholder="Describe the scene you want to generate...",
                            lines=3,
                        )
                        blind_director = gr.Dropdown(
                            label="🎭 Director Style",
                            choices=director_choices,
                            value="anderson",
                            info="Choose a director's style to compare against base model",
                        )

                        with gr.Accordion("⚙️ Advanced Options", open=False):
                            blind_seed = gr.Number(
                                label="Seed (optional)",
                                value=None,
                                precision=0,
                                info="Set a specific seed for reproducible results",
                            )

                        blind_button = gr.Button(
                            "🎲 Generate Blind Comparison", variant="primary", size="lg"
                        )

                        blind_reveal = gr.Textbox(
                            label="Results", visible=False, interactive=False
                        )

                        vote_status = gr.Textbox(
                            label="Vote Status", visible=False, interactive=False
                        )

                with gr.Row():
                    blind_left = gr.Image(label="🅰️ Image A", height=512)
                    blind_right = gr.Image(label="🅱️ Image B", height=512)

                with gr.Row():
                    vote_left = gr.Button(
                        "🗳️ Vote for A", variant="secondary", size="lg"
                    )
                    vote_right = gr.Button(
                        "🗳️ Vote for B", variant="secondary", size="lg"
                    )

                # Leaderboard display
                leaderboard_display = gr.Markdown(
                    get_leaderboard_display(), label="Leaderboard"
                )

                def generate_and_track_blind(
                    prompt: str, lora_selection: str, seed: Optional[int] = None
                ):
                    """Generate blind comparison images and track the LoRA used."""
                    img_a, img_b, reveal_text = generate_blind_comparison(
                        prompt, lora_selection, seed
                    )
                    return (
                        img_a,
                        img_b,
                        gr.update(value=reveal_text, visible=False),
                        lora_selection,
                    )

                def vote_and_update(
                    choice: str, reveal_text: str, current_lora_name: str
                ):
                    """Process vote and update leaderboard."""
                    if not reveal_text:
                        return (
                            gr.update(),
                            gr.update(
                                value="Please generate images first!", visible=True
                            ),
                            get_leaderboard_display(),
                        )

                    # Parse reveal text
                    parts = reveal_text.split(",")
                    if len(parts) != 2:
                        return (
                            gr.update(),
                            gr.update(value="Error processing vote!", visible=True),
                            get_leaderboard_display(),
                        )

                    left_info = parts[0].split("=")[1]  # "base" or "lora"
                    right_info = parts[1].split("=")[1]  # "lora" or "base"

                    # Determine what was voted for
                    if choice == "left":
                        voted_for = left_info
                    else:  # choice == "right"
                        voted_for = right_info

                    # Update leaderboard
                    if voted_for == "base":
                        leaderboard["base_model"] += 1
                        result_msg = "You voted for the Base FLUX model!"
                    elif voted_for == "lora":
                        leaderboard[current_lora_name] += 1
                        director_name = DIRECTOR_MAP.get(
                            current_lora_name, current_lora_name
                        )
                        result_msg = f"You voted for {director_name}'s style!"
                    else:
                        result_msg = "Vote processing error!"

                    return (
                        gr.update(value=reveal_text, visible=True),
                        gr.update(value=result_msg, visible=True),
                        get_leaderboard_display(),
                    )

                blind_button.click(
                    fn=generate_and_track_blind,
                    inputs=[blind_prompt, blind_director, blind_seed],
                    outputs=[blind_left, blind_right, blind_reveal, current_lora],
                )

                vote_left.click(
                    fn=lambda reveal_text, current_lora_name: vote_and_update(
                        "left", reveal_text, current_lora_name
                    ),
                    inputs=[blind_reveal, current_lora],
                    outputs=[blind_reveal, vote_status, leaderboard_display],
                )

                vote_right.click(
                    fn=lambda reveal_text, current_lora_name: vote_and_update(
                        "right", reveal_text, current_lora_name
                    ),
                    inputs=[blind_reveal, current_lora],
                    outputs=[blind_reveal, vote_status, leaderboard_display],
                )

                # Examples for Blind Vote tab
                gr.Examples(
                    examples=example_prompts,
                    inputs=[blind_prompt, blind_director],
                    label="🗳️ Click any example for blind voting!",
                )

    demo.queue(max_size=20)
    return mount_gradio_app(app=web_app, blocks=demo, path="/")
