#!/usr/bin/env python

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List
import json
import asyncio
from contextlib import nullcontext
from tqdm import tqdm

import modal
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from accelerate import Accelerator
from diffusers import FluxPipeline
from peft import LoraConfig, get_peft_model_state_dict
import logging
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
)
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import CLIPTokenizer, T5TokenizerFast
import copy

# Enhanced image with performance optimizations
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "accelerate",
        "datasets",
        "diffusers",
        "sentencepiece",
        "rich",
        "ftfy",
        "huggingface-hub[hf_transfer]",
        "numpy",
        "peft>=0.8.0",
        "torch>=2.1.0",
        "torchvision",
        "triton",
        "wandb",
        "peft",
        "smart_open",
        "optimum",
        "xformers",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TORCH_CUDNN_V8_API_ENABLED": "1",
        }
    )
)

volume = modal.Volume.from_name("director-diffusion", create_if_missing=True)
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)
wandb_secret = modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])

app = modal.App(
    name="multi-director-flux-train",
    image=image,
    volumes={"/volume": volume},
    secrets=[huggingface_secret, wandb_secret],
)

MODEL_DIR = "/volume/models"
DIRECTORS_DIR = "/volume/trained_loras"


@dataclass
class DirectorConfig:
    """Configuration for a single director's training."""

    name: str
    style_description: str
    data_path: str
    trigger_phrase: str

    # Training hyperparameters per director
    learning_rate: float = 4e-4
    max_train_steps: int = 500
    rank: int = 16
    lora_alpha: int = 16


@dataclass
class MultiDirectorConfig:
    """Configuration for multi-director training pipeline."""

    directors: List[DirectorConfig] = field(
        default_factory=lambda: [
            DirectorConfig(
                name="anderson",
                style_description="Wes Anderson cinematic style",
                data_path="/volume/data/anderson",
                trigger_phrase="<anderson-style>",
            )
        ]
    )

    captions_dataset_path: str = "/volume/labels/captions.json"

    base_model: str = "black-forest-labs/FLUX.1-dev"
    resolution: int = 1024
    train_batch_size: int = 2
    gradient_accumulation_steps: int = 4

    enable_torch_compile: bool = False
    enable_xformers: bool = True
    enable_gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"
    use_tf32: bool = True
    cache_latents: bool = True

    sequential_training: bool = True
    shared_text_encoder: bool = False

    num_validation_images: int = 4
    validation_epochs: int = 50


class DirectorDataset(Dataset):
    """Dataset that loads from caption dataset and filters by director."""

    def __init__(
        self,
        caption_path,
        director_name,
        trigger_phrase,
        resolution=1024,
        cache_latents=True,
        vae=None,
    ):
        self.caption_path = Path(caption_path)
        self.director_name = director_name
        self.trigger_phrase = trigger_phrase
        self.resolution = resolution
        self.cache_latents = cache_latents
        self.vae = vae

        # Load and filter data for this director
        self.load_director_data()

        # Setup transforms
        self.setup_transforms()

        # Pre-cache latents if enabled
        if cache_latents and vae is not None:
            self.cache_image_latents()

    def load_director_data(self):
        """Load and filter data for specific director from main caption dataset."""

        logging.info(f"Loading data for director: {self.director_name}")

        # Load main caption dataset
        with open(self.caption_path, "r") as f:
            all_captions = json.load(f)

        # Filter for this director's images
        self.items = []
        director_path_pattern = f"/volume/data/{self.director_name}/"

        for item in all_captions:
            if director_path_pattern in item["file_name"]:
                # Convert absolute path to be accessible in container
                image_path = Path(item["file_name"])

                # Verify image exists
                if image_path.exists():
                    self.items.append(
                        {
                            "image_path": image_path,
                            "caption": item[
                                "text"
                            ],  # Caption already has trigger phrase
                        }
                    )
                else:
                    logging.warning(f"Image not found: {image_path}")

        logging.info(
            f"Loaded {len(self.items)} images for director {self.director_name}"
        )

        if len(self.items) == 0:
            raise ValueError(
                f"No images found for director {self.director_name} in {self.caption_path}"
            )

    def setup_transforms(self):
        """Setup optimized image transforms."""

        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    self.resolution, interpolation=transforms.InterpolationMode.LANCZOS
                ),
                transforms.CenterCrop(self.resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def cache_image_latents(self):
        """Pre-cache VAE latents for faster training."""

        logging.info(f"Caching latents for {len(self.items)} images...")
        self.cached_latents = []

        with torch.no_grad():
            for item in self.items:
                image = Image.open(item["image_path"]).convert("RGB")
                image_tensor = self.transforms(image).unsqueeze(0)

                # Encode with VAE
                latent = self.vae.encode(
                    image_tensor.to(self.vae.device, dtype=self.vae.dtype)
                ).latent_dist.sample()

                # Handle potential extra dimensions
                if len(latent.shape) == 5 and latent.shape[1] == 1:
                    # If shape is [1, 1, channels, height, width], squeeze the extra dimension
                    latent = latent.squeeze(1)
                    print(f"Squeezed latent shape: {latent.shape}")

                self.cached_latents.append(latent.cpu())

        logging.info("Latent caching completed!")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        if self.cache_latents and hasattr(self, "cached_latents"):
            # Use pre-cached latents
            latent = self.cached_latents[idx]
            return {
                "latents": latent,
                "caption": item["caption"],
            }
        else:
            # Load and transform image on-the-fly
            image = Image.open(item["image_path"]).convert("RGB")
            image_tensor = self.transforms(image)

            return {
                "pixel_values": image_tensor,
                "caption": item["caption"],
            }


class FluxTrainer:
    """High-performance multi-director FLUX LoRA trainer."""

    def __init__(self, config: MultiDirectorConfig):
        self.config = config
        self.accelerator = None
        self.base_pipeline = None
        self.transformer = None
        self.vae = None
        self.text_encoders = None

    def setup_accelerator(self):
        """Initialize accelerator with performance optimizations."""
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with="wandb",
            project_dir="/tmp/accelerate_logs",
        )

        self.accelerator.init_trackers(
            project_name="director-diffusion",
            config=asdict(self.config),
        )

        # Performance settings
        if self.config.use_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Enable optimized attention
        if self.config.enable_xformers:
            try:
                import xformers

                torch.backends.cuda.enable_flash_sdp(True)
            except ImportError:
                logging.warning("xformers not available, using default attention")

    def load_base_models(self):
        """Load base FLUX models and setup text encoders."""
        logging.info("Loading base FLUX models...")

        # Load pipeline
        self.base_pipeline = FluxPipeline.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.bfloat16,
            variant="fp16" if self.config.mixed_precision == "fp16" else None,
        )

        # Extract components
        self.transformer = self.base_pipeline.transformer
        self.vae = self.base_pipeline.vae
        self.text_encoders = [
            self.base_pipeline.text_encoder,
            self.base_pipeline.text_encoder_2,
        ]

        # Setup text encoders and tokenizers
        self.setup_text_encoders_and_tokenizers()

        # Move to device
        device = self.accelerator.device
        weight_dtype = (
            torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16
        )

        self.vae.to(device, dtype=weight_dtype)
        self.transformer.to(device, dtype=weight_dtype)
        for encoder in self.text_encoders:
            encoder.to(device, dtype=weight_dtype)

        # Freeze base model parameters
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)
        for encoder in self.text_encoders:
            encoder.requires_grad_(False)

        # Enable gradient checkpointing for memory efficiency
        if self.config.enable_gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        # Compile models for performance
        if self.config.enable_torch_compile:
            try:
                self.transformer = torch.compile(
                    self.transformer,
                    mode="max-autotune",
                    dynamic=False,
                )
                logging.info("Successfully compiled transformer")
            except Exception as e:
                logging.warning(f"Failed to compile transformer: {e}")

    def create_director_lora(self, director_config: DirectorConfig) -> str:
        """Create and configure LoRA adapter for a director."""

        target_modules = [
            "single_transformer_blocks.0.attn.to_q",
            "single_transformer_blocks.0.attn.to_k",
            "single_transformer_blocks.0.attn.to_v",
            "single_transformer_blocks.0.attn.to_out.0",
            "single_transformer_blocks.0.linear1",
            "single_transformer_blocks.0.linear2",
        ]

        # Create LoRA config with corrected target modules
        lora_config = LoraConfig(
            r=director_config.rank,
            lora_alpha=director_config.lora_alpha,
            lora_dropout=0.1,
            init_lora_weights="gaussian",
            target_modules=target_modules,
            modules_to_save=None,
        )

        # Add adapter with director-specific name
        adapter_name = f"director_{director_config.name}"
        self.transformer.add_adapter(lora_config, adapter_name=adapter_name)

        return adapter_name

    async def train_director_async(self, director_config: DirectorConfig) -> str:
        """Train LoRA adapter for a single director asynchronously."""

        logging.info(f"Starting training for director: {director_config.name}")

        # Create director-specific LoRA adapter
        adapter_name = self.create_director_lora(director_config)

        # Activate this director's adapter
        self.transformer.set_adapter(adapter_name)

        # Load director's dataset with optimizations
        dataset = await self.load_director_dataset(director_config)

        # Create optimized dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
        )

        # Setup optimizer for this director
        optimizer = self.setup_optimizer(director_config)

        # Prepare with accelerator
        dataloader, optimizer = self.accelerator.prepare(dataloader, optimizer)

        # Training loop with optimizations
        await self.training_loop(director_config, dataloader, optimizer, adapter_name)

        # Save director's LoRA weights
        output_path = f"{DIRECTORS_DIR}/{director_config.name}"
        self.save_director_lora(adapter_name, output_path)

        logging.info(f"Completed training for director: {director_config.name}")
        return output_path

    async def load_director_dataset(self, director_config: DirectorConfig):
        """Load and preprocess dataset for a director with optimizations."""

        dataset = DirectorDataset(
            caption_path=self.config.captions_dataset_path,
            director_name=director_config.name,
            trigger_phrase=director_config.trigger_phrase,
            resolution=self.config.resolution,
            cache_latents=self.config.cache_latents,
            vae=self.vae if self.config.cache_latents else None,
        )

        return dataset

    def setup_optimizer(self, director_config: DirectorConfig):
        """Setup optimized optimizer for director training."""

        # Get trainable parameters for current adapter
        trainable_params = [p for p in self.transformer.parameters() if p.requires_grad]

        # Use AdamW with optimized settings
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=director_config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        )

        return optimizer

    async def training_loop(
        self, director_config: DirectorConfig, dataloader, optimizer, adapter_name: str
    ):
        """Optimized training loop for a director."""

        global_step = 0

        # Enable autocast for mixed precision
        autocast_ctx = (
            torch.autocast(
                device_type=self.accelerator.device.type,
                dtype=torch.bfloat16
                if self.config.mixed_precision == "bf16"
                else torch.float16,
            )
            if self.config.mixed_precision != "no"
            else nullcontext()
        )

        for _ in tqdm(range(1000)):
            for batch in tqdm(dataloader):
                with self.accelerator.accumulate(self.transformer):
                    with autocast_ctx:
                        # Forward pass with optimizations
                        loss = await self.compute_loss(batch)

                    # Backward pass
                    self.accelerator.backward(loss)

                    # Gradient clipping
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.transformer.parameters(), max_norm=1.0
                        )

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)  # More memory efficient

                if self.accelerator.sync_gradients:
                    global_step += 1

                    # Send metrics to wandb
                    self.accelerator.log(
                        {"train/loss": loss.item()},
                        step=global_step,
                    )

                    # Console print
                    if global_step % 10 == 0:
                        logging.info(
                            f"Director {director_config.name} "
                            f"- Step {global_step}, Loss: {loss.item():.4f}"
                        )

                    # Validation
                    if global_step % self.config.validation_epochs == 0:
                        await self.validate_director(director_config, adapter_name)

                    if global_step >= director_config.max_train_steps:
                        return
            print(f"Global step: {global_step}")

    async def compute_loss(self, batch):
        """Compute flow matching loss with optimizations."""

        device = self.accelerator.device
        weight_dtype = (
            torch.bfloat16 if self.config.mixed_precision == "bf16" else torch.float16
        )

        # Get captions
        prompts = (
            batch["caption"]
            if isinstance(batch["caption"], list)
            else [batch["caption"]]
        )

        # Encode text prompts
        prompt_embeds, pooled_prompt_embeds, text_ids = self.compute_text_embeddings(
            self.text_encoders, self.tokenizers, prompts, max_sequence_length=512
        )

        # Get model input (latents)
        if "latents" in batch:
            # Use pre-cached latents
            model_input = batch["latents"].to(device, dtype=weight_dtype)
            # Apply VAE scaling
            vae_config_shift_factor = self.vae.config.shift_factor
            vae_config_scaling_factor = self.vae.config.scaling_factor
            model_input = (
                model_input - vae_config_shift_factor
            ) * vae_config_scaling_factor
        else:
            # Encode images on-the-fly
            pixel_values = batch["pixel_values"].to(device, dtype=self.vae.dtype)
            model_input = self.vae.encode(pixel_values).latent_dist.sample()
            # Apply VAE scaling
            vae_config_shift_factor = self.vae.config.shift_factor
            vae_config_scaling_factor = self.vae.config.scaling_factor
            model_input = (
                model_input - vae_config_shift_factor
            ) * vae_config_scaling_factor
            model_input = model_input.to(dtype=weight_dtype)

        # Handle unexpected tensor dimensions - squeeze if needed
        if len(model_input.shape) == 5:
            # If shape is [batch, 1, channels, height, width], squeeze the extra dimension
            model_input = model_input.squeeze(1)
        elif len(model_input.shape) != 4:
            raise ValueError(
                f"Unexpected model input shape: {model_input.shape}. Expected 4D tensor [batch, channels, height, width]"
            )

        # Sample noise and timesteps
        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        # Sample timesteps using density-based sampling
        u = compute_density_for_timestep_sampling(
            weighting_scheme="uniform",
            batch_size=bsz,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler_copy.timesteps[indices].to(
            device=model_input.device
        )

        # Add noise to the model input according to the flow matching formulation
        sigmas = self.get_sigmas(
            timesteps, n_dim=model_input.ndim, dtype=model_input.dtype
        )
        noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

        # Pack latents for FLUX transformer
        # Get actual dimensions from the tensor
        batch_size, num_channels, height, width = noisy_model_input.shape

        packed_noisy_model_input = self._pack_latents(
            noisy_model_input,
            batch_size=batch_size,
            num_channels_latents=num_channels,
            height=height,
            width=width,
        )

        # Prepare image IDs using latent dimensions (not original image dimensions)
        img_ids = self._prepare_latent_image_ids(
            height=height * 8,  # Convert back to original image size for ID calculation
            width=width * 8,  # Convert back to original image size for ID calculation
        )
        guidance = torch.full((batch_size,), 3.5, device=device, dtype=weight_dtype)

        # Predict the noise residual
        model_pred = self.transformer(
            hidden_states=packed_noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            guidance=guidance,
            txt_ids=text_ids,
            img_ids=img_ids.to(device=device, dtype=weight_dtype),
            return_dict=False,
        )[0]

        # Unpack the model prediction
        model_pred = self._unpack_latents(model_pred, height=height, width=width)

        # Compute loss using flow matching target
        target = noise - model_input
        loss = torch.nn.functional.mse_loss(
            model_pred.float(), target.float(), reduction="mean"
        )

        return loss

    def _pack_latents(self, latents, batch_size, num_channels_latents, height, width):
        """Pack latents for FLUX transformer input."""
        # FLUX packing: convert from (B, 16, H, W) to (B, H*W/4, 64)
        # FLUX uses 16 input channels and packs them to 64 channels
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )
        return latents

    def _unpack_latents(self, latents, height, width):
        """Unpack latents from FLUX transformer output."""
        batch_size, num_patches, channels = latents.shape
        # Calculate the correct unpacked dimensions
        patch_size = 2
        latent_height = height
        latent_width = width

        # FLUX unpacking: convert from (B, H*W/4, 64) back to (B, 16, H, W)
        channels_per_patch = channels // 4  # 64 // 4 = 16
        patches_per_dim_h = latent_height // patch_size
        patches_per_dim_w = latent_width // patch_size

        latents = latents.view(
            batch_size,
            patches_per_dim_h,
            patches_per_dim_w,
            channels_per_patch,
            patch_size,
            patch_size,
        )
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(
            batch_size, channels_per_patch, latent_height, latent_width
        )
        return latents

    def _prepare_latent_image_ids(self, height, width):
        """Prepare image IDs for FLUX transformer as a 2D tensor."""
        # Calculate latent dimensions (VAE downscales by 8x)
        latent_height = height // 8
        latent_width = width // 8

        # Create position IDs for patches (2x2 packing)
        patch_height = latent_height // 2
        patch_width = latent_width // 2

        latent_image_ids = torch.zeros(patch_height, patch_width, 3)
        latent_image_ids[..., 1] = (
            latent_image_ids[..., 1] + torch.arange(patch_height)[:, None]
        )
        latent_image_ids[..., 2] = (
            latent_image_ids[..., 2] + torch.arange(patch_width)[None, :]
        )

        # Flatten to sequence
        latent_image_ids = latent_image_ids.reshape(patch_height * patch_width, 3)

        return latent_image_ids

    async def validate_director(
        self, director_config: DirectorConfig, adapter_name: str
    ):
        """Run validation for a director's training."""

        # Switch to evaluation mode
        self.transformer.eval()

        # Generate validation images
        prompt = f"a cinematic scene {director_config.trigger_phrase}"

        with torch.no_grad():
            # Generate validation images using the pipeline
            pipeline = FluxPipeline.from_pretrained(
                MODEL_DIR,
                transformer=self.transformer,
                torch_dtype=torch.bfloat16,
            )
            pipeline.to(self.accelerator.device)

            # Generate multiple validation images
            validation_images = []
            for _ in range(self.config.num_validation_images):
                image = pipeline(
                    prompt=prompt,
                    height=self.config.resolution,
                    width=self.config.resolution,
                    num_inference_steps=20,
                    guidance_scale=3.5,
                ).images[0]
                validation_images.append(image)

            # Log to wandb if available
            try:
                import wandb

                wandb.log(
                    {
                        f"validation/{adapter_name}": [
                            wandb.Image(img, caption=f"{prompt} - {i}")
                            for i, img in enumerate(validation_images)
                        ]
                    }
                )
            except ImportError:
                logging.warning("wandb not available for validation logging")

            # Clean up pipeline to free memory
            del pipeline
            torch.cuda.empty_cache()

        # Switch back to training mode
        self.transformer.train()

    def save_director_lora(self, adapter_name: str, output_path: str):
        """Save director's LoRA weights with metadata."""

        Path(output_path).mkdir(parents=True, exist_ok=True)

        # Get adapter state dict
        lora_state_dict = get_peft_model_state_dict(
            self.transformer, adapter_name=adapter_name
        )

        # Save with FluxPipeline format
        FluxPipeline.save_lora_weights(
            save_directory=output_path,
            transformer_lora_layers=lora_state_dict,
        )

        logging.info(f"Saved LoRA weights to {output_path}")

    def setup_text_encoders_and_tokenizers(self):
        """Setup text encoders and tokenizers."""
        # Load tokenizers
        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            MODEL_DIR,
            subfolder="tokenizer",
        )
        self.tokenizer_two = T5TokenizerFast.from_pretrained(
            MODEL_DIR,
            subfolder="tokenizer_2",
        )

        # Tokenizers list for convenience
        self.tokenizers = [self.tokenizer_one, self.tokenizer_two]

        # Setup noise scheduler
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            MODEL_DIR, subfolder="scheduler"
        )
        self.noise_scheduler_copy = copy.deepcopy(self.noise_scheduler)

    def tokenize_prompt(self, tokenizer, prompt, max_sequence_length):
        """Tokenize prompt."""
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids

    def _encode_prompt_with_t5(
        self,
        text_encoder,
        tokenizer,
        max_sequence_length=512,
        prompt=None,
        num_images_per_prompt=1,
        device=None,
        text_input_ids=None,
    ):
        """Encode prompt with T5 encoder."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_length=False,
                return_overflowing_tokens=False,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError(
                    "text_input_ids must be provided when the tokenizer is not specified"
                )

        prompt_embeds = text_encoder(text_input_ids.to(device))[0]

        if hasattr(text_encoder, "module"):
            dtype = text_encoder.module.dtype
        else:
            dtype = text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )

        return prompt_embeds

    def _encode_prompt_with_clip(
        self,
        text_encoder,
        tokenizer,
        prompt: str,
        device=None,
        text_input_ids=None,
        num_images_per_prompt: int = 1,
    ):
        """Encode prompt with CLIP encoder."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if tokenizer is not None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_overflowing_tokens=False,
                return_length=False,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        else:
            if text_input_ids is None:
                raise ValueError(
                    "text_input_ids must be provided when the tokenizer is not specified"
                )

        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=False
        )

        if hasattr(text_encoder, "module"):
            dtype = text_encoder.module.dtype
        else:
            dtype = text_encoder.dtype

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        text_encoders,
        tokenizers,
        prompt: str,
        max_sequence_length,
        device=None,
        num_images_per_prompt: int = 1,
        text_input_ids_list=None,
    ):
        """Main prompt encoding function."""
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if hasattr(text_encoders[0], "module"):
            dtype = text_encoders[0].module.dtype
        else:
            dtype = text_encoders[0].dtype

        pooled_prompt_embeds = self._encode_prompt_with_clip(
            text_encoder=text_encoders[0],
            tokenizer=tokenizers[0],
            prompt=prompt,
            device=device if device is not None else text_encoders[0].device,
            num_images_per_prompt=num_images_per_prompt,
            text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
        )

        prompt_embeds = self._encode_prompt_with_t5(
            text_encoder=text_encoders[1],
            tokenizer=tokenizers[1],
            max_sequence_length=max_sequence_length,
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device if device is not None else text_encoders[1].device,
            text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        return prompt_embeds, pooled_prompt_embeds, text_ids

    def compute_text_embeddings(
        self, text_encoders, tokenizers, prompts, max_sequence_length
    ):
        """Compute text embeddings for prompts."""
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
                text_encoders,
                tokenizers,
                prompts,
                max_sequence_length=max_sequence_length,
            )
            prompt_embeds = prompt_embeds.to(self.accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.accelerator.device)
            text_ids = text_ids.to(self.accelerator.device)
            return prompt_embeds, pooled_prompt_embeds, text_ids

    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        """Get sigmas for flow matching."""
        sigmas = self.noise_scheduler_copy.sigmas.to(
            device=self.accelerator.device, dtype=dtype
        )
        schedule_timesteps = self.noise_scheduler_copy.timesteps.to(
            self.accelerator.device
        )
        timesteps = timesteps.to(self.accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma


@app.function(
    gpu="H200",
    timeout=14400,
    memory=32768,
    cpu=8,
)
async def train_multi_director(config_dict: dict):
    """Main training function for multiple directors."""

    config = MultiDirectorConfig(**config_dict)
    trainer = FluxTrainer(config)

    # Setup
    trainer.setup_accelerator()
    trainer.load_base_models()

    if config.sequential_training:
        # Train directors sequentially for memory efficiency
        for director_config in config.directors:
            await trainer.train_director_async(director_config)
    else:
        # Train directors in parallel (requires more memory)
        tasks = [
            trainer.train_director_async(director_config)
            for director_config in config.directors
        ]
        await asyncio.gather(*tasks)

    logging.info("Multi-director training completed!")
    volume.commit()


@app.function(
    gpu="H200",
    timeout=1800,  # 30 minutes for model download
)
def download_flux_model():
    """Download FLUX.1-dev model to volume storage."""
    import torch
    from diffusers import FluxPipeline
    from huggingface_hub import snapshot_download
    from pathlib import Path

    model_name = "black-forest-labs/FLUX.1-dev"

    print(f"🚀 Downloading FLUX model: {model_name}")
    print(f"📁 Target directory: {MODEL_DIR}")

    # Create model directory if it doesn't exist
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    # Download model using snapshot_download for better control
    snapshot_download(
        model_name,
        local_dir=MODEL_DIR,
        ignore_patterns=["*.pt", "*.bin"],  # Use safetensors for efficiency
    )

    # Verify the model loads correctly
    print("🔍 Verifying model installation...")
    FluxPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16)
    print("✅ FLUX model successfully downloaded and verified!")

    # Commit the volume to persist the model
    volume.commit()
    return f"Model downloaded to {MODEL_DIR}"


@app.local_entrypoint()
def main():
    """Main entry point for multi-director training."""

    print("🔄 Step 1: Installing FLUX model...")
    download_flux_model.remote()
    with modal.enable_output(show_progress=True):
        print("🔄 Step 2: Starting multi-director training...")
        config = MultiDirectorConfig()
        train_multi_director.remote(config.__dict__)
