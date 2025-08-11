#!/usr/bin/env python

from dataclasses import dataclass
from typing import Optional

import modal

# Modal infrastructure
volume = modal.Volume.from_name("director-diffusion", create_if_missing=True)
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)
wandb_secret = modal.Secret.from_name("wandb-secret", required_keys=["WANDB_API_KEY"])

# Directory paths
MODEL_DIR = "/volume/flux-krea"
DIRECTORS_DIR = "/volume/director_loras"

DIRECTOR_MAP = {
    "anderson": "Wes Anderson",
    "fincher": "David Fincher",
    "nolan": "Christopher Nolan",
    "scorsese": "Martin Scorsese",
    "villeneuve": "Denis Villeneuve",
}


@dataclass
class EvaluationConfig:
    """Configuration for LoRA vs Base model evaluation."""

    # Test prompts for each director style
    test_prompts: Optional[list[str]] = None

    # Number of images to generate per prompt/director combo
    num_samples_per_test: int = 3

    # Inference settings for evaluation
    eval_num_inference_steps: int = 20  # Faster for evaluation
    eval_guidance_scale: float = 3.5
    eval_resolution: int = 1024

    # Seeds for reproducible evaluation
    eval_seeds: Optional[list[int]] = None

    # Metrics to compute
    compute_clip_similarity: bool = True
    compute_aesthetic_score: bool = True
    compute_style_consistency: bool = True
    compute_prompt_adherence: bool = True

    # Cost optimization
    use_smaller_clip_model: bool = True  # Use ViT-B/32 instead of ViT-L/14
    batch_evaluation: bool = True

    def __post_init__(self):
        if self.test_prompts is None:
            self.test_prompts = [
                "a cinematic portrait of a person in dramatic lighting",
                "a symmetrical architectural interior",
                "a moody urban landscape at night",
                "a minimalist still life composition",
                "a wide shot of a futuristic cityscape",
                "a close-up of hands holding an object",
                "a person walking down a long corridor",
                "a group of people in formal attire",
            ]

        if self.eval_seeds is None:
            self.eval_seeds = [42, 123, 456, 789, 999]


@dataclass
class GradioConfig:
    title: str = "Flux LoRA Comparison"
    description: str = "Compare your LoRA finetuned Flux model with the base model"
    theme: str = "Soft"  # Gradio theme
    allow_flagging: str = "never"
    show_error: bool = True

    # Example prompts for the interface
    example_prompts: Optional[list[str]] = None

    def __post_init__(self):
        if self.example_prompts is None:
            self.example_prompts = [
                "a cinematic landscape at sunset",
                "a portrait of a person in dramatic lighting",
                "an architectural interior with modern design",
                "a still life with vintage objects",
                "a cyberpunk cityscape with neon lights",
                "a minimalist geometric composition",
            ]


@dataclass
class InferenceConfig:
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    height: int = 1024
    width: int = 1024
    max_sequence_length: int = 512

    # Performance settings
    enable_cpu_offload: bool = True
    enable_xformers: bool = True
    cache_pipelines: bool = True


@dataclass
class LoRAInfo:
    name: str
    path: str
    trigger_phrase: str
    description: str
