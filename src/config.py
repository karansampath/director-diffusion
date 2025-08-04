#!/usr/bin/env python
"""
Configuration module for Flux LoRA Gradio app.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class GradioConfig:
    """Configuration for Gradio app settings."""
    
    title: str = "Flux LoRA Comparison"
    description: str = "Compare your LoRA finetuned Flux model with the base model"
    theme: str = "Soft"  # Gradio theme
    allow_flagging: str = "never"
    show_error: bool = True
    
    # Example prompts for the interface
    example_prompts: List[str] = None
    
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
    """Configuration for inference parameters."""
    
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
    """Information about a LoRA adapter."""
    
    name: str
    path: str
    trigger_phrase: str
    description: str