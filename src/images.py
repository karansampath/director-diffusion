"""Shared Modal image definitions to eliminate duplication across services."""

from pathlib import Path

import modal

# Common environment variables
_BASE_ENV = {
    "HF_HUB_ENABLE_HF_TRANSFER": "1",
    "TORCH_CUDNN_V8_API_ENABLED": "1",
}

# Core ML packages used across services
_CORE_ML_PACKAGES = [
    "torch>=2.7.0",
    "torchvision",
    "transformers==4.51.3",
    "accelerate==1.6.0",
    "diffusers==0.33.1",
    "peft>=0.8.0",
    "huggingface-hub[hf_transfer]",
    "pillow",
    "numpy",
]


def _cuda_base():
    """Base CUDA image with common ML packages."""
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04",
            add_python="3.12",
        )
        .entrypoint([])
        .uv_pip_install(*_CORE_ML_PACKAGES)
        .env(_BASE_ENV)
    )


def training_image():
    return _cuda_base().uv_pip_install(
        "datasets",
        "wandb",
        "smart_open",
        "optimum",
        "xformers",
        "triton",
        "ftfy",
        "rich",
        "sentencepiece",
    )


def serving_image(cache_dir: Path):
    return (
        _cuda_base()
        .uv_pip_install(
            "fastapi[standard]",
            "gradio~=5.7.1",
            "pydantic==2.10.6",
            "para-attn==0.3.32",
            "safetensors==0.5.3",
            "sentencepiece",
        )
        .env(
            {
                **_BASE_ENV,
                "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
                "CUDA_CACHE_PATH": str(cache_dir / ".nv_cache"),
                "HF_HUB_CACHE": str(cache_dir / ".hf_hub_cache"),
                "TORCHINDUCTOR_CACHE_DIR": str(cache_dir / ".inductor_cache"),
                "TRITON_CACHE_DIR": str(cache_dir / ".triton_cache"),
            }
        )
    )


def evaluation_image():
    return _cuda_base().uv_pip_install(
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pandas",
        "open-clip-torch",
    )


def caption_image():
    return (
        modal.Image.debian_slim()
        .uv_pip_install(
            "torch",
            "torchvision",
            "transformers",
            "huggingface-hub[hf_transfer]",
            "accelerate",
            "numpy",
        )
        .env(
            {
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        .add_local_dir("src", "/src", copy=True)
    )


def simple_image(*packages):
    return modal.Image.debian_slim().uv_pip_install(*packages)
