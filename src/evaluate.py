#!/usr/bin/env python
import json
import logging
from dataclasses import asdict
from pathlib import Path

import modal
import numpy as np
import open_clip
import torch
from diffusers import FluxPipeline
from PIL import Image

from src.config import (
    DIRECTOR_MAP,
    DIRECTORS_DIR,
    MODEL_DIR,
    EvaluationConfig,
    huggingface_secret,
    volume,
    wandb_secret,
)
from src.images import evaluation_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Optimized Modal image for evaluation
eval_image = evaluation_image()

app = modal.App(
    name="director-diffusion-eval",
    image=eval_image,
    volumes={"/volume": volume},
    secrets=[huggingface_secret, wandb_secret],
)


@app.cls(gpu="H100", timeout=3600)  # 1 hour timeout for full evaluation
class DirectorEvaluator:
    """Comprehensive evaluator for LoRA vs base model comparison."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.base_pipeline = None
        self.clip_model = None
        self.clip_preprocess = None
        self.clip_tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")

    @modal.enter()
    def setup(self):
        """Initialize models and evaluation tools."""
        logger.info("Setting up evaluation environment...")

        try:
            # Load base Flux model
            logger.info(f"Loading base Flux pipeline from {MODEL_DIR}...")
            self.base_pipeline = FluxPipeline.from_pretrained(
                MODEL_DIR,
                torch_dtype=torch.bfloat16,
                use_safetensors=True,
            ).to(self.device)
            logger.info("Base Flux pipeline loaded successfully")

            # Load CLIP model for similarity evaluation
            logger.info("Loading CLIP model...")
            model_name = (
                "ViT-B-32" if self.config.use_smaller_clip_model else "ViT-L-14"
            )
            (
                self.clip_model,
                _,
                self.clip_preprocess,
            ) = open_clip.create_model_and_transforms(
                model_name, pretrained="openai", device=self.device
            )
            self.clip_tokenizer = open_clip.get_tokenizer(model_name)
            logger.info("CLIP model loaded successfully")

            logger.info("Evaluation environment ready!")

        except Exception as e:
            logger.error(f"Failed to setup evaluation environment: {e}")
            raise

    def generate_comparison_batch(
        self, prompt: str, director: str, seeds: list[int]
    ) -> tuple[list[Image.Image], list[Image.Image]]:
        """Generate base and LoRA images for comparison."""

        if self.base_pipeline is None:
            raise RuntimeError("Base pipeline not initialized. Call setup() first.")

        base_images = []
        lora_images = []

        # Generate base model images
        logger.info(f"Generating base images for: {prompt}")
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)

            base_image = self.base_pipeline(
                prompt=prompt,
                num_inference_steps=self.config.eval_num_inference_steps,
                guidance_scale=self.config.eval_guidance_scale,
                height=self.config.eval_resolution,
                width=self.config.eval_resolution,
                generator=generator,
            ).images[0]
            base_images.append(base_image)

        # Generate LoRA images
        logger.info(f"Generating {director} LoRA images for: {prompt}")

        # Load LoRA adapter
        lora_path = f"{DIRECTORS_DIR}/{director}"
        if Path(lora_path).exists():
            self.base_pipeline.load_lora_weights(lora_path)

            # Add director trigger phrase
            lora_prompt = f"{prompt} <{director}-style>"

            for seed in seeds:
                generator = torch.Generator().manual_seed(seed)

                lora_image = self.base_pipeline(
                    prompt=lora_prompt,
                    num_inference_steps=self.config.eval_num_inference_steps,
                    guidance_scale=self.config.eval_guidance_scale,
                    height=self.config.eval_resolution,
                    width=self.config.eval_resolution,
                    generator=generator,
                ).images[0]
                lora_images.append(lora_image)

            # Unload LoRA to clean up
            self.base_pipeline.unload_lora_weights()
        else:
            logger.warning(f"LoRA not found for director: {director}")
            lora_images = [None] * len(seeds)

        return base_images, lora_images

    def compute_clip_similarity(
        self, images: list[Image.Image], reference_texts: list[str]
    ) -> float:
        """Compute CLIP similarity between images and reference texts."""
        if not images or any(img is None for img in images):
            return 0.0

        with torch.no_grad():
            # Preprocess images
            image_inputs = torch.stack(
                [self.clip_preprocess(img) for img in images]
            ).to(self.device)

            # Encode images
            image_features = self.clip_model.encode_image(image_inputs)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            # Encode reference texts
            text_inputs = self.clip_tokenizer(reference_texts).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # Compute similarity
            similarities = torch.mm(image_features, text_features.T)
            return similarities.max(dim=1)[0].mean().item()

    def compute_aesthetic_score(self, images: list[Image.Image]) -> float:
        """Simple aesthetic scoring based on image statistics."""
        if not images or any(img is None for img in images):
            return 0.0

        scores = []
        for img in images:
            if img is None:
                continue

            # Convert to numpy array
            img_array = np.array(img)

            gray = np.mean(img_array, axis=2)
            contrast = np.std(gray) / 255.0  # Normalize to [0,1]

            # Edge density approximation using gradient magnitude
            grad_x = np.abs(np.diff(gray, axis=1))
            grad_y = np.abs(np.diff(gray, axis=0))
            edge_density = (np.mean(grad_x) + np.mean(grad_y)) / 2 / 255.0

            # Combined score (weighted)
            aesthetic_score = 0.6 * contrast + 0.4 * edge_density
            scores.append(aesthetic_score)

        return float(np.mean(scores)) if scores else 0.0

    def compute_director_style_score(
        self, images: list[Image.Image], director: str
    ) -> float:
        """Compute how well images match the director's style."""
        if not images or any(img is None for img in images):
            return 0.0

        # Director-specific style descriptors
        style_descriptors = {
            "anderson": [
                "symmetrical composition",
                "pastel colors",
                "centered framing",
                "whimsical cinematography",
                "retro aesthetic",
                "precise staging",
            ],
            "nolan": [
                "dark cinematography",
                "complex narrative",
                "cool color palette",
                "practical effects",
                "dramatic lighting",
                "architectural elements",
            ],
            "villeneuve": [
                "minimalist composition",
                "brutalist architecture",
                "epic scale",
                "muted colors",
                "atmospheric mood",
                "contemplative scenes",
            ],
            "fincher": [
                "green and yellow tints",
                "precise framing",
                "dark atmosphere",
                "urban environments",
                "meticulous detail",
                "cold lighting",
            ],
            "scorsese": [
                "warm color palette",
                "dynamic camera movement",
                "urban settings",
                "character-driven scenes",
                "rich textures",
                "energetic composition",
            ],
        }

        director_texts = style_descriptors.get(director, [])
        if not director_texts:
            return 0.0

        return self.compute_clip_similarity(images, director_texts)

    @modal.method()
    def evaluate_director_comparison(self, director: str) -> dict:
        logger.info(f"Evaluating director: {director} ({DIRECTOR_MAP[director]})")

        results = {
            "director": director,
            "director_name": DIRECTOR_MAP[director],
            "prompt_results": [],
            "overall_metrics": {},
        }

        base_scores = {
            "clip_similarity": [],
            "aesthetic_score": [],
            "style_score": [],
            "prompt_adherence": [],
        }

        lora_scores = {
            "clip_similarity": [],
            "aesthetic_score": [],
            "style_score": [],
            "prompt_adherence": [],
        }

        # Test each prompt
        for prompt_idx, prompt in enumerate(self.config.test_prompts):
            logger.info(
                f"Testing prompt {prompt_idx + 1}/{len(self.config.test_prompts)}: {prompt}"
            )

            # Generate comparison images
            seeds = self.config.eval_seeds[: self.config.num_samples_per_test]
            base_images, lora_images = self.generate_comparison_batch(
                prompt, director, seeds
            )

            # Compute metrics
            prompt_result = {
                "prompt": prompt,
                "seeds": seeds,
                "base_metrics": {},
                "lora_metrics": {},
                "improvement": {},
            }

            # Base model metrics
            if self.config.compute_clip_similarity:
                base_clip = self.compute_clip_similarity(base_images, [prompt])
                base_scores["clip_similarity"].append(base_clip)
                prompt_result["base_metrics"]["clip_similarity"] = base_clip

            if self.config.compute_aesthetic_score:
                base_aesthetic = self.compute_aesthetic_score(base_images)
                base_scores["aesthetic_score"].append(base_aesthetic)
                prompt_result["base_metrics"]["aesthetic_score"] = base_aesthetic

            if self.config.compute_style_consistency:
                base_style = self.compute_director_style_score(base_images, director)
                base_scores["style_score"].append(base_style)
                prompt_result["base_metrics"]["style_score"] = base_style

            # LoRA model metrics
            if any(img is not None for img in lora_images):
                if self.config.compute_clip_similarity:
                    lora_clip = self.compute_clip_similarity(lora_images, [prompt])
                    lora_scores["clip_similarity"].append(lora_clip)
                    prompt_result["lora_metrics"]["clip_similarity"] = lora_clip

                if self.config.compute_aesthetic_score:
                    lora_aesthetic = self.compute_aesthetic_score(lora_images)
                    lora_scores["aesthetic_score"].append(lora_aesthetic)
                    prompt_result["lora_metrics"]["aesthetic_score"] = lora_aesthetic

                if self.config.compute_style_consistency:
                    lora_style = self.compute_director_style_score(
                        lora_images, director
                    )
                    lora_scores["style_score"].append(lora_style)
                    prompt_result["lora_metrics"]["style_score"] = lora_style

                # Compute improvements
                for metric in ["clip_similarity", "aesthetic_score", "style_score"]:
                    if (
                        metric in prompt_result["base_metrics"]
                        and metric in prompt_result["lora_metrics"]
                    ):
                        base_val = prompt_result["base_metrics"][metric]
                        lora_val = prompt_result["lora_metrics"][metric]
                        improvement = (
                            (lora_val - base_val) / max(base_val, 1e-6)
                        ) * 100
                        prompt_result["improvement"][metric] = improvement

            results["prompt_results"].append(prompt_result)

        # Compute overall metrics
        for metric in ["clip_similarity", "aesthetic_score", "style_score"]:
            if base_scores[metric] and lora_scores[metric]:
                base_mean = np.mean(base_scores[metric])
                lora_mean = np.mean(lora_scores[metric])
                improvement_pct = ((lora_mean - base_mean) / max(base_mean, 1e-6)) * 100

                results["overall_metrics"][metric] = {
                    "base_mean": float(base_mean),
                    "lora_mean": float(lora_mean),
                    "improvement_percent": float(improvement_pct),
                    "base_std": float(np.std(base_scores[metric])),
                    "lora_std": float(np.std(lora_scores[metric])),
                }

        return results

    @modal.method()
    def run_full_evaluation(self) -> dict:
        """Run evaluation for all directors."""
        logger.info("Starting full director evaluation...")

        evaluation_results = {
            "config": asdict(self.config),
            "directors": {},
            "summary": {},
        }

        director_results = []

        # Evaluate each director
        for director in DIRECTOR_MAP.keys():
            lora_path = Path(f"{DIRECTORS_DIR}/{director}")
            if lora_path.exists():
                director_result = self.evaluate_director_comparison.remote(director)
                evaluation_results["directors"][director] = director_result
                director_results.append(director_result)
            else:
                logger.warning(f"Skipping {director}: LoRA not found at {lora_path}")

        # Compute summary statistics
        if director_results:
            summary = {}
            for metric in ["clip_similarity", "aesthetic_score", "style_score"]:
                improvements = [
                    r["overall_metrics"][metric]["improvement_percent"]
                    for r in director_results
                    if metric in r["overall_metrics"]
                ]
                if improvements:
                    summary[metric] = {
                        "mean_improvement": float(np.mean(improvements)),
                        "std_improvement": float(np.std(improvements)),
                        "best_director": max(
                            [
                                (
                                    r["director"],
                                    r["overall_metrics"][metric]["improvement_percent"],
                                )
                                for r in director_results
                                if metric in r["overall_metrics"]
                            ],
                            key=lambda x: x[1],
                        )[0]
                        if improvements
                        else None,
                    }

            evaluation_results["summary"] = summary

        return evaluation_results


@app.function(timeout=3600)
def run_evaluation(config_dict: dict | None = None) -> dict:
    """Run the complete evaluation pipeline."""

    # Use default config if none provided
    if config_dict is None:
        config = EvaluationConfig()
    else:
        config = EvaluationConfig(**config_dict)

    # Use Modal class method to ensure proper setup
    evaluator = DirectorEvaluator(config)
    results = evaluator.run_full_evaluation.remote()

    # Save results to volume
    results_path = "/volume/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Evaluation complete! Results saved to {results_path}")
    volume.commit()

    return results


@app.function(timeout=1800)
def quick_evaluation(director: str, num_prompts: int = 3) -> dict:
    """Run a quick evaluation for a single director."""

    config = EvaluationConfig()
    # Limit prompts for quick test
    config.test_prompts = config.test_prompts[:num_prompts]
    config.num_samples_per_test = 2
    config.eval_seeds = config.eval_seeds[:2]

    # Use Modal class method
    evaluator = DirectorEvaluator(config)
    result = evaluator.evaluate_director_comparison.remote(director)

    return result


@app.local_entrypoint()
def main(director: str | None = None, quick: bool = False, num_prompts: int = 3):
    """Main entry point for evaluation."""

    if director and quick:
        # Run quick evaluation for single director
        result = quick_evaluation.remote(director, num_prompts)
        print(f"\n=== Quick Evaluation Results for {director} ===")

        if "overall_metrics" in result:
            for metric, data in result["overall_metrics"].items():
                improvement = data["improvement_percent"]
                print(f"{metric}: {improvement:+.1f}% improvement")

    elif director:
        # Run full evaluation for single director
        config = EvaluationConfig()
        evaluator = DirectorEvaluator(config)
        result = evaluator.evaluate_director_comparison.remote(director)
        print(f"\n=== Full Evaluation Results for {director} ===")
        print(json.dumps(result, indent=2))

    else:
        # Run full evaluation for all directors
        results = run_evaluation.remote()

        print("\n=== Director Diffusion Evaluation Summary ===")
        if "summary" in results:
            for metric, data in results["summary"].items():
                mean_improvement = data["mean_improvement"]
                best_director = data["best_director"]
                print(f"{metric}: {mean_improvement:+.1f}% average improvement")
                print(f"  Best performer: {best_director}")
