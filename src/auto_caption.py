import os
import json
from glob import glob
from PIL import Image
from tqdm import tqdm
import logging

import torch
from torch.utils.data import Dataset
import modal

from src.utils import STYLE_MAP

image = (
    modal.Image.debian_slim()
    .uv_pip_install(
        "torch",
        "torchvision",
        "accelerate",
        "numpy",
        "transformers",
        "huggingface-hub[hf_transfer]",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_dir("src", "/src", copy=True)
)
volume = modal.Volume.from_name(
    "director-diffusion", create_if_missing=True
)  # upload data to this volume
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)
app = modal.App(
    name="director-diffusion-caption",
    image=image,
    volumes={"/volume": volume},
    secrets=[huggingface_secret],
)


class ImageCaptionDataset(Dataset):
    """Dataset for image captioning with style tokens. Written to be compatible with Modal and PyTorch DataSets."""

    def __init__(self, data_dir="/volume/data"):
        self.samples = []

        # Collect all image paths with their style tokens
        for style_dir in os.listdir(data_dir):
            if style_dir not in STYLE_MAP:
                continue

            token = STYLE_MAP[style_dir]
            image_paths = glob(f"{data_dir}/{style_dir}/*.jpg")

            for path in image_paths:
                self.samples.append({"image_path": path, "style_token": token})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_samples(self):
        return self.samples


@app.cls(image=image, gpu="H200")
class Model:
    @modal.enter()
    def load_model(self):
        volume.reload()

        from transformers import pipeline
        from huggingface_hub import snapshot_download

        snapshot_download(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            local_dir="/volume/models/Qwen/Qwen2.5-VL-3B-Instruct",
        )
        self.pipe = pipeline(
            "image-text-to-text",
            model="/volume/models/Qwen/Qwen2.5-VL-3B-Instruct",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_new_tokens=128,
        )
        self.pipe.model = torch.compile(self.pipe.model, mode="reduce-overhead")

    def preprocess_image(self, image_path):
        """Preprocess image to reduce memory usage"""
        img = Image.open(image_path).convert("RGB")
        # Resize large images to reduce memory usage
        max_size = 1024
        if max(img.size) > max_size:
            img = img.resize((max_size, max_size), Image.Resampling.LANCZOS)
        return img

    @modal.method()
    def caption_image(self, data):
        """Process a batch of images for captioning."""
        try:
            img = self.preprocess_image(data["image_path"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {
                            "type": "text",
                            "text": "Describe this image in detail, focusing on the cinematic style, composition, lighting, and color palette. Then, describe the content of the image. Be as descriptive as possible and do not hallucinate. Use less than 50 words.",
                        },
                    ],
                },
            ]

            with torch.no_grad():
                raw_captions = self.pipe(messages, batch_size=1)
            results = []
            final_caption = (
                f"{data['style_token']} {extract_assistant_content(raw_captions)}"
            )
            results.append({"file_name": data["image_path"], "text": final_caption})
            return results

        except Exception as e:
            logging.error(f"Error processing batch with pipeline: {e}")
            return []


def extract_assistant_content(raw_caption):
    """
    Extracts the assistant's content from a conversation list.
    """
    try:
        return raw_caption[0]["generated_text"][1]["content"]
    except Exception:
        return ""


@app.function()
def process_dataset():
    # Create dataset and dataloader
    dataset = ImageCaptionDataset()

    model = Model()
    all_results = []

    # Process dataset by mapping over it
    for result in tqdm(
        model.caption_image.map(dataset.get_samples()),
        total=len(dataset),
        desc="Processing images",
    ):
        if result:
            all_results.extend(result)
    logging.info(f"Captioning complete! Processed {len(all_results)} images.")

    # Save results
    return all_results


@app.local_entrypoint()
def main():
    results = process_dataset.remote()
    with open("main_caption_dataset.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
