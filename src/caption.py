import json
import logging
import os
from glob import glob

import modal
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from src.images import caption_image
from src.utils import STYLE_MAP

image = caption_image()
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

        from huggingface_hub import snapshot_download
        from transformers import pipeline

        snapshot_download(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            local_dir="/volume/labels/Qwen/Qwen2.5-VL-3B-Instruct",
        )
        self.pipe = pipeline(
            "image-text-to-text",
            model="/volume/labels/Qwen/Qwen2.5-VL-3B-Instruct",
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
        try:
            img = self.preprocess_image(data["image_path"])
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {
                            "type": "text",
                            "text": """
                            You are a professional image annotator for training a director's visual style. Create a caption that:

                            1. Starts with the director's style characteristics in the first sentence
                            2. Describes cinematographic elements: lighting, composition, color grading
                            3. Mentions camera techniques: angles, depth of field, framing
                            4. Notes atmosphere and mood specific to this director's work
                            5. Keep it under 75 words total
                            6. Focus on visual elements that make this director's style distinctive
                            7. Use concrete, specific descriptors rather than generic terms
                            """,
                        },
                    ],
                },
            ]

            with torch.no_grad():
                raw_captions = self.pipe(messages, batch_size=1)
            results = []
            final_caption = create_reinforced_caption(
                data["style_token"], extract_assistant_content(raw_captions)
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
        if not raw_caption or not isinstance(raw_caption, list):
            logging.warning("Invalid raw_caption format: not a list or empty")
            return ""
        if len(raw_caption) == 0:
            logging.warning("Empty raw_caption list")
            return ""
        generated_text = raw_caption[0].get("generated_text")
        if not generated_text or len(generated_text) < 2:
            logging.warning("Invalid generated_text structure")
            return ""
        return generated_text[1].get("content", "")
    except (KeyError, IndexError, TypeError) as e:
        logging.warning(f"Failed to extract assistant content: {e}")
        return ""


def create_reinforced_caption(style_token, raw_caption):
    """Create caption that reinforces trigger phrase without dilution."""
    if not style_token or not isinstance(style_token, str):
        logging.warning(f"Invalid style_token: {style_token}")
        return raw_caption if raw_caption else ""

    if not raw_caption or not isinstance(raw_caption, str):
        logging.warning(f"Invalid raw_caption: {raw_caption}")
        return f"{style_token}, in {style_token} style"

    # Truncate if too long
    if len(raw_caption.split()) > 75:
        words = raw_caption.split()[:75]
        raw_caption = " ".join(words)

    # Strategic placement: beginning, middle hint, and end
    return f"{style_token} {raw_caption}, in {style_token} style"


@app.function(timeout=1500)
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
