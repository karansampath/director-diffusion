import os
import json
from glob import glob
from PIL import Image

import torch
import modal

from src.utils import STYLE_MAP

image = (
    modal.Image.debian_slim()
    .uv_pip_install(
        "torch==2.7.1",
        "torchvision",
        "accelerate",
        "numpy",
        "transformers",
        "huggingface-hub==0.26.2",
        "hf_transfer==0.1.8",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir("src", "/src", copy=True)
    .add_local_dir("data", "/data")
)
volume = modal.Volume.from_name("director-diffusion-caption", create_if_missing=True)
huggingface_secret = modal.Secret.from_name(
    "huggingface-secret", required_keys=["HF_TOKEN"]
)
app = modal.App(
    name="director-diffusion-caption",
    image=image,
    volumes={"/volume": volume},
    secrets=[huggingface_secret],
)


@app.cls(image=image, gpu="H100")
class Model:
    @modal.enter()
    def load_model(self):
        volume.reload()

        from transformers import AutoProcessor, Blip2ForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            "Salesforce/blip2-opt-2.7b", use_fast=True
        )
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16, device_map="auto"
        )
        self.processor = processor
        self.model = model

    @modal.method()
    def caption_image(self, image_path):

        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(img, "Describe exactly what you see in a few words, don't hallucinate.", return_tensors="pt").to("cuda")
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

@app.local_entrypoint()
def main():
    dataset = []

    for style_dir in os.listdir("data"):
        if style_dir not in STYLE_MAP:
            continue

        token = STYLE_MAP[style_dir]
        image_paths = glob(f"data/{style_dir}/*.jpg")
        for path in image_paths:
            try:
                # Convert local path to Modal path
                modal_path = path.replace("data/", "/data/")
                raw_caption = Model().caption_image.remote(modal_path)
                final_caption = f"{token} {raw_caption}"
                dataset.append({"file_name": path, "text": final_caption})
                print(f"Processed {path}")
            except Exception as e:
                print(f"Error processing {path}: {e}")

    with open("caption_dataset.json", "w") as f:
        json.dump(dataset, f)
