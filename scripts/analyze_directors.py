#!/usr/bin/env python

"""
Script to analyze the main caption dataset and show available directors.
Run this first to see what directors you have data for.

Usage:
    modal run scripts/analyze_directors.py
"""

import modal

app = modal.App(name="analyze-directors")

# Minimal image for analysis
image = modal.Image.debian_slim().uv_pip_install("json")

# Use your existing volume
volume = modal.Volume.from_name("director-diffusion")


@app.function(image=image, volumes={"/volume": volume})
def analyze_directors():
    """Analyze the main caption dataset to see available directors."""
    import json
    from collections import Counter

    # Load main caption dataset
    with open("/volume/labels/main_caption_dataset.json", "r") as f:
        all_captions = json.load(f)

    # Extract director names from file paths
    directors = []
    sample_images = {}  # Store sample images for each director

    for item in all_captions:
        file_path = item["file_name"]
        if "/volume/data/" in file_path:
            # Extract director name from path like "/volume/data/anderson/image.jpg"
            parts = file_path.split("/volume/data/")[1].split("/")
            if len(parts) > 0:
                director = parts[0]
                directors.append(director)

                # Store sample caption for each director
                if director not in sample_images:
                    sample_images[director] = {
                        "image": parts[1] if len(parts) > 1 else "unknown",
                        "caption": item["text"],
                    }

    # Count images per director
    director_counts = Counter(directors)

    print("\n🎬 Available Directors in Dataset:")
    print("=" * 50)
    for director, count in sorted(director_counts.items()):
        print(f"{director:15} {count:4} images")

        # Show sample caption
        if director in sample_images:
            sample = sample_images[director]
            print(f"    Sample: {sample['caption'][:80]}...")
        print()

    print(f"Total Directors: {len(director_counts)}")
    print(f"Total Images: {sum(director_counts.values())}")

    # Show configuration template
    print("\n📋 Configuration Template:")
    print("=" * 50)
    print("Add these to your MultiDirectorConfig.directors list:")
    print()

    print("\n✅ Analysis complete!")
    return director_counts


@app.local_entrypoint()
def main():
    analyze_directors.remote()


if __name__ == "__main__":
    main()
