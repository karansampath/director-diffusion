# director-diffusion

Director-Diffusion is an open-source package to train 

## Uses

- Flux 1 - Krea dev (black-forest-labs/FLUX.1-Krea-dev) as the base model for training
- uv for package management
- ruff for code quality
- ty for type checking
- modal for infrastructure
- shotdeck (https://shotdeck.com/welcome/home) and ffmpeg for training stills and data
- Qwen 2.5VL - 3B for image captioning


## Getting Started

To run the code, sync the uv environment and run the following commands:

- captioning: `uv run modal run -m src.auto_caption`. You can then upload the images 
- train: `uv run modal run -m src.train`. You will likely need to add the `detach` flag as well
- serve: `uv run modal serve -m src.serve`.


## Considerations
over black-forest-labs/FLUX.1-dev due to photorealism
and ffmpeg for training stills
No BLIP-2


## Dataset

🎬 Director Counts:
========================================
anderson         201 images
fincher          214 images
nolan            232 images
scorsese         215 images
villeneuve       197 images

## Acknowledgments
I'd like to thank Alec Powell and the team at Modal for their help in giving me GPU credits to help complete this work. Also thank you to the team at Astral for their great open-source work!

## Relevant Papers and Blog posts

https://www.krea.ai/blog/flux-krea-open-source-release
https://modal.com/blog/flux-3x-faster
https://modal.com/docs/examples/diffusers_lora_finetune
https://arxiv.org/pdf/2210.03142
https://arxiv.org/abs/2407.15811
https://arxiv.org/abs/2006.11239
https://arxiv.org/abs/2010.02502
https://arxiv.org/abs/2207.12598