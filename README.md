# director-diffusion

Training a diffusion model from scratch on a nano budget.

Uses:

- uv for package management
- ruff for code quality
- ty for type checking
- modal for infrastructure
- shotdeck (https://shotdeck.com/welcome/home) and ffmpeg for training stills
- BLIP-2 and Qwen 2.5VL - 3B for image captioning
- Choice of Flux 1 - Krea dev (black-forest-labs/FLUX.1-Krea-dev) over black-forest-labs/FLUX.1-dev due to photorealism

🎬 Director Counts in Dataset:
========================================
anderson         201 images
fincher          214 images
nolan            232 images
scorsese         215 images
villeneuve       197 images



Acknowledgments:
I'd like to thank Alec Powell and the team at Modal for their help in giving me GPU credits to help complete this work. Also thank you to the team at Astral for their grat 