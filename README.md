# director-diffusion

<p align="center">
  <a href="https://gradio.app/" target="_blank">
    <img src="https://raw.githubusercontent.com/gradio-app/gradio/main/assets/logo.svg" alt="Gradio Logo" width="120" />
  </a>
  <br>
  <a href="https://nano-diffusion--flux-lora-gradio-gradio-app.modal.run/" target="_blank">
    <b>Try the Director-Diffusion Gradio App &rarr;</b>
  </a>
</p>


Director-Diffusion is an open-source package to train Low Rank Adaptation Matrices of the Flux1.Krea-dev model to fit the style of famous directors. The directors chosen in this package are Christopher Nolan, Martin Scorsese, Wes Anderson, Denis Villeneuve, and David Fincher. They are so chosen for their unique styles and my personal affinity for their work.



## Roadmap

- [x] Image Collection
- [x] Caption data and verify performance
- [x] Train base LoRAs
- [x] Serve model off Gradio App
- [ ] Evaluate using Frechet Inception Distance (FID) and CLIP Score
- [ ] Create LCM distillation

## Uses

- Flux 1 - Krea dev (black-forest-labs/FLUX.1-Krea-dev) as the base model for training
- uv for package management
- ruff for code quality
- ty for type checking
- modal for infrastructure
- shotdeck (https://shotdeck.com/welcome/home) and ffmpeg for training stills and data
- Qwen 2.5VL - 3B for image captioning

## Getting Started

To run the code, sync the uv environment (`uv sync`) and run the following commands:

- captioning: `uv run modal run -m src.caption`. You can then upload the images to remote modal storage using `modal volume`.
- train: `uv run modal run -m src.train`. You will likely need to add the `detach` flag as well to ensure that the train run does not get limited by session length.
- serve: `uv run modal deploy -m src.serve`, `uv run modal serve -m src.serve` (for local dev).


## Considerations
over black-forest-labs/FLUX.1-dev due to photorealism
and ffmpeg for training stills
No BLIP-2


## Dataset

Images were collected from shotdeck.com. Director counts for each image are listed below:

- Anderson: 201 images
- Fincher: 214 images
- Nolan: 232 images
- Scorsese: 215 images
- Villeneuve: 197 images

## Acknowledgments
I'd like to thank Alec Powell and the team at Modal for their support of this work through GPU credits. Also, thank you to the team at Astral for their great open-source work!

## Relevant Papers and Blog posts

- https://www.krea.ai/blog/flux-krea-open-source-release
- https://modal.com/blog/flux-3x-faster
- https://modal.com/docs/examples/diffusers_lora_finetune
- https://arxiv.org/pdf/2210.03142
- https://arxiv.org/abs/2407.15811
- https://arxiv.org/abs/2006.11239
- https://arxiv.org/abs/2010.02502
- https://arxiv.org/abs/2207.12598