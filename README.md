<h2 align="center"><img src="./figures/moonstep.png" alt="Moonstep logo" style="height: 1em; vertical-align: middle;"> One Small Step in Latent, One Giant Leap for Pixels:<br>Fast Latent Upscale Adapter for Your Diffusion Models</h2>

<div align="center" style="margin-bottom: 1rem;">
<a href="https://razinaleksandr.github.io/" target="_blank">Aleksandr Razin</a><sup>*</sup> ·
<a href="https://www.linkedin.com/in/danil-kazancev/" target="_blank">Danil Kazantsev</a><sup>*</sup> ·
<a href="https://www.linkedin.com/in/iamakarov/" target="_blank">Ilya Makarov</a>
</div>

<p align="center" style="margin-top: 1.25rem;">
<a href="https://github.com/vaskers5/LUA" target="_blank"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2511.10629" target="_blank"><img src="https://img.shields.io/badge/arXiv-PDF-b31b1b" alt="arXiv"></a>
<a href="https://huggingface.co/papers/2511.10629" target="_blank"><img src="https://img.shields.io/badge/HuggingFace-Paper-orange" alt="HuggingFace Paper"></a>
<a href="https://huggingface.co/vaskers5/LUA-FLUX" target="_blank"><img src="https://img.shields.io/badge/HuggingFace-Weights-yellow" alt="HuggingFace Weights"></a>
<a href="https://www.youtube.com/watch?v=7jV9NYu32dk" target="_blank"><img src="https://img.shields.io/badge/YouTube-Demo-red" alt="YouTube Demo"></a>
</p>

This repository contains the official implementation of the paper **"One Small Step in Latent, One Giant Leap for Pixels: Fast Latent Upscale Adapter for Your Diffusion Models"**.

We present the **Latent Upscaler Adapter (LUA)**, a lightweight module that performs super-resolution directly on the generator's latent code before the final VAE decoding step. LUA integrates as a drop-in component, requiring no modifications to the base model or additional diffusion stages. It enables high-resolution synthesis through a single feed-forward pass in latent space, achieving comparable perceptual quality to pixel-space methods while reducing decoding and upscaling time.

<div align="center">
<img src="./figures/figure_2.png" alt="Teaser" width="80%">
</div>

## Installation

```bash
git clone https://github.com/vaskers5/LUA.git
cd LUA
pip install -r requirements.txt
```

## Quick Start

LUA weights are hosted on HuggingFace and downloaded automatically on first use.

### Python API

```python
import torch
from diffusers import FluxPipeline
from lua import load_model, upscale_latent

# Load models
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.vae.enable_tiling()

lua_model = load_model(device="cuda")  # auto-downloads weights from HF

# Generate base latent at 1024x1024
result = pipe("a cat astronaut", output_type="latent", width=1024, height=1024)

# Unpack to VAE space
latent = pipe._unpack_latents(result.images, 1024, 1024, pipe.vae_scale_factor)
latent = (latent / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

# Upscale x2 (1024 -> 2048) or x4 (1024 -> 4096)
upscaled = upscale_latent(lua_model, latent, head="x2")

# Decode to image
image = pipe.vae.decode(upscaled.to(torch.bfloat16), return_dict=False)[0]
image = pipe.image_processor.postprocess(image, output_type="pil")[0]
image.save("output_2k.png")
```

### CLI Inference

```bash
# 2K image (1024 -> 2048)
python inference.py --prompt "a mountain landscape, cinematic" --head x2

# 4K image (1024 -> 4096)
python inference.py --prompt "a mountain landscape, cinematic" --head x4 --output landscape_4k.png

# Use a local checkpoint
python inference.py --prompt "hello" --weights ./my_checkpoint.pth --head x2
```

### Gradio Demo

Interactive demo with side-by-side comparison against direct FLUX generation:

```bash
python gradio_demo.py
```

The demo compares LUA path (FLUX@1024 + LUA upscale) vs Direct path (FLUX@target) at the same output resolution, with interactive magnifying loupes and timing breakdowns.

You can configure the FLUX model via environment variables:
```bash
FLUX_MODEL_ID="black-forest-labs/FLUX.1-dev" python gradio_demo.py
```

## Model Details

| | |
|---|---|
| **Architecture** | SwinIR-based transformer with multi-head upsampling |
| **Parameters** | ~250M |
| **Input** | 16-channel VAE latent (FLUX latent space) |
| **Heads** | x2 (2x upscaling), x4 (4x upscaling) |
| **Output** | Upscaled 16-channel VAE latent |

LUA operates entirely in the latent space — it upscales the latent code before the VAE decoder, which means the expensive VAE decode only happens once at the target resolution.

## Training

Training code will be released soon.

## Citation

```bibtex
@article{razin2024lua,
  title={One Small Step in Latent, One Giant Leap for Pixels: Fast Latent Upscale Adapter for Your Diffusion Models},
  author={Razin, Aleksandr and Kazantsev, Danil and Makarov, Ilya},
  journal={arXiv preprint arXiv:2511.10629},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 — see the [LICENSE](LICENSE) file for details.
