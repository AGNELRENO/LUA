#!/usr/bin/env python3
"""
LUA inference script — upscale FLUX latents using LUA.

Examples:
    # Generate a 2K image via FLUX + LUA x2 upscaling
    python inference.py --prompt "a cat astronaut" --head x2

    # Generate a 4K image with a specific seed
    python inference.py --prompt "a mountain landscape" --head x4 --seed 123

    # Use a local checkpoint instead of downloading from HuggingFace
    python inference.py --prompt "hello" --weights ./my_checkpoint.pth
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from lua import load_model, upscale_latent


def main():
    parser = argparse.ArgumentParser(description="LUA: Latent Upscale Adapter inference")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for FLUX generation")
    parser.add_argument("--head", type=str, default="x2", choices=["x2", "x4"],
                        help="Upscaling head: x2 (1024→2048) or x4 (1024→4096)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--steps", type=int, default=28, help="Number of FLUX inference steps")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance scale")
    parser.add_argument("--flux-model", type=str, default="black-forest-labs/FLUX.1-dev",
                        help="FLUX model ID on HuggingFace")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to LUA weights (downloads from HF if not set)")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--base-res", type=int, default=1024, help="Base generation resolution")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    scale = {"x2": 2, "x4": 4}[args.head]
    target = args.base_res * scale

    # Load models
    print(f"[1/4] Loading FLUX pipeline ({args.flux_model}) ...")
    from diffusers import FluxPipeline
    pipe = FluxPipeline.from_pretrained(args.flux_model, torch_dtype=dtype)
    pipe.to(device)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    print(f"[2/4] Loading LUA model ...")
    lua_model = load_model(weights_path=args.weights, device=device)

    # Generate base latent
    print(f"[3/4] Generating base latent ({args.base_res}x{args.base_res}, {args.steps} steps) ...")
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    packed = pipe(
        prompt=args.prompt, num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        width=args.base_res, height=args.base_res,
        output_type="latent", generator=gen,
    ).images
    t_flux = time.time() - t0

    # Unpack to VAE space
    vae_lat = pipe._unpack_latents(
        packed, height=args.base_res, width=args.base_res,
        vae_scale_factor=pipe.vae_scale_factor,
    )
    vae_lat = (vae_lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

    # LUA upscale
    print(f"[4/4] LUA {args.head} upscaling → {target}x{target} ...")
    t0 = time.time()
    up_lat = upscale_latent(lua_model, vae_lat, head=args.head)
    t_lua = time.time() - t0

    # VAE decode
    t0 = time.time()
    dec = pipe.vae.decode(up_lat.to(dtype=dtype), return_dict=False)[0]
    image = pipe.image_processor.postprocess(dec, output_type="pil")[0]
    t_dec = time.time() - t0

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(output_path))

    total = t_flux + t_lua + t_dec
    print(f"\nDone! Saved {target}x{target} image to {output_path}")
    print(f"  FLUX generation: {t_flux:.2f}s")
    print(f"  LUA upscale:     {t_lua:.2f}s")
    print(f"  VAE decode:      {t_dec:.2f}s")
    print(f"  Total:           {total:.2f}s")


if __name__ == "__main__":
    main()
