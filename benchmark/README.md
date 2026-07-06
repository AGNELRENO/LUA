# LUA benchmark

Measures FLUX **with vs without LUA** at 2K and 4K on a single GPU — four
paths per sample, same prompt and seed everywhere:

| path | pipeline |
|---|---|
| `lua_2k` | FLUX@1024 → LUA x2 → tiled VAE decode @2048 |
| `direct_2k` | FLUX@2048 → tiled VAE decode @2048 |
| `lua_4k` | FLUX@1024 → LUA x4 → tiled VAE decode @4096 |
| `direct_4k` | FLUX@4096 → tiled VAE decode @4096 |

```bash
HF_TOKEN=... python benchmark/run_benchmark.py
```

Needs a ~48 GB GPU (peaks: LUA paths ~34 GB; direct 4K fits because both text
encoders are offloaded to CPU after prompt pre-encoding — a shared, excluded
~0.2 s cost). Every stage is timed between `cuda.synchronize()` barriers with
per-stage peak-VRAM tracking; `direct_4k` is OOM-guarded.

Outputs land in `benchmark_out/`: one folder per sample with `prompt.txt`,
`info.json`, decoder-ready latents (`latent_base.pt`, per-path
`latent_upscaled.pt`/`latent.pt`), decoded images before/after upscale, plus
top-level `timings.json` and a ready-made `report.md` with 2K/4K comparison
tables and speedups.

Version notes (fresh installs, July 2026): latest diffusers needs the torch
triplet upgraded together (`torch==2.6.0 torchvision==0.21.0
torchaudio==2.6.0`); transformers 5.x may ignore `torch_dtype` for the text
encoders (the script re-casts every component to bf16); bare `vae.decode()`
must run under `torch.inference_mode()` (the script does).
