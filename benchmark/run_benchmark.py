#!/usr/bin/env python3
"""LUA benchmark: FLUX with/without LUA at 2K and 4K, on a single GPU.

For each sample (same prompt + seed everywhere), measures four paths:

  lua_2k     FLUX@1024 -> LUA x2 -> tiled VAE decode @2048
  direct_2k  FLUX@2048 -> tiled VAE decode @2048
  lua_4k     FLUX@1024 -> LUA x4 -> tiled VAE decode @4096   (base gen shared)

(direct FLUX@4096 is intentionally omitted — ~15-20 min/sample.)

Methodology notes:
  * Prompts are pre-encoded once, then BOTH text encoders are moved to CPU.
    This frees ~10 GB so direct FLUX@4096 fits on a 48 GB card, and every
    path sees identical conditions. Encoding time is reported separately
    (it is a constant ~0.2 s shared by all paths).
  * Every stage is timed between torch.cuda.synchronize() barriers and
    records its own peak VRAM (reset per stage).
  * The LUA-path total = base generation + upscale + final decode. The
    base-latent *image* decode is saved as an artifact but not counted —
    it is not part of the production path.
  * direct_4k is wrapped in an OOM guard: a failure is recorded and the
    benchmark continues.

Artifacts, per sample, under benchmark_out/{name}/:
  prompt.txt, info.json, latent_base.pt, base_1024.png,
  lua_2k/{latent_upscaled.pt, final_2048.png}
  lua_4k/{latent_upscaled.pt, final_4096.png}
  direct_2k/{latent.pt, final_2048.png}
  direct_4k/{latent.pt, final_4096.png}
Plus benchmark_out/{timings.json, report.md}.
"""

import json
import sys
import time
from pathlib import Path

import torch
from diffusers import FluxPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root
from lua import load_model, upscale_latent  # noqa: E402

JOBS = [
    dict(name="01_valley", seed=101,
         prompt="a misty alpine valley at golden hour, cinematic wide shot, volumetric light"),
    dict(name="02_fisherman", seed=102,
         prompt="portrait of an elderly fisherman with weathered skin, dramatic rim lighting, 85mm photo"),
    dict(name="03_fox", seed=103,
         prompt="a red fox leaping over a frozen creek in fresh snow, high-speed wildlife photography"),
    dict(name="04_tokyo", seed=104,
         prompt="futuristic Tokyo street at night in the rain, neon reflections, cyberpunk atmosphere"),
    dict(name="05_bee", seed=105,
         prompt="macro shot of a honeybee on a sunflower, dewdrops, shallow depth of field"),
]
STEPS, GUIDANCE, BASE = 28, 3.5, 1024
FLUX_MODEL = "black-forest-labs/FLUX.1-dev"
ROOT = Path("benchmark_out")
ROOT.mkdir(parents=True, exist_ok=True)

device, dtype = "cuda", torch.bfloat16


def sync() -> None:
    torch.cuda.synchronize()


def gb(x: int) -> float:
    return round(x / 2**30, 2)


# ------------------------------------------------------------------- setup
print(f"[setup] loading FLUX pipeline ({FLUX_MODEL}) ...", flush=True)
t0 = time.time()
pipe = FluxPipeline.from_pretrained(FLUX_MODEL, torch_dtype=dtype)
pipe.to(device)
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
# transformers 5.x can load the text encoders in fp32 despite torch_dtype=bf16
for comp_name in ("transformer", "text_encoder", "text_encoder_2", "vae"):
    comp = getattr(pipe, comp_name, None)
    if comp is not None:
        comp.to(dtype)
lua_model = load_model(device=device)

# GPU-residency proof: every weight-bearing module must live on cuda.
devices = dict(
    transformer=str(next(pipe.transformer.parameters()).device),
    vae=str(next(pipe.vae.parameters()).device),
    lua=str(next(lua_model.parameters()).device),
)
assert all(d.startswith("cuda") for d in devices.values()), f"CPU fallback: {devices}"
print(f"[setup] devices: {devices}", flush=True)

# Pre-encode prompts, then offload text encoders (see module docstring).
embeds = {}
sync()
t0 = time.time()
with torch.inference_mode():
    for job in JOBS:
        pe, ppe, _ = pipe.encode_prompt(
            prompt=job["prompt"], prompt_2=None, device=device,
            num_images_per_prompt=1,
        )
        embeds[job["name"]] = (pe, ppe)
sync()
t_encode_all = round(time.time() - t0, 2)
# Drop the encoders from the pipeline: frees ~10GB AND keeps
# _execution_device on cuda (a CPU-parked encoder makes the pipeline
# build its initial latents on CPU -> device-mismatch crash).
pipe.text_encoder = None
pipe.text_encoder_2 = None
torch.cuda.empty_cache()
print(f"[setup] prompts encoded in {t_encode_all}s; text encoders released; "
      f"cuda allocated: {gb(torch.cuda.memory_allocated())}GB", flush=True)


def gen_latent(job: dict, res: int):
    """FLUX generation at res x res -> (decoder-ready latent, seconds)."""
    pe, ppe = embeds[job["name"]]
    sync()
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(job["seed"])
    packed = pipe(
        prompt_embeds=pe, pooled_prompt_embeds=ppe,
        num_inference_steps=STEPS, guidance_scale=GUIDANCE,
        width=res, height=res, output_type="latent", generator=gen,
    ).images
    sync()
    elapsed = time.time() - t0
    lat = pipe._unpack_latents(packed, height=res, width=res,
                               vae_scale_factor=pipe.vae_scale_factor)
    lat = (lat / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    return lat, elapsed


@torch.inference_mode()
def decode_timed(lat: torch.Tensor):
    """Tiled VAE decode -> (PIL image, seconds)."""
    sync()
    t0 = time.time()
    img = pipe.vae.decode(lat.to(device=device, dtype=dtype), return_dict=False)[0]
    pil = pipe.image_processor.postprocess(img, output_type="pil")[0]
    return pil, time.time() - t0


def save_latent(path: Path, lat: torch.Tensor, job: dict, path_name: str) -> None:
    torch.save({
        "latent": lat.detach().float().cpu(),
        "space": "vae latent, scale/shift corrected (decoder-ready)",
        "path": path_name,
        "prompt": job["prompt"], "seed": job["seed"],
        "steps": STEPS, "guidance": GUIDANCE, "flux_model": FLUX_MODEL,
    }, path)


# --------------------------------------------------------------- benchmark
records = []
for i, job in enumerate(JOBS, 1):
    name = job["name"]
    d = ROOT / name
    for sub in ("lua_2k", "lua_4k", "direct_2k"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    (d / "prompt.txt").write_text(
        f"{job['prompt']}\n\nseed: {job['seed']}\nsteps: {STEPS}\n"
        f"guidance: {GUIDANCE}\nbase_res: {BASE}\n")
    info: dict = dict(name=name, seed=job["seed"], prompt=job["prompt"], stages={})
    print(f"[{i}/{len(JOBS)}] {name}: {job['prompt'][:70]}", flush=True)

    # -- shared base generation (the LUA paths' first stage)
    torch.cuda.reset_peak_memory_stats()
    lat_base, t_base = gen_latent(job, BASE)
    assert lat_base.is_cuda, "base latent left the GPU"
    peak_base = gb(torch.cuda.max_memory_allocated())
    save_latent(d / "latent_base.pt", lat_base, job, "base_1024")
    img, t_dec_base = decode_timed(lat_base)
    img.save(d / "base_1024.png")
    info["stages"]["base_1024"] = dict(
        t_gen=round(t_base, 2), t_decode=round(t_dec_base, 2), peak_vram_gb=peak_base)
    print(f"    base_1024: gen={t_base:.1f}s decode={t_dec_base:.1f}s "
          f"peak={peak_base}GB", flush=True)

    # -- LUA paths (x2 -> 2048, x4 -> 4096) reusing the base latent
    for head, res in (("x2", 2048), ("x4", 4096)):
        key = f"lua_{res // 1024}k"
        torch.cuda.reset_peak_memory_stats()
        sync()
        t0 = time.time()
        up = upscale_latent(lua_model, lat_base, head=head)
        sync()
        t_up = time.time() - t0
        assert up.is_cuda, "LUA output left the GPU"
        save_latent(d / key / "latent_upscaled.pt", up, job, key)
        img, t_dec = decode_timed(up)
        img.save(d / key / f"final_{res}.png")
        peak = gb(torch.cuda.max_memory_allocated())
        total = t_base + t_up + t_dec
        info["stages"][key] = dict(
            t_base_gen=round(t_base, 2), t_upscale=round(t_up, 2),
            t_decode=round(t_dec, 2), t_total=round(total, 2), peak_vram_gb=peak)
        print(f"    {key}: upscale={t_up:.1f}s decode={t_dec:.1f}s "
              f"total={total:.1f}s peak={peak}GB", flush=True)
        del up, img

    del lat_base
    torch.cuda.empty_cache()

    # -- direct paths (4096 omitted: impractically slow, ~15-20 min/sample)
    for res in (2048,):
        key = f"direct_{res // 1024}k"
        torch.cuda.reset_peak_memory_stats()
        try:
            lat, t_gen = gen_latent(job, res)
            save_latent(d / key / "latent.pt", lat, job, key)
            img, t_dec = decode_timed(lat)
            img.save(d / key / f"final_{res}.png")
            peak = gb(torch.cuda.max_memory_allocated())
            info["stages"][key] = dict(
                t_gen=round(t_gen, 2), t_decode=round(t_dec, 2),
                t_total=round(t_gen + t_dec, 2), peak_vram_gb=peak)
            print(f"    {key}: gen={t_gen:.1f}s decode={t_dec:.1f}s "
                  f"total={t_gen + t_dec:.1f}s peak={peak}GB", flush=True)
            del lat, img
        except torch.OutOfMemoryError as e:
            info["stages"][key] = dict(error=f"OOM: {e}")
            print(f"    {key}: OOM — skipped", flush=True)
        torch.cuda.empty_cache()

    st = info["stages"]
    info["speedup_2k"] = (round(st["direct_2k"]["t_total"] / st["lua_2k"]["t_total"], 2)
                          if "t_total" in st.get("direct_2k", {}) else None)
    (d / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    records.append(info)

# ------------------------------------------------------------------ report
meta = dict(
    gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
    steps=STEPS, guidance=GUIDANCE, base_res=BASE,
    t_encode_all_prompts=t_encode_all,
    devices=devices,
    note="text encoders released after prompt encoding (all paths alike); "
         "direct FLUX@4096 omitted as impractically slow",
    samples=records,
)
(ROOT / "timings.json").write_text(json.dumps(meta, indent=2))


def table(scale: str) -> str:
    lua_k, dir_k, res = f"lua_{scale}", f"direct_{scale}", scale.upper()
    lines = [
        f"| sample | LUA: base gen | LUA: upscale | LUA: decode | **LUA total** "
        f"| direct: gen | direct: decode | **direct total** | speedup | "
        f"peak VRAM LUA/direct (GB) |",
        "|" + "---|" * 10,
    ]
    tot_l, tot_d, n = 0.0, 0.0, 0
    for r in records:
        lua, dr = r["stages"][lua_k], r["stages"][dir_k]
        if "t_total" not in dr:
            lines.append(f"| {r['name']} | {lua['t_base_gen']} | {lua['t_upscale']} "
                         f"| {lua['t_decode']} | **{lua['t_total']}** | — | — "
                         f"| **{dr.get('error', 'failed')}** | — | {lua['peak_vram_gb']} / — |")
            continue
        sp = r[f"speedup_{scale}"]
        lines.append(
            f"| {r['name']} | {lua['t_base_gen']} | {lua['t_upscale']} | {lua['t_decode']} "
            f"| **{lua['t_total']}** | {dr['t_gen']} | {dr['t_decode']} "
            f"| **{dr['t_total']}** | ×{sp} | {lua['peak_vram_gb']} / {dr['peak_vram_gb']} |")
        tot_l += lua["t_total"]
        tot_d += dr["t_total"]
        n += 1
    if n:
        lines.append(f"| **mean** |  |  |  | **{tot_l / n:.1f}** |  |  "
                     f"| **{tot_d / n:.1f}** | ×{tot_d / tot_l:.2f} |  |")
    return "\n".join(lines)


def lua_table(scale: str) -> str:
    key = f"lua_{scale}"
    lines = [
        "| sample | base gen | upscale | decode | **LUA total** | peak VRAM (GB) |",
        "|" + "---|" * 6,
    ]
    tot, n = 0.0, 0
    for r in records:
        lua = r["stages"][key]
        lines.append(
            f"| {r['name']} | {lua['t_base_gen']} | {lua['t_upscale']} "
            f"| {lua['t_decode']} | **{lua['t_total']}** | {lua['peak_vram_gb']} |")
        tot += lua["t_total"]
        n += 1
    if n:
        lines.append(f"| **mean** |  |  |  | **{tot / n:.1f}** |  |")
    return "\n".join(lines)


report = f"""# LUA vs direct FLUX — 2K & 4K benchmark

Single GPU ({meta['gpu']}), torch {meta['torch']}, {STEPS} steps, guidance {GUIDANCE},
seconds; same prompt+seed for every path of a sample. Prompt encoding
({t_encode_all}s for all {len(JOBS)} prompts) is shared by all paths and excluded.
LUA path = FLUX@{BASE} -> LUA head -> tiled decode. Direct = FLUX@target -> tiled decode.

## 2K (2048x2048) — LUA vs direct

{table('2k')}

## 4K (4096x4096) — LUA path

Direct FLUX@4096 was intentionally omitted (~15-20 min per sample on this GPU;
1024->2048 already scales sampling by ~x6.6, so a 4096 baseline would be
roughly that much slower again over direct 2K).

{lua_table('4k')}

Note: with the same seed, the 1024-base and direct-2048/4096 trajectories start
from different-shaped noise, so images differ in composition — compare quality,
not pixel identity.
"""
(ROOT / "report.md").write_text(report)
print("BENCH DONE", flush=True)
