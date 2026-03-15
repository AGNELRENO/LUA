#!/usr/bin/env python3
"""
LUA (Latent Upscale Adapter) -- Interactive Gradio Demo

Fair comparison at the SAME output resolution:
  - LUA path:    FLUX@1024 -> LUA upscale -> VAE decode -> target image
  - Direct path: FLUX@target -> VAE decode -> target image

Toggle: 1K (base only), 2K (LUA x2 vs FLUX@2048), 4K (LUA x4 vs FLUX@4096)

Usage:
    pip install -r requirements.txt
    python gradio_demo.py
"""

from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Gradio temp dir
_gradio_tmp = Path.home() / ".cache" / "gradio_tmp"
_gradio_tmp.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(_gradio_tmp)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gradio as gr
from diffusers import FluxPipeline

from lua import load_model, upscale_latent

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FLUX_MODEL_ID = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-dev")
LUA_WEIGHTS   = os.environ.get("LUA_WEIGHTS", None)  # None = auto-download from HF
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE         = torch.bfloat16 if DEVICE == "cuda" else torch.float32

RES_MAP = {
    "1K (1024)": {"target": 1024, "head": None,  "scale": 1},
    "2K (2048)": {"target": 2048, "head": "x2",  "scale": 2},
    "4K (4096)": {"target": 4096, "head": "x4",  "scale": 4},
}

# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_flux_pipeline():
    print(f"[FLUX] Loading '{FLUX_MODEL_ID}' (dtype={DTYPE}) ...")
    pipe = FluxPipeline.from_pretrained(FLUX_MODEL_ID, torch_dtype=DTYPE)
    pipe.to(DEVICE)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    print("[FLUX] Ready.")
    return pipe


print("=" * 64)
print("  LUA Gradio Demo  --  loading models ...")
print("=" * 64)
pipe      = load_flux_pipeline()
lua_model = load_model(weights_path=LUA_WEIGHTS, device=DEVICE)
print("=" * 64)
print("  All models loaded. Launching UI ...")
print("=" * 64)


# ---------------------------------------------------------------------------
# Latent visualization
# ---------------------------------------------------------------------------
def visualize_latent(latent: torch.Tensor) -> Image.Image:
    """16-ch latent [1,16,H,W] -> 4x4 channel grid + false-color mean map."""
    lat = latent[0].float().cpu()
    C, H, W = lat.shape
    tiles = []
    for c in range(C):
        ch = lat[c]; lo, hi = ch.min(), ch.max()
        ch = (ch - lo) / (hi - lo) if hi - lo > 1e-6 else ch * 0
        tiles.append((ch.numpy() * 255).astype(np.uint8))

    gap = 2; rows, cols = 4, 4
    gh = rows * H + (rows - 1) * gap
    gw = cols * W + (cols - 1) * gap
    canvas = np.zeros((gh, gw), dtype=np.uint8)
    for i, t in enumerate(tiles[:16]):
        r, c = divmod(i, cols)
        canvas[r*(H+gap):r*(H+gap)+H, c*(W+gap):c*(W+gap)+W] = t

    mean = lat.mean(0).numpy()
    lo, hi = mean.min(), mean.max()
    n = (mean - lo) / (hi - lo) if hi - lo > 1e-6 else mean * 0
    rm = np.clip(np.where(n < .5, .3, (n-.5)*4), 0, 1)
    gm = np.clip(np.where(n < .5, n*2, 1-(n-.5)), 0, 1)
    bm = np.clip(np.where(n < .5, .5+n, 1-(n-.5)*2), 0, 1)
    cmap = np.stack([(rm*255).astype(np.uint8), (gm*255).astype(np.uint8),
                     (bm*255).astype(np.uint8)], -1)
    cpil = Image.fromarray(cmap).resize((int(W*(gh/H)), gh), Image.NEAREST)

    grid3 = np.stack([canvas]*3, -1)
    combined = np.concatenate(
        [grid3, np.zeros((gh, gap*3, 3), dtype=np.uint8), np.array(cpil)], 1)
    return Image.fromarray(combined)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(s): return f"{s:.2f}s"
def _sz(img): return f"{img.size[0]}x{img.size[1]}"

def _unpack_to_vae(pipe, packed, h, w):
    spatial = pipe._unpack_latents(
        packed, height=h, width=w, vae_scale_factor=pipe.vae_scale_factor)
    return (spatial / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor

def _decode(pipe, lat):
    dec = pipe.vae.decode(lat.to(dtype=DTYPE), return_dict=False)[0]
    return pipe.image_processor.postprocess(dec, output_type="pil")[0]


# ---------------------------------------------------------------------------
# Interactive side-by-side comparison with synced loupes
# ---------------------------------------------------------------------------
def _pil_to_b64(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _build_comparison_html(
    left_pil: Image.Image,
    right_pil: Image.Image,
    left_label: str = "LUA path",
    right_label: str = "Direct FLUX",
) -> str:
    def _fit(img, mx=1500):
        w, h = img.size
        if max(w, h) <= mx:
            return img
        r = mx / max(w, h)
        return img.resize((int(w * r), int(h * r)), Image.LANCZOS)

    left_pil, right_pil = _fit(left_pil), _fit(right_pil)
    b64_l = _pil_to_b64(left_pil, quality=88)
    b64_r = _pil_to_b64(right_pil, quality=88)
    w, h = left_pil.size
    aspect = h / w * 100
    uid = f"cmp{int(time.time()*1000)}"

    _mv = (
        "var r=this.getBoundingClientRect(),"
        "px=Math.max(0,Math.min(1,(event.clientX-r.left)/r.width)),"
        "py=Math.max(0,Math.min(1,(event.clientY-r.top)/r.height)),"
        "lps=this.parentElement.querySelectorAll('.lp'),Z=3,S=180;"
        "for(var i=0;lps.length>i;i++){"
        "var l=lps[i],pr=l.parentElement.getBoundingClientRect(),"
        "zW=pr.width*Z,zH=pr.height*Z;"
        "l.style.backgroundSize=zW+'px '+zH+'px';"
        "l.style.backgroundPosition=-(px*zW-S/2)+'px '+-(py*zH-S/2)+'px';"
        "l.style.left=Math.max(0,Math.min(pr.width-S,px*pr.width-S/2))+'px';"
        "l.style.top=Math.max(0,Math.min(pr.height-S,py*pr.height-S/2))+'px';"
        "l.style.display='block';}"
    )
    _lv = (
        "var lps=this.parentElement.querySelectorAll('.lp');"
        "for(var i=0;lps.length>i;i++){lps[i].style.display='none';}"
    )
    _ld = "this.parentElement.querySelector('.lp').style.backgroundImage='url('+this.src+')'"

    return f"""<style>
#{uid} {{ display:flex; gap:12px; width:100%; user-select:none;
           -webkit-user-select:none; }}
#{uid} .pn {{ flex:1; position:relative; border-radius:8px; overflow:hidden;
              box-shadow:0 2px 10px rgba(0,0,0,.3); cursor:crosshair; }}
#{uid} .pn .pd {{ width:100%; padding-bottom:{aspect:.2f}%; }}
#{uid} .pn img {{ position:absolute; top:0; left:0; width:100%; height:100%;
                  object-fit:cover; pointer-events:none; display:block; }}
#{uid} .pn .lb {{ position:absolute; top:10px; left:10px; padding:5px 12px;
                  background:rgba(0,0,0,.65); color:#fff; border-radius:6px;
                  font:600 13px/1 system-ui,sans-serif; z-index:5;
                  pointer-events:none; }}
#{uid} .lp {{ position:absolute; width:180px; height:180px; border-radius:50%;
              border:3px solid #fff; box-shadow:0 0 14px rgba(0,0,0,.6);
              pointer-events:none; display:none; z-index:20;
              background-repeat:no-repeat; }}
</style>
<div id="{uid}">
  <div class="pn" onmousemove="{_mv}" onmouseleave="{_lv}">
    <div class="pd"></div>
    <img src="data:image/jpeg;base64,{b64_l}" onload="{_ld}">
    <div class="lb">{left_label}</div>
    <div class="lp"></div>
  </div>
  <div class="pn" onmousemove="{_mv}" onmouseleave="{_lv}">
    <div class="pd"></div>
    <img src="data:image/jpeg;base64,{b64_r}" onload="{_ld}">
    <div class="lb">{right_label}</div>
    <div class="lp"></div>
  </div>
</div>"""


# ---------------------------------------------------------------------------
# Output slot indices  (9 outputs)
# ---------------------------------------------------------------------------
def _blank():
    return [None]*6 + [""]*2 + [""]


# ---------------------------------------------------------------------------
# Core inference (generator)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run(prompt, num_steps, guidance_scale, target_res, seed):
    if not prompt or not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    cfg = RES_MAP[target_res]
    target   = cfg["target"]
    head     = cfg["head"]
    scale    = cfg["scale"]
    base     = 1024
    need_cmp = target > base
    seed     = int(seed)

    T = {}
    lines = []
    def log(m):
        lines.append(m); return "\n".join(lines)
    out = _blank()

    n_steps = 5 if not need_cmp else 7

    # 1. FLUX @ 1024
    out[6] = log(f"[Step 1/{n_steps}] FLUX generating latent "
                 f"({base}x{base}, {int(num_steps)} steps) ...")
    yield tuple(out)

    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    packed = pipe(
        prompt=prompt, num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        width=base, height=base, output_type="latent", generator=gen,
    ).images
    T["flux_base"] = time.time() - t0

    vae_lat = _unpack_to_vae(pipe, packed, base, base)
    lat_shape = list(vae_lat.shape)

    out[6] = log(f"           Done in {_fmt(T['flux_base'])}  |  latent {lat_shape}")

    # 2. Visualize base latent
    out[6] = log(f"[Step 2/{n_steps}] Visualizing base latent "
                 f"({lat_shape[2]}x{lat_shape[3]}, 16 ch) ...")
    out[0] = visualize_latent(vae_lat)
    yield tuple(out)

    # 3. Decode base 1024
    out[6] = log(f"[Step 3/{n_steps}] Decoding base image ({base}x{base}) ...")
    yield tuple(out)

    t0 = time.time()
    base_pil = _decode(pipe, vae_lat)
    T["dec_base"] = time.time() - t0
    out[1] = base_pil
    out[6] = log(f"           Done in {_fmt(T['dec_base'])}  |  {_sz(base_pil)}")
    yield tuple(out)

    if not need_cmp:
        summary = (
            f"{'='*50}\n"
            f"  Base generation ({_sz(base_pil)})\n"
            f"{'─'*50}\n"
            f"  FLUX generation:   {_fmt(T['flux_base']):>8}\n"
            f"  VAE decode:        {_fmt(T['dec_base']):>8}\n"
            f"{'─'*50}\n"
            f"  Total:             {_fmt(T['flux_base']+T['dec_base']):>8}\n"
            f"{'='*50}"
        )
        out[7] = summary
        out[6] = log("\nDone (1K mode -- no upscaling comparison).")
        yield tuple(out)
        return

    # 4. LUA upscale
    out[6] = log(f"[Step 4/{n_steps}] LUA upscaling latent "
                 f"({head}, {scale}x -> {target}x{target}) ...")
    yield tuple(out)

    t0 = time.time()
    up_lat = upscale_latent(lua_model, vae_lat, head)
    T["lua"] = time.time() - t0
    up_shape = list(up_lat.shape)

    out[6] = log(f"           Done in {_fmt(T['lua'])}  |  latent {lat_shape} -> {up_shape}")
    out[2] = visualize_latent(up_lat)
    yield tuple(out)

    # 5. Decode LUA image
    out[6] = log(f"[Step 5/{n_steps}] Decoding LUA image ({target}x{target}) ...")
    yield tuple(out)

    t0 = time.time()
    lua_pil = _decode(pipe, up_lat)
    T["dec_lua"] = time.time() - t0

    out[3] = lua_pil
    out[6] = log(f"           Done in {_fmt(T['dec_lua'])}  |  {_sz(lua_pil)}")
    yield tuple(out)

    # 6. FLUX direct @ target
    out[6] = log(f"[Step 6/{n_steps}] FLUX generating directly at "
                 f"{target}x{target} (same prompt, same seed) ...")
    yield tuple(out)

    t0 = time.time()
    gen2 = torch.Generator(device="cpu").manual_seed(seed)
    packed2 = pipe(
        prompt=prompt, num_inference_steps=int(num_steps),
        guidance_scale=float(guidance_scale),
        width=target, height=target, output_type="latent", generator=gen2,
    ).images
    T["flux_direct"] = time.time() - t0

    vae_lat2 = _unpack_to_vae(pipe, packed2, target, target)
    out[4] = visualize_latent(vae_lat2)
    out[6] = log(f"           Done in {_fmt(T['flux_direct'])}")

    # 7. Decode direct
    out[6] = log(f"[Step 7/{n_steps}] Decoding direct FLUX image ({target}x{target}) ...")
    yield tuple(out)

    t0 = time.time()
    direct_pil = _decode(pipe, vae_lat2)
    T["dec_direct"] = time.time() - t0

    out[5] = direct_pil
    out[6] = log(f"           Done in {_fmt(T['dec_direct'])}  |  {_sz(direct_pil)}")

    # Summary
    lua_total    = T["flux_base"] + T["lua"] + T["dec_lua"]
    direct_total = T["flux_direct"] + T["dec_direct"]
    speedup      = direct_total / lua_total if lua_total > 0 else 0

    summary = (
        f"{'='*50}\n"
        f"  SAME-RESOLUTION COMPARISON  ({target}x{target})\n"
        f"{'='*50}\n\n"
        f"  LUA path  (FLUX@1024 -> LUA {head} -> decode)\n"
        f"  {'─'*46}\n"
        f"  FLUX @1024:          {_fmt(T['flux_base']):>8}\n"
        f"  LUA upscale ({head}):    {_fmt(T['lua']):>8}\n"
        f"  VAE decode @{target}:   {_fmt(T['dec_lua']):>8}\n"
        f"  {'─'*46}\n"
        f"  Total:               {_fmt(lua_total):>8}\n\n"
        f"  Direct path  (FLUX@{target} -> decode)\n"
        f"  {'─'*46}\n"
        f"  FLUX @{target}:         {_fmt(T['flux_direct']):>8}\n"
        f"  VAE decode @{target}:   {_fmt(T['dec_direct']):>8}\n"
        f"  {'─'*46}\n"
        f"  Total:               {_fmt(direct_total):>8}\n\n"
        f"  {'='*46}\n"
        f"  LUA is {speedup:.1f}x faster\n"
        f"  ({_fmt(lua_total)} vs {_fmt(direct_total)})\n"
        f"  Output: {target}x{target}  "
        f"({scale*scale}x pixels vs base 1024x1024)\n"
        f"{'='*50}"
    )
    out[7] = summary

    out[8] = _build_comparison_html(
        lua_pil, direct_pil,
        left_label=f"LUA {head}  ({_fmt(lua_total)})",
        right_label=f"Direct FLUX  ({_fmt(direct_total)})",
    )

    out[6] = log(f"\nAll steps complete.  "
                 f"LUA: {_fmt(lua_total)} vs Direct: {_fmt(direct_total)} "
                 f"({speedup:.1f}x faster)")
    yield tuple(out)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
CSS = """
.mono textarea {
    font-family: monospace !important; font-size: 13px !important;
    line-height: 1.5 !important;
}
.step-log textarea {
    font-family: monospace !important; font-size: 12px !important;
    line-height: 1.4 !important;
    background: #1a1a2e !important; color: #00ff88 !important;
}
"""

with gr.Blocks(title="LUA Demo", css=CSS) as demo:
    gr.Markdown(
        "# LUA -- Latent Upscale Adapter\n"
        "Fair comparison at the **same output resolution**:  \n"
        "**LUA path** (FLUX@1024 -> LUA upscale -> decode)  vs  "
        "**Direct path** (FLUX@target -> decode)"
    )

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(
                label="Prompt", lines=2,
                placeholder="A photorealistic cat astronaut floating in space, "
                            "Earth in the background, cinematic lighting",
            )
        with gr.Column(scale=1, min_width=180):
            seed = gr.Number(value=42, label="Seed", precision=0)
            btn  = gr.Button("Generate", variant="primary", size="lg")

    with gr.Row():
        num_steps  = gr.Slider(1, 50, value=28, step=1, label="Inference steps")
        guidance   = gr.Slider(0.0, 20.0, value=3.5, step=0.5, label="Guidance scale")
        target_res = gr.Radio(
            list(RES_MAP.keys()), value="2K (2048)",
            label="Output resolution",
        )

    step_log = gr.Textbox(
        label="Progress", lines=8, interactive=False, elem_classes=["step-log"],
    )
    gr.Markdown("---")

    gr.Markdown("### Base: FLUX@1024 -> Latent -> Decode")
    with gr.Row():
        latent_vis = gr.Image(label="Base latent (16-ch)", type="pil")
        base_img   = gr.Image(label="Base decode (1024)", type="pil")

    gr.Markdown("---")

    gr.Markdown("### LUA path: Latent -> LUA Upscale -> Decode")
    with gr.Row():
        lua_latent_vis = gr.Image(label="LUA-upscaled latent", type="pil")
        lua_img        = gr.Image(label="LUA decode (target res)", type="pil")

    gr.Markdown("---")

    gr.Markdown("### Direct path: FLUX@target -> Decode")
    with gr.Row():
        direct_latent_vis = gr.Image(label="Direct FLUX latent (16-ch)", type="pil")
        direct_img = gr.Image(label="FLUX direct (target res)", type="pil")

    gr.Markdown("---")

    gr.Markdown("### Comparison: LUA vs Direct  (hover to magnify and compare details)")
    with gr.Row():
        with gr.Column(scale=3):
            comparison_html = gr.HTML(value="<div style='color:#888;text-align:center;"
                                        "padding:60px 0'>Generate images to compare</div>")
        with gr.Column(scale=1, min_width=280):
            summary_box = gr.Textbox(
                label="Timing Summary", lines=22, interactive=False,
                elem_classes=["mono"],
            )

    btn.click(
        fn=run,
        inputs=[prompt, num_steps, guidance, target_res, seed],
        outputs=[
            latent_vis, base_img, lua_latent_vis, lua_img,
            direct_latent_vis, direct_img,
            step_log, summary_box, comparison_html,
        ],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
