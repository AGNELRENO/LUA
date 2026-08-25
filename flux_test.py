import torch
from diffusers import FluxPipeline

MODEL_ID = "black-forest-labs/FLUX.1-dev"

print("=" * 50)
print("LUA REPLICATION - FLUX TEST")
print("=" * 50)

print("GPU:", torch.cuda.get_device_name(0))
print("VRAM:", round(
    torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
), "GB")

print("\nLoading FLUX.1-dev...")
print("This may take a long time on the first run.")

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
)

print("\nFLUX model loaded successfully.")

pipe.enable_sequential_cpu_offload()

print("Sequential CPU offloading enabled.")

pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

print("VAE slicing enabled.")
print("VAE tiling enabled.")

print("\nFLUX TEST PASSED!")