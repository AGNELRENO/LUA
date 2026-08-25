import torch
from lua import load_model, upscale_latent

print("=" * 50)
print("LUA UPSCALING TEST")
print("=" * 50)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

print("\nLoading LUA model...")

model = load_model(
    device=device,
    dtype=torch.float32
)

print("LUA model loaded!")

# Small synthetic FLUX VAE latent
latent = torch.randn(
    1,
    16,
    32,
    32,
    device=device,
    dtype=torch.float32
)

print("\nInput latent shape:", latent.shape)

# x2 test
print("\nTesting x2 upscaling...")

with torch.no_grad():
    output_x2 = upscale_latent(
        model,
        latent,
        head="x2"
    )

print("x2 output shape:", output_x2.shape)

# x4 test
print("\nTesting x4 upscaling...")

with torch.no_grad():
    output_x4 = upscale_latent(
        model,
        latent,
        head="x4"
    )

print("x4 output shape:", output_x4.shape)

print("\n" + "=" * 50)
print("LUA UPSCALING TEST COMPLETE")
print("=" * 50)