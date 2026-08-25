import time
import torch

from lua import load_model, upscale_latent


device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 50)
print("LUA BENCHMARK")
print("=" * 50)

print("Device:", device)

if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2
        ),
        "GB"
    )

print("\nLoading LUA model...")

model = load_model(
    device=device,
    dtype=torch.float32
)

print("LUA model loaded!")

latent = torch.randn(
    1, 16, 32, 32,
    device=device
)

print("\nInput latent:", latent.shape)


# X2 TEST
print("\nTesting X2...")

if device == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

with torch.inference_mode():
    output_x2 = upscale_latent(
        model,
        latent,
        head="x2"
    )

if device == "cuda":
    torch.cuda.synchronize()

x2_time = time.perf_counter() - start

print("X2 output:", output_x2.shape)
print(f"X2 inference time: {x2_time:.4f} seconds")


# X4 TEST
print("\nTesting X4...")

if device == "cuda":
    torch.cuda.synchronize()

start = time.perf_counter()

with torch.inference_mode():
    output_x4 = upscale_latent(
        model,
        latent,
        head="x4"
    )

if device == "cuda":
    torch.cuda.synchronize()

x4_time = time.perf_counter() - start

print("X4 output:", output_x4.shape)
print(f"X4 inference time: {x4_time:.4f} seconds")


print("\n" + "=" * 50)
print("BENCHMARK COMPLETE")
print("=" * 50)

print(f"X2: 32x32 -> 64x64 | {x2_time:.4f}s")
print(f"X4: 32x32 -> 128x128 | {x4_time:.4f}s")