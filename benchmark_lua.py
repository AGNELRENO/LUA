import time
import torch

from lua import load_model, upscale_latent


device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 60)
print("LUA MULTI-RUN BENCHMARK")
print("=" * 60)

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

latent = torch.randn(
    1, 16, 32, 32,
    device=device
)

print("Input:", latent.shape)

WARMUP = 3
RUNS = 10


def benchmark(head):
    print(f"\nBenchmarking {head}...")

    # Warm-up
    for _ in range(WARMUP):
        with torch.inference_mode():
            _ = upscale_latent(model, latent, head=head)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []

    for i in range(RUNS):
        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        with torch.inference_mode():
            output = upscale_latent(
                model,
                latent,
                head=head
            )

        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start
        times.append(elapsed)

    average = sum(times) / len(times)

    print("Output:", output.shape)
    print("Runs:", RUNS)
    print(f"Average: {average:.4f} seconds")
    print(f"Minimum: {min(times):.4f} seconds")
    print(f"Maximum: {max(times):.4f} seconds")

    return average


x2_avg = benchmark("x2")
x4_avg = benchmark("x4")


print("\n" + "=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(f"X2 average: {x2_avg:.4f} seconds")
print(f"X4 average: {x4_avg:.4f} seconds")