# LUA Replication Results

## 1. Objective

This project replicates and evaluates the core Latent Upscale Adapter (LUA)
implementation from the official LUA repository.

The replication focuses on verifying the LUA latent upscaling module,
including its x2 and x4 upscaling heads.

---

## 2. Hardware and Software

### Hardware

- GPU: NVIDIA GeForce RTX 2050
- VRAM: 4 GB
- Device: CUDA

### Software

- Python: 3.11
- PyTorch: Installed with CUDA support
- Diffusers: 0.40.0
- Gradio: 6.25.0

---

## 3. LUA Model

The LUA model was successfully loaded locally.

- Parameters: approximately 54.9M
- Available heads:
  - x2
  - x4

The model successfully performed latent upscaling on the RTX 2050.

---

## 4. x2 Upscaling Results

### Input

Latent tensor:

32 × 32

### Output

Latent tensor:

64 × 64

This corresponds to the x2 upscaling operation.

### Benchmark

10 inference runs were performed after warm-up.

- Average: 0.1650 seconds
- Minimum: 0.1519 seconds
- Maximum: 0.2044 seconds

### Result

**PASS**

---

## 5. x4 Upscaling Results

### Input

Latent tensor:

32 × 32

### Output

Latent tensor:

128 × 128

This corresponds to the x4 upscaling operation.

### Benchmark

10 inference runs were performed after warm-up.

- Average: 0.1728 seconds
- Minimum: 0.1612 seconds
- Maximum: 0.1996 seconds

### Result

**PASS**

---

## 6. Inference and Observations

The LUA model successfully performed both x2 and x4 latent
upscaling on the NVIDIA RTX 2050.

The output tensor dimensions matched the expected scaling factors:

- x2: 32 × 32 → 64 × 64
- x4: 32 × 32 → 128 × 128

The measured average latency was:

- x2: 0.1650 seconds
- x4: 0.1728 seconds

The x4 operation took slightly longer than x2 in the multi-run
benchmark.

These timings represent the LUA latent upscaling operation only.
They do not include FLUX generation or VAE decoding.

### Visual quality

Full visual quality evaluation is currently pending because the
FLUX.1-dev model has not yet been completely downloaded.

Therefore, no claims are made yet regarding:

- image sharpness
- texture quality
- preservation of objects
- artifacts
- 2K/4K visual quality

These will be evaluated after the complete FLUX + LUA pipeline is
available.

---

## 7. Current Limitations

The full FLUX.1-dev model could not be completely downloaded because
of network and storage/cache limitations.

Therefore, the following have not yet been completed:

- FLUX 1024 × 1024 generation
- FLUX latent → LUA integration using a generated image
- VAE decoding to 2K
- VAE decoding to 4K
- visual side-by-side comparison
- full end-to-end inference timing
- Gradio comparison demo

---

## 8. Future Direction

The remaining work is to complete the FLUX + LUA end-to-end pipeline.

Planned workflow:

FLUX.1-dev
→ 1024 × 1024 base generation
→ latent extraction
→ LUA x2 / x4
→ VAE decoding
→ 2048 × 2048 / 4096 × 4096 output

After completing the pipeline, the following will be evaluated:

1. Side-by-side visual comparisons.
2. x2 and x4 output quality.
3. Full end-to-end inference time.
4. VRAM usage.
5. Artifacts and quality limitations.
6. Comparison with direct FLUX generation where feasible.

---

## 9. Replication Status

### Completed

- Repository setup
- Python environment
- CUDA/GPU setup
- LUA model loading
- LUA x2 testing
- LUA x4 testing
- Multi-run latency benchmarking
- GitHub replication repository

### Pending

- Complete FLUX.1-dev download
- End-to-end FLUX + LUA generation
- Visual comparisons
- Final 2K/4K output evaluation

**Core LUA replication: COMPLETED**

**Full end-to-end replication: IN PROGRESS**