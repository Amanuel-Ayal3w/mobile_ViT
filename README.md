# mobile_ViT

## Project Overview

This project implements and benchmarks the MobileViT family of models (MobileViT-XXS, MobileViT-XS, MobileViT-S) and compares them to a standard ResNet-50 baseline. MobileViT combines the strengths of convolutional neural networks (CNNs) and vision transformers (ViTs) for efficient and accurate image classification, especially on edge devices.

## Model Descriptions

- **MobileViT-XXS, XS, S**: Lightweight hybrid models that use both convolutional and transformer blocks. They are designed for high efficiency and accuracy on mobile and edge hardware.
- **ResNet-50**: A widely used CNN baseline for image classification, included for comparison.

## Benchmarking Methodology

All models were benchmarked on a CPU using random input tensors of shape (1, 3, 256, 256). The following metrics were measured:

- **Params (M)**: Number of trainable parameters (in millions)
- **FLOPs (G)**: Number of floating point operations (in billions, estimated via THOP)
- **Latency (ms)**: Average inference time per image (milliseconds)
- **Throughput (FPS)**: Images processed per second

### Benchmark Script

The benchmarking is performed by `benchmark.py`, which:
- Runs a warmup phase
- Measures average latency and throughput over 50 runs
- Reports parameter count and FLOPs

## Results

| Model          | Params (M) | FLOPs (G) | Latency (ms) | Throughput (FPS) |
|--------------- |----------- |-----------|--------------|------------------|
| MobileViT-XXS  | 1.33       | 0.35      | 18.92        | 52.86            |
| MobileViT-XS   | 2.38       | 0.93      | 27.68        | 36.12            |
| MobileViT-S    | 5.64       | 1.79      | 47.97        | 20.85            |
| ResNet-50      | 25.56      | 5.40      | 59.41        | 16.83            |

## Analysis

- **Parameter Efficiency**: MobileViT models use far fewer parameters than ResNet-50, making them suitable for resource-constrained environments.
- **Speed**: MobileViT-XXS and XS are significantly faster than ResNet-50, with much higher throughput.
- **FLOPs**: MobileViT models require fewer FLOPs, indicating lower computational cost.
- **Scalability**: As model size increases (XXS → XS → S), accuracy and computational cost both increase, but all MobileViT variants remain more efficient than ResNet-50.

## Mathematical Flow Explanation

1. **Input**: (B, C, H, W) — Standard image batch.
2. **Unfold**: (B, P, N, d) — Reshape into patches. P = pixels per patch, N = number of patches. This treats local patches as a sequence while preserving spatial arrangement.
3. **Transform**: Transformers process 'N' (global) while 'P' is batch-dim-like, allowing global information mixing across patches.
4. **Fold**: Return to (B, C, H, W) — Reconstructs the feature map from the transformed patches.

## Deployment Notes

- **CPU vs GPU**: Reshaping (rearrange/permute) is memory-bound. CPUs struggle more compared to pure contiguous convolutions. GPUs handle parallelism better but have some overhead.
- **Edge/NPU**: Standard ops (Conv, Linear, Softmax) are CoreML/TFLite friendly. 'Unfold/Fold' are often compiled efficiently on NPUs compared to pure ViT attention matrices.

## How to Run

1. Install requirements: `pip install -r requirements.txt`
2. Run benchmarks: `python3 benchmark.py`
3. Review results in the console output.

---
For more details, see the code in `mobilevit.py` and `benchmark.py`.