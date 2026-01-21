import torch
import time
import torch.nn as nn
from mobilevit import mobilevit_xxs, mobilevit_xs, mobilevit_s
from thop import profile

# Try to import thop for FLOPs calculation, handle if missing
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: 'thop' library not found. FLOPs will not be calculated.")
    print("To install: pip install thop")

# Try to import torchvision for ResNet50 baseline
try:
    from torchvision.models import resnet50
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("Warning: 'torchvision' not found. ResNet50 baseline will be skipped.")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_flops_g(model, input_size, device):
    if not THOP_AVAILABLE:
        return "N/A"
    
    input_sample = torch.randn(1, *input_size).to(device)
    # thop.profile returns (macs, params). MACs is approx FLOPs / 2 usually, 
    # but thop usually reports MACs. GFLOPs ≈ GMACs * 2. 
    # However, common convention often treats them interchangeably or checks the specific lib output.
    # We will report GMACs as GFLOPs proxy or explicitly as G-FLOPs if we multiply by 2.
    # Let's stick to thop's output / 1e9 for "G".
    macs, _ = profile(model, inputs=(input_sample, ), verbose=False)
    return f"{macs / 1e9:.2f}"

def benchmark_model(model_name, model, input_size=(3, 256, 256), device='cpu', warmup=10, runs=50):
    model.to(device)
    model.eval()
    
    # Input tensor
    input_tensor = torch.randn(1, *input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # Sync before timing (if cuda)
    if device == 'cuda':
        torch.cuda.synchronize()
        
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)
            
    # Sync after timing (if cuda)
    if device == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency_ms = (total_time / runs) * 1000
    throughput_fps = 1000 / avg_latency_ms
    
    params_m = count_parameters(model) / 1e6
    flops_g = get_flops_g(model, input_size, device)
    
    return {
        "Model": model_name,
        "Params (M)": f"{params_m:.2f}",
        "FLOPs (G)": flops_g,
        "Latency (ms)": f"{avg_latency_ms:.2f}",
        "Throughput (FPS)": f"{throughput_fps:.2f}"
    }

def print_table(results):
    # dynamic column width
    headers = ["Model", "Params (M)", "FLOPs (G)", "Latency (ms)", "Throughput (FPS)"]
    widths = [max(len(h), max(len(str(r[h])) for r in results)) + 2 for h in headers]
    
    # Header
    header_str = "".join(h.ljust(w) for h, w in zip(headers, widths))
    print("-" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    
    # Rows
    for r in results:
        print("".join(str(r[h]).ljust(w) for h, w in zip(headers, widths)))
    print("-" * len(header_str))

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on device: {device.upper()}\n")
    
    models_to_test = [
        ("MobileViT-XXS", mobilevit_xxs()),
        ("MobileViT-XS", mobilevit_xs()),
        ("MobileViT-S", mobilevit_s())
    ]
    
    if TORCHVISION_AVAILABLE:
        # ResNet50 usually takes 224x224, but MobileViT here is set for 256x256. 
        # We'll use 256x256 for fair comparison or 224 if strictly standard.
        # Let's use 256x256 to match the MobileViT config we define.
        resnet = resnet50(pretrained=False)
        # Adjust first conv if necessary? Standard ResNet takes 3 channels.
        models_to_test.append(("ResNet-50", resnet))
        
    results = []
    
    for name, model in models_to_test:
        print(f"Benchmarking {name}...")
        res = benchmark_model(name, model, input_size=(3, 256, 256), device=device)
        results.append(res)
        
    print("\nBenchmark Results:")
    print_table(results)

    print("\n" + "="*50)
    print("MATHEMATICAL FLOW EXPLANATION")
    print("="*50)
    print("1. Input: (B, C, H, W)")
    print("   Standard image batch.")
    print("2. Unfold: (B, P, N, d)")
    print("   Reshape into patches. P = pixels per patch, N = number of patches.")
    print("   This treats local patches as a sequence without losing spatial arrangement entirely.")
    print("3. Transform: Transformers process 'N' (global) while 'P' is batch-dim-like.")
    print("   Allows global information mixing across patches.")
    print("4. Fold: Return to (B, C, H, W)")
    print("   Reconstructs the feature map from the transformed patches.")
    print("\nNOTE ON DEPLOYMENT:")
    print("- CPU vs GPU: Reshaping (rearrange/permute) is memory-bound. CPUs struggle more")
    print("  compared to pure contiguous convs. GPUs handle parallelism better but overhead exists.")
    print("- Edge/NPU: Standard ops (Conv, Linear, Softmax) are CoreML/TFLite friendly.")
    print("  'Unfold/Fold' are often compiled efficiently on NPUs compared to pure ViT attention matrices.")

if __name__ == "__main__":
    main()
