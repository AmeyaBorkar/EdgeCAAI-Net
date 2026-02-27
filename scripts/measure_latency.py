"""
Standardized latency benchmarking script for EdgeCAAI-Net.

Measures CPU inference latency per exit path and average latency
under a given confidence threshold policy.

Usage:
    python scripts/measure_latency.py --config configs/default.yaml \
        --checkpoint results/fma_artist_disjoint/checkpoints/best.pt
"""

import argparse
import platform
import time

import numpy as np
import torch
import yaml

from models.edgecaai_net import EdgeCAAINet


def measure_single_exit_latency(model, dummy_input, exit_index, num_runs=100):
    """Measure latency for a specific exit path."""
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input, return_all_exits=True)

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy_input, return_all_exits=True)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    return np.array(latencies)


def measure_adaptive_latency(model, dummy_input, threshold=0.8, num_runs=100):
    """Measure latency with confidence-based early exit."""
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model.inference(dummy_input, confidence_threshold=threshold)

    latencies = []
    exit_counts = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            result = model.inference(dummy_input, confidence_threshold=threshold)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        exit_counts.append(result["exit_index"])

    return np.array(latencies), exit_counts


def get_hardware_info():
    """Collect hardware info for reporting."""
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
    }


def main():
    parser = argparse.ArgumentParser(description="Latency benchmarking")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Model checkpoint path (optional)")
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    audio_cfg = config["audio"]
    feat_cfg = config["features"]

    # Create model
    model = EdgeCAAINet(
        num_classes=8,
        n_mels=feat_cfg["n_mels"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_blocks=model_cfg["n_blocks"],
        ffn_dim=model_cfg["ffn_dim"],
        stem_channels=model_cfg["stem_channels"],
        kernel_size=model_cfg["depthwise_kernel_size"],
        dropout=0.0,  # No dropout at inference
    )

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])

    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Create dummy input (3-second segment)
    sr = audio_cfg["sample_rate"]
    seg_len = audio_cfg["segment_length"]
    n_frames = int(seg_len * sr / feat_cfg["hop_length"]) + 1
    n_mels = feat_cfg["n_mels"]
    dummy_input = torch.randn(1, n_frames, n_mels)

    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Segment length: {seg_len}s @ {sr} Hz")

    # Hardware info
    hw_info = get_hardware_info()
    print(f"\nHardware: {hw_info['processor']}")
    print(f"Platform: {hw_info['platform']}")

    # Measure full model latency
    print(f"\n{'='*50}")
    print(f"Full model latency ({args.num_runs} runs):")
    latencies = measure_single_exit_latency(
        model, dummy_input, exit_index=-1, num_runs=args.num_runs
    )
    print(f"  Mean: {latencies.mean():.2f} ms")
    print(f"  Std:  {latencies.std():.2f} ms")
    print(f"  P50:  {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95:  {np.percentile(latencies, 95):.2f} ms")

    # Measure adaptive latency
    print(f"\n{'='*50}")
    print(f"Adaptive latency (threshold={args.threshold}):")
    latencies, exit_counts = measure_adaptive_latency(
        model, dummy_input, threshold=args.threshold, num_runs=args.num_runs
    )
    print(f"  Mean: {latencies.mean():.2f} ms")
    print(f"  Std:  {latencies.std():.2f} ms")

    # Exit distribution
    exit_counts = np.array(exit_counts)
    for i in range(3):
        pct = (exit_counts == i).sum() / len(exit_counts) * 100
        print(f"  Exit {i+1}: {pct:.1f}%")


if __name__ == "__main__":
    main()
