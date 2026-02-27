"""
EdgeCAAI-Net evaluation script.

Loads a trained model and evaluates with early exit inference.
Computes macro-F1, balanced accuracy, confusion matrix, ECE,
and per-exit statistics.

Usage:
    python eval.py --config configs/fma_artist_disjoint.yaml \
        --checkpoint results/fma_artist_disjoint/checkpoints/best.pt \
        --threshold 0.8
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, balanced_accuracy_score, confusion_matrix, classification_report
)
import yaml

from models.edgecaai_net import EdgeCAAINet
from train import CachedMelDataset, load_config


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_ece(confidences, predictions, labels, n_bins=15):
    """
    Compute Expected Calibration Error.

    Args:
        confidences: (N,) confidence scores.
        predictions: (N,) predicted classes.
        labels: (N,) ground truth classes.
        n_bins: Number of calibration bins.

    Returns:
        ECE scalar.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences < hi)
        if not mask.any():
            continue
        bin_acc = (predictions[mask] == labels[mask]).mean()
        bin_conf = confidences[mask].mean()
        bin_size = mask.sum()
        ece += (bin_size / total) * abs(bin_acc - bin_conf)

    return ece


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_exits(model, dataloader, threshold, device, num_exits=3):
    """
    Evaluate model with confidence-based early exit.

    Returns detailed per-exit and aggregate metrics.
    """
    model.eval()

    all_preds = []
    all_labels = []
    all_confidences = []
    all_exit_indices = []

    for batch in dataloader:
        inputs = batch["input"].to(device)
        genres = batch["genre"].numpy()

        for i in range(inputs.shape[0]):
            single = inputs[i:i+1]
            result = model.inference(single, confidence_threshold=threshold)

            probs = F.softmax(result["logits"], dim=-1)
            pred = probs.argmax(dim=-1).cpu().item()
            conf = probs.max(dim=-1).values.cpu().item()

            all_preds.append(pred)
            all_labels.append(genres[i])
            all_confidences.append(conf)
            all_exit_indices.append(result["exit_index"])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)
    all_exit_indices = np.array(all_exit_indices)

    # Aggregate metrics
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    bal_acc = balanced_accuracy_score(all_labels, all_preds)
    ece = compute_ece(all_confidences, all_preds, all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    # Per-exit statistics
    exit_stats = {}
    for e in range(num_exits):
        mask = all_exit_indices == e
        count = mask.sum()
        if count > 0:
            exit_f1 = f1_score(all_labels[mask], all_preds[mask], average="macro",
                               zero_division=0)
            exit_acc = (all_preds[mask] == all_labels[mask]).mean()
            exit_conf = all_confidences[mask].mean()
        else:
            exit_f1 = exit_acc = exit_conf = 0.0

        exit_stats[f"exit_{e+1}"] = {
            "count": int(count),
            "percentage": float(count / len(all_labels) * 100),
            "macro_f1": float(exit_f1),
            "accuracy": float(exit_acc),
            "avg_confidence": float(exit_conf),
        }

    return {
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(bal_acc),
        "ece": float(ece),
        "confusion_matrix": cm.tolist(),
        "exit_stats": exit_stats,
        "total_samples": len(all_labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate EdgeCAAI-Net")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--split", default="test",
                        choices=["val", "test"])
    parser.add_argument("--output", default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    feat_cfg = config["features"]
    dataset_cfg = config["dataset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = EdgeCAAINet(
        num_classes=dataset_cfg["num_genres"],
        n_mels=feat_cfg["n_mels"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_blocks=model_cfg["n_blocks"],
        ffn_dim=model_cfg["ffn_dim"],
        stem_channels=model_cfg["stem_channels"],
        kernel_size=model_cfg["depthwise_kernel_size"],
        dropout=0.0,  # No dropout at inference
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Dataset
    split_file = os.path.join(
        dataset_cfg["split_dir"],
        dataset_cfg["split_files"][args.split]
    )
    dataset = CachedMelDataset(
        split_file, dataset_cfg["processed_dir"],
        augment=False, config=config
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=config.get("num_workers", 4),
    )
    print(f"{args.split} samples: {len(dataset)}")

    # Evaluate
    print(f"\nEvaluating with confidence threshold = {args.threshold}...")
    results = evaluate_with_exits(
        model, dataloader, args.threshold, device,
        num_exits=len(config.get("exits", {}).get("positions", [2, 4, 6])),
    )

    # Print results
    print(f"\n{'='*50}")
    print(f"Results (threshold={args.threshold}):")
    print(f"  Macro-F1:           {results['macro_f1']:.4f}")
    print(f"  Balanced Accuracy:  {results['balanced_accuracy']:.4f}")
    print(f"  ECE:                {results['ece']:.4f}")
    print(f"  Total samples:      {results['total_samples']}")

    print(f"\nExit distribution:")
    for exit_name, stats in results["exit_stats"].items():
        print(f"  {exit_name}: {stats['count']} ({stats['percentage']:.1f}%) "
              f"F1={stats['macro_f1']:.4f} acc={stats['accuracy']:.4f} "
              f"conf={stats['avg_confidence']:.4f}")

    # Load genre map for readable confusion matrix
    genre_map_path = os.path.join(
        dataset_cfg["split_dir"],
        f"{dataset_cfg['name']}_genre_map.json"
    )
    if os.path.exists(genre_map_path):
        with open(genre_map_path) as f:
            genre_map = json.load(f)
        genre_names = sorted(genre_map.keys(), key=lambda g: genre_map[g])
        print(f"\nClassification Report:")
        # Reconstruct for sklearn report
        print(f"  Genres: {genre_names}")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
