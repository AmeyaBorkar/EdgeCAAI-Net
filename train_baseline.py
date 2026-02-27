"""
Baseline model training script.

Simplified training loop for baseline models (no early exits, no budget loss,
no artist invariance). Uses plain cross-entropy loss.

Usage:
    python train_baseline.py --config configs/baselines/tiny_cnn_track.yaml
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score
from tqdm import tqdm

from train import CachedMelDataset, load_config, set_seed, get_cosine_schedule_with_warmup
from models.baselines.tiny_cnn import TinyCNN
from models.baselines.small_crnn import SmallCRNN
from models.baselines.mobilenet_baseline import MobileNetBaseline
from models.baselines.tiny_transformer import TinyTransformer


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(model_cfg, num_classes, feat_cfg):
    """Create a baseline model from config."""
    name = model_cfg["name"]
    n_mels = feat_cfg["n_mels"]

    if name == "tiny_cnn":
        return TinyCNN(num_classes=num_classes, n_mels=n_mels)
    elif name == "small_crnn":
        return SmallCRNN(
            num_classes=num_classes, n_mels=n_mels,
            gru_hidden=model_cfg.get("gru_hidden", 128),
        )
    elif name == "mobilenet":
        return MobileNetBaseline(
            num_classes=num_classes,
            pretrained=model_cfg.get("pretrained", False),
        )
    elif name == "tiny_transformer":
        return TinyTransformer(
            num_classes=num_classes, n_mels=n_mels,
            d_model=model_cfg.get("d_model", 128),
            n_heads=model_cfg.get("n_heads", 4),
            n_blocks=model_cfg.get("n_blocks", 4),
            ffn_dim=model_cfg.get("ffn_dim", 256),
            dropout=model_cfg.get("dropout", 0.15),
        )
    else:
        raise ValueError(f"Unknown baseline model: {name}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, scheduler, device, label_smoothing=0.0):
    """Train for one epoch with plain cross-entropy."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        inputs = batch["input"].to(device)
        genres = batch["genre"].to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = F.cross_entropy(logits, genres, label_smoothing=label_smoothing)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(genres.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate baseline model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        inputs = batch["input"].to(device)
        genres = batch["genre"].to(device)

        logits = model(inputs)
        loss = F.cross_entropy(logits, genres)
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(genres.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    return {
        "loss": avg_loss,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
    }


def run_training(config_path, resume=None):
    """Run a full baseline training from a config path. Returns best F1."""
    config = load_config(config_path)
    train_cfg = config["training"]
    model_cfg = config["model"]
    feat_cfg = config["features"]
    dataset_cfg = config["dataset"]
    reg_cfg = config.get("regularization", {})

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training: {model_cfg['name']} | {config.get('experiment', {}).get('name', 'unknown')}")
    print(f"Device: {device}")

    # Datasets
    train_split = os.path.join(dataset_cfg["split_dir"],
                               dataset_cfg["split_files"]["train"])
    val_split = os.path.join(dataset_cfg["split_dir"],
                             dataset_cfg["split_files"]["val"])

    train_dataset = CachedMelDataset(
        train_split, dataset_cfg["processed_dir"],
        augment=True, config=config
    )
    val_dataset = CachedMelDataset(
        val_split, dataset_cfg["processed_dir"],
        augment=False, config=config
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"],
        shuffle=True, num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"],
        shuffle=False, num_workers=config.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    model = create_model(model_cfg, dataset_cfg["num_genres"], feat_cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"]
    )

    # LR scheduler
    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = int(total_steps * train_cfg.get("warmup_fraction", 0.05))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Checkpointing setup
    results_cfg = config.get("results", {})
    ckpt_dir = results_cfg.get("checkpoint_dir", "results/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    label_smoothing = reg_cfg.get("label_smoothing", 0.0)

    # Resume
    start_epoch = 0
    best_f1 = 0.0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # Training loop
    patience = train_cfg.get("early_stopping_patience", 10)
    patience_counter = 0

    print(f"Starting training for {train_cfg['epochs']} epochs...")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        t0 = time.time()

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, label_smoothing
        )
        val_metrics = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1}/{train_cfg['epochs']} "
            f"[{elapsed:.1f}s] "
            f"train_loss={train_loss:.4f} train_F1={train_f1:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_F1={val_metrics['macro_f1']:.4f} "
            f"val_BA={val_metrics['balanced_accuracy']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "config": config,
            }, os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> New best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break

    print(f"Training complete. Best val F1: {best_f1:.4f}")
    return best_f1


def main():
    parser = argparse.ArgumentParser(description="Train baseline model")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    run_training(args.config, args.resume)


if __name__ == "__main__":
    main()
