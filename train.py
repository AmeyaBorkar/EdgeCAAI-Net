"""
EdgeCAAI-Net training entry point.

Supports:
- Deep supervision loss across all exits
- Budget-aware training penalty
- Artist invariance via GRL or GroupDRO
- SpecAugment, mixup, label smoothing
- Cosine LR schedule with warmup
- Early stopping on validation macro-F1

Usage:
    python train.py --config configs/fma_artist_disjoint.yaml
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score
from tqdm import tqdm
import yaml

from models.edgecaai_net import EdgeCAAINet
from models.exits import compute_deep_supervision_loss, compute_budget_loss
from models.artist_invariance import ArtistClassifier, GroupDRO


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CachedMelDataset(Dataset):
    """Dataset that loads cached log-mel .pt tensors."""

    def __init__(self, split_file, processed_dir, augment=False, config=None):
        with open(split_file, "r") as f:
            self.records = json.load(f)
        self.processed_dir = processed_dir
        self.augment = augment
        self.config = config or {}

        # Build index: list of (pt_path, genre_idx, artist_id)
        self.samples = []
        for rec in self.records:
            track_id = rec["track_id"]
            # Find all segments for this track
            seg_idx = 0
            while True:
                pt_path = os.path.join(processed_dir, f"{track_id}_seg{seg_idx}.pt")
                if os.path.exists(pt_path):
                    self.samples.append({
                        "path": pt_path,
                        "genre_idx": rec["genre_idx"],
                        "artist_id": rec.get("artist_id", -1),
                    })
                    seg_idx += 1
                else:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = torch.load(sample["path"], weights_only=True)
        log_mel = data["log_mel"]  # (n_mels, T)

        # Transpose to (T, n_mels) for model input
        log_mel = log_mel.squeeze(0).T  # (T, n_mels)

        # Apply augmentations
        if self.augment:
            log_mel = self._apply_augmentation(log_mel)

        return {
            "input": log_mel,
            "genre": sample["genre_idx"],
            "artist_id": sample["artist_id"],
        }

    def _apply_augmentation(self, log_mel):
        """Apply SpecAugment and random gain."""
        aug_cfg = self.config.get("augmentation", {})

        # SpecAugment: time masking
        if aug_cfg.get("spec_augment", True):
            T, F = log_mel.shape
            t_param = aug_cfg.get("time_mask_param", 20)
            f_param = aug_cfg.get("freq_mask_param", 10)

            # Time mask
            if T > t_param:
                t_start = random.randint(0, T - t_param)
                t_len = random.randint(0, t_param)
                log_mel[t_start:t_start + t_len, :] = 0

            # Frequency mask
            if F > f_param:
                f_start = random.randint(0, F - f_param)
                f_len = random.randint(0, f_param)
                log_mel[:, f_start:f_start + f_len] = 0

        # Random gain
        if aug_cfg.get("random_gain", True):
            gain_range = aug_cfg.get("gain_range", [0.8, 1.2])
            gain = random.uniform(*gain_range)
            log_mel = log_mel * gain

        return log_mel


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine LR schedule with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_config(config_path):
    """Load experiment config, merging with defaults if specified."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Merge with defaults
    defaults_file = config.get("defaults")
    if defaults_file:
        defaults_path = os.path.join(os.path.dirname(config_path), defaults_file)
        with open(defaults_path, "r") as f:
            defaults = yaml.safe_load(f)
        # Deep merge: config overrides defaults
        merged = _deep_merge(defaults, config)
        return merged

    return config


def _deep_merge(base, override):
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, scheduler, config, device,
                    artist_classifier=None, group_dro=None):
    """Train for one epoch."""
    model.train()
    if artist_classifier:
        artist_classifier.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    exit_cfg = config.get("exits", {})
    budget_cfg = config.get("budget", {})
    ai_cfg = config.get("artist_invariance", {})
    reg_cfg = config.get("regularization", {})

    label_smoothing = reg_cfg.get("label_smoothing", 0.0)

    for batch in tqdm(dataloader, desc="Training", leave=False):
        inputs = batch["input"].to(device)
        genres = batch["genre"].to(device)
        artist_ids = batch["artist_id"].to(device)

        optimizer.zero_grad()

        exits_enabled = exit_cfg.get("enabled", True)

        output = model(inputs, return_all_exits=True)

        if exits_enabled:
            # Deep supervision loss across all exits
            loss_cls = compute_deep_supervision_loss(
                output["logits"], genres,
                weights=exit_cfg.get("weights", [0.2, 0.3, 0.5]),
            )
        else:
            # Only use final exit output
            loss_cls = F.cross_entropy(output["logits"][-1], genres)

        # Budget penalty (only if exits and budget both enabled)
        budget_enabled = budget_cfg.get("enabled", True)
        if exits_enabled and budget_enabled:
            loss_budget = compute_budget_loss(
                output["gate_logits"],
                compute_costs=exit_cfg.get("compute_costs", [0.35, 0.70, 1.0]),
                lambda_budget=budget_cfg.get("lambda_budget", 0.02),
            )
        else:
            loss_budget = 0.0

        loss = loss_cls + loss_budget

        # Artist invariance loss
        if ai_cfg.get("method") == "grl" and artist_classifier is not None:
            artist_logits = artist_classifier(output["embedding"])
            valid_mask = artist_ids >= 0
            if valid_mask.any():
                loss_artist = ai_cfg.get("alpha", 0.2) * F.cross_entropy(
                    artist_logits[valid_mask], artist_ids[valid_mask]
                )
                loss = loss + loss_artist

        elif ai_cfg.get("method") == "groupdro" and group_dro is not None:
            per_sample_loss = F.cross_entropy(
                output["logits"][-1], genres, reduction="none"
            )
            loss_dro = group_dro(per_sample_loss, artist_ids)
            loss = loss + loss_dro

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()

        # Track predictions from final exit
        preds = output["logits"][-1].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(genres.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, macro_f1


@torch.no_grad()
def evaluate(model, dataloader, config, device):
    """Evaluate model on a dataset."""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0

    exit_cfg = config.get("exits", {})

    exits_enabled = exit_cfg.get("enabled", True)

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        inputs = batch["input"].to(device)
        genres = batch["genre"].to(device)

        output = model(inputs, return_all_exits=True)

        if exits_enabled:
            loss = compute_deep_supervision_loss(
                output["logits"], genres,
                weights=exit_cfg.get("weights", [0.2, 0.3, 0.5]),
            )
        else:
            loss = F.cross_entropy(output["logits"][-1], genres)
        total_loss += loss.item()

        preds = output["logits"][-1].argmax(dim=-1)
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


def main():
    parser = argparse.ArgumentParser(description="Train EdgeCAAI-Net")
    parser.add_argument("--config", required=True, help="Experiment config path")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    model_cfg = config["model"]
    feat_cfg = config["features"]
    dataset_cfg = config["dataset"]
    ai_cfg = config.get("artist_invariance", {})

    set_seed(config.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    model = EdgeCAAINet(
        num_classes=dataset_cfg["num_genres"],
        n_mels=feat_cfg["n_mels"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_blocks=model_cfg["n_blocks"],
        ffn_dim=model_cfg["ffn_dim"],
        stem_channels=model_cfg["stem_channels"],
        kernel_size=model_cfg["depthwise_kernel_size"],
        dropout=model_cfg["dropout"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Artist invariance module
    artist_classifier = None
    group_dro = None
    if ai_cfg.get("method") == "grl":
        # Count unique artists from training data
        artist_ids = set(r.get("artist_id", -1) for r in train_dataset.records)
        artist_ids.discard(-1)
        if artist_ids:
            num_artists = max(artist_ids) + 1
            artist_classifier = ArtistClassifier(
                model_cfg["d_model"] * 2, num_artists,
                alpha=ai_cfg.get("alpha", 0.2)
            ).to(device)
    elif ai_cfg.get("method") == "groupdro":
        artist_ids = set(r.get("artist_id", -1) for r in train_dataset.records)
        artist_ids.discard(-1)
        if artist_ids:
            num_artists = max(artist_ids) + 1
            group_dro = GroupDRO(num_artists).to(device)

    # Optimizer
    params = list(model.parameters())
    if artist_classifier:
        params += list(artist_classifier.parameters())

    optimizer = torch.optim.AdamW(
        params, lr=train_cfg["lr"],
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

    # Resume if specified
    start_epoch = 0
    best_f1 = 0.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        print(f"Resumed from epoch {start_epoch}, best F1: {best_f1:.4f}")

    # Training loop
    patience = train_cfg.get("early_stopping_patience", 10)
    patience_counter = 0

    print(f"\nStarting training for {train_cfg['epochs']} epochs...")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        t0 = time.time()

        train_loss, train_f1 = train_one_epoch(
            model, train_loader, optimizer, scheduler, config, device,
            artist_classifier=artist_classifier, group_dro=group_dro,
        )

        val_metrics = evaluate(model, val_loader, config, device)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch+1}/{train_cfg['epochs']} "
            f"[{elapsed:.1f}s] "
            f"train_loss={train_loss:.4f} train_F1={train_f1:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_F1={val_metrics['macro_f1']:.4f} "
            f"val_BA={val_metrics['balanced_accuracy']:.4f}"
        )

        # Checkpointing
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

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} (patience={patience})")
            break

    print(f"\nTraining complete. Best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
