"""
Early exit heads and exit management for EdgeCAAI-Net.

Exit heads are attached at intermediate backbone depths to enable
confidence-based early stopping during inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentivePooling1D(nn.Module):
    """Lightweight attentive pooling for exit heads."""

    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (B, T, D)
        weights = F.softmax(self.attn(x), dim=1)  # (B, T, 1)
        return (weights * x).sum(dim=1)             # (B, D)


class EarlyExitHead(nn.Module):
    """
    Single early exit head: pooling + MLP + confidence estimator.

    Args:
        d_model: Input feature dimension.
        num_classes: Number of output classes.
        dropout: Dropout rate.
    """

    def __init__(self, d_model, num_classes, dropout=0.15):
        super().__init__()
        self.pooling = AttentivePooling1D(d_model)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )
        # Learned confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, T, D) sequence features from backbone.

        Returns:
            logits: (B, num_classes)
            embedding: (B, D) pooled embedding
        """
        embedding = self.pooling(x)
        logits = self.classifier(embedding)
        return logits, embedding

    def get_confidence(self, x):
        """Get learned confidence score."""
        embedding = self.pooling(x)
        return self.confidence_head(embedding).squeeze(-1)


class ExitManager(nn.Module):
    """
    Manages multiple early exit heads.

    Args:
        d_model: Feature dimension.
        num_classes: Number of classes.
        exit_positions: List of block indices where exits are placed.
        dropout: Dropout rate.
    """

    def __init__(self, d_model, num_classes, exit_positions, dropout=0.15):
        super().__init__()
        self.exit_positions = exit_positions
        self.exit_heads = nn.ModuleDict({
            str(pos): EarlyExitHead(d_model, num_classes, dropout)
            for pos in exit_positions
        })

    def get_exit(self, block_idx, x):
        """Get predictions from exit at given block index."""
        head = self.exit_heads[str(block_idx)]
        return head(x)

    def get_confidence(self, block_idx, x):
        """Get confidence score from exit at given block index."""
        head = self.exit_heads[str(block_idx)]
        return head.get_confidence(x)


def compute_deep_supervision_loss(exit_logits, targets, weights=None):
    """
    Compute weighted cross-entropy across all exits.

    Args:
        exit_logits: List of (B, C) logit tensors, one per exit.
        targets: (B,) ground truth labels.
        weights: Per-exit loss weights. Default: [0.2, 0.3, 0.5].

    Returns:
        Scalar loss.
    """
    if weights is None:
        weights = [0.2, 0.3, 0.5]

    assert len(exit_logits) == len(weights), \
        f"Got {len(exit_logits)} exits but {len(weights)} weights"

    loss = sum(
        w * F.cross_entropy(logits, targets)
        for w, logits in zip(weights, exit_logits)
    )
    return loss


def compute_budget_loss(gate_logits, compute_costs=None, lambda_budget=0.02):
    """
    Compute budget penalty to encourage early exits.

    L_budget = lambda * sum(p_e * c_e)

    Args:
        gate_logits: (B, num_exits) raw gating logits.
        compute_costs: Per-exit normalized compute costs.
        lambda_budget: Budget penalty strength.

    Returns:
        Scalar loss.
    """
    if compute_costs is None:
        compute_costs = [0.35, 0.70, 1.0]

    costs = torch.tensor(compute_costs, device=gate_logits.device, dtype=gate_logits.dtype)
    probs = F.softmax(gate_logits, dim=-1)  # (B, num_exits)
    expected_cost = (probs * costs).sum(dim=-1).mean()
    return lambda_budget * expected_cost
