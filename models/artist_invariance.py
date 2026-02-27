"""
Artist invariance modules for EdgeCAAI-Net.

Prevents the model from using artist identity as a shortcut for genre prediction.
Two approaches: Gradient Reversal Layer (GRL) and Group DRO.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class _GradientReversal(Function):
    """Gradient reversal function for adversarial training."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for artist-invariant representations.

    During forward pass: identity function.
    During backward pass: reverses and scales gradients by alpha.

    Args:
        alpha: Gradient reversal strength (recommended 0.1-0.5).
    """

    def __init__(self, alpha=0.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return _GradientReversal.apply(x, self.alpha)

    def set_alpha(self, alpha):
        """Update reversal strength (e.g., for scheduling)."""
        self.alpha = alpha


class ArtistClassifier(nn.Module):
    """
    Auxiliary artist classifier attached via GRL.

    Args:
        input_dim: Embedding dimension (2*d_model for attentive stats pooling).
        num_artists: Number of unique artists.
        alpha: GRL strength.
    """

    def __init__(self, input_dim, num_artists, alpha=0.2):
        super().__init__()
        self.grl = GradientReversalLayer(alpha)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_artists),
        )

    def forward(self, embedding):
        """
        Args:
            embedding: (B, input_dim) shared embedding from backbone.

        Returns:
            (B, num_artists) artist logits.
        """
        reversed_emb = self.grl(embedding)
        return self.classifier(reversed_emb)


class GroupDRO(nn.Module):
    """
    Group Distributionally Robust Optimization.

    Treats each artist as a group and optimizes the worst-group loss.
    Maintains per-group weights that are updated via exponentiated gradient.

    Args:
        num_groups: Number of artist groups.
        step_size: Exponentiated gradient step size for weight updates.
    """

    def __init__(self, num_groups, step_size=0.01):
        super().__init__()
        self.num_groups = num_groups
        self.step_size = step_size
        # Uniform initialization of group weights
        self.register_buffer(
            'group_weights',
            torch.ones(num_groups) / num_groups
        )

    @torch.no_grad()
    def update_weights(self, group_losses):
        """
        Update group weights using exponentiated gradient ascent.

        Args:
            group_losses: (num_groups,) per-group average losses.
        """
        self.group_weights *= torch.exp(self.step_size * group_losses)
        self.group_weights /= self.group_weights.sum()

    def forward(self, losses, group_ids):
        """
        Compute GroupDRO loss.

        Args:
            losses: (B,) per-sample losses (unreduced).
            group_ids: (B,) artist/group ID per sample.

        Returns:
            Scalar weighted loss.
        """
        # Compute per-group average loss
        group_losses = torch.zeros(self.num_groups, device=losses.device)
        group_counts = torch.zeros(self.num_groups, device=losses.device)

        for g in range(self.num_groups):
            mask = group_ids == g
            if mask.any():
                group_losses[g] = losses[mask].mean()
                group_counts[g] = mask.sum()

        # Update weights based on group losses
        active_mask = group_counts > 0
        if active_mask.any():
            self.update_weights(group_losses)

        # Weighted combination of group losses
        weighted_loss = (self.group_weights * group_losses).sum()
        return weighted_loss


def compute_artist_invariance_loss(
    embedding, artist_labels, artist_classifier, alpha=0.2
):
    """
    Compute artist invariance loss using GRL.

    Args:
        embedding: (B, D) shared backbone embedding.
        artist_labels: (B,) artist IDs.
        artist_classifier: ArtistClassifier module.
        alpha: Loss weight.

    Returns:
        Scalar artist classification loss (with reversed gradients).
    """
    artist_logits = artist_classifier(embedding)
    return alpha * F.cross_entropy(artist_logits, artist_labels)
