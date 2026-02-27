"""
EdgeCAAI-Net: Lightweight Compute-Adaptive, Artist-Invariant Music Genre Classifier.

Architecture: Tiny Conformer-Lite backbone with multi-exit inference.
- Stem: depthwise-separable 2D convolution
- 6 Conformer-Lite blocks (depthwise temporal conv + multi-head attention + FFN)
- Attentive statistics pooling
- Compact classifier head
- 3 early exit points (after blocks 2, 4, 6)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .exits import EarlyExitHead, ExitManager


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable 2D convolution for the stem."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class Stem(nn.Module):
    """Stem layer: projects spectrogram into model feature dimension."""

    def __init__(self, n_mels=64, d_model=128, stem_channels=64):
        super().__init__()
        self.conv = DepthwiseSeparableConv2d(
            1, stem_channels, kernel_size=3, stride=(1, 1), padding=1
        )
        # Collapse frequency axis into d_model via linear projection
        self.proj = nn.Linear(stem_channels * n_mels, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, F) log-mel spectrogram
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.conv(x)    # (B, stem_channels, T, F)
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)  # (B, T, C*F)
        x = self.proj(x)    # (B, T, d_model)
        return self.norm(x)


class DepthwiseTemporalConv(nn.Module):
    """Depthwise temporal convolution for local pattern capture."""

    def __init__(self, d_model, kernel_size=11):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=padding, groups=d_model, bias=False
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.act(self.bn(self.conv(x)))
        return x.transpose(1, 2)  # (B, T, D)


class ConformerLiteBlock(nn.Module):
    """Single Conformer-Lite block: temporal conv + attention + FFN."""

    def __init__(self, d_model=128, n_heads=4, ffn_dim=256,
                 kernel_size=11, dropout=0.15):
        super().__init__()
        # Depthwise temporal convolution
        self.conv_norm = nn.LayerNorm(d_model)
        self.temporal_conv = DepthwiseTemporalConv(d_model, kernel_size)

        # Multi-head self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feed-forward network
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Temporal convolution with residual
        residual = x
        x = self.conv_norm(x)
        x = self.temporal_conv(x) + residual

        # Self-attention with residual
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = attn_out + residual

        # FFN with residual
        residual = x
        x = self.ffn(self.ffn_norm(x)) + residual

        return x


class AttentiveStatsPooling(nn.Module):
    """Attentive statistics pooling: weighted mean + std."""

    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: (B, T, D)
        attn_weights = self.attention(x)          # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        mean = (attn_weights * x).sum(dim=1)      # (B, D)
        var = (attn_weights * (x - mean.unsqueeze(1)) ** 2).sum(dim=1)
        std = (var + 1e-6).sqrt()
        return torch.cat([mean, std], dim=1)       # (B, 2*D)


class ClassifierHead(nn.Module):
    """Compact MLP classifier head."""

    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.15):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.head(x)


class EdgeCAAINet(nn.Module):
    """
    EdgeCAAI-Net: Tiny Conformer-Lite with multi-exit inference.

    Args:
        num_classes: Number of genre classes.
        n_mels: Number of mel frequency bins.
        d_model: Model hidden dimension.
        n_heads: Number of attention heads.
        n_blocks: Number of conformer-lite blocks.
        ffn_dim: FFN intermediate dimension.
        stem_channels: Stem convolution output channels.
        kernel_size: Depthwise temporal conv kernel size.
        dropout: Dropout rate.
        exit_positions: Block indices after which to place exits.
        num_artists: Number of artists (for artist invariance head, 0 to disable).
    """

    def __init__(
        self,
        num_classes=8,
        n_mels=64,
        d_model=128,
        n_heads=4,
        n_blocks=6,
        ffn_dim=256,
        stem_channels=64,
        kernel_size=11,
        dropout=0.15,
        exit_positions=None,
        num_artists=0,
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.exit_positions = exit_positions or [2, 4, 6]
        self.num_artists = num_artists

        # Stem
        self.stem = Stem(n_mels, d_model, stem_channels)

        # Core blocks
        self.blocks = nn.ModuleList([
            ConformerLiteBlock(d_model, n_heads, ffn_dim, kernel_size, dropout)
            for _ in range(n_blocks)
        ])

        # Final pooling and classifier (exit 3 / final exit)
        self.final_pooling = AttentiveStatsPooling(d_model)
        self.classifier = ClassifierHead(d_model * 2, d_model, num_classes, dropout)

        # Early exit heads (exits 1 and 2)
        self.exit_manager = ExitManager(
            d_model=d_model,
            num_classes=num_classes,
            exit_positions=self.exit_positions[:-1],  # exclude final
            dropout=dropout,
        )

        # Optional artist classification head (for GRL training)
        self.artist_head = None
        if num_artists > 0:
            self.artist_head = nn.Linear(d_model * 2, num_artists)

        # Gating network for budget-aware training
        self.gate = nn.Linear(d_model * 2, len(self.exit_positions))

    def forward(self, x, return_all_exits=True):
        """
        Args:
            x: (B, T, F) log-mel spectrogram.
            return_all_exits: If True, return logits from all exits (training).
                              If False, use confidence-based early exit (inference).

        Returns:
            dict with keys:
                'logits': list of logit tensors per exit
                'gate_logits': gating logits for budget loss
                'embedding': final embedding (for artist head)
        """
        x = self.stem(x)

        exit_logits = []
        exit_embeddings = []

        for i, block in enumerate(self.blocks):
            x = block(x)
            block_idx = i + 1  # 1-indexed

            # Check if this is an early exit point (not the final one)
            if block_idx in self.exit_positions[:-1]:
                logits, emb = self.exit_manager.get_exit(block_idx, x)
                exit_logits.append(logits)
                exit_embeddings.append(emb)

        # Final exit
        final_emb = self.final_pooling(x)
        final_logits = self.classifier(final_emb)
        exit_logits.append(final_logits)

        # Gating logits for budget loss (computed from final embedding)
        gate_logits = self.gate(final_emb)

        result = {
            'logits': exit_logits,
            'gate_logits': gate_logits,
            'embedding': final_emb,
        }

        # Artist prediction if head exists
        if self.artist_head is not None:
            result['artist_logits'] = self.artist_head(final_emb)

        return result

    def inference(self, x, confidence_threshold=0.8):
        """
        Inference with early exit based on confidence.

        Returns:
            dict with 'logits', 'exit_index', 'confidence'
        """
        x = self.stem(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            block_idx = i + 1

            if block_idx in self.exit_positions[:-1]:
                logits, _ = self.exit_manager.get_exit(block_idx, x)
                confidence = F.softmax(logits, dim=-1).max(dim=-1).values
                if confidence.mean() >= confidence_threshold:
                    return {
                        'logits': logits,
                        'exit_index': self.exit_positions.index(block_idx),
                        'confidence': confidence,
                    }

        # Final exit (always taken if earlier exits not confident)
        final_emb = self.final_pooling(x)
        final_logits = self.classifier(final_emb)
        confidence = F.softmax(final_logits, dim=-1).max(dim=-1).values
        return {
            'logits': final_logits,
            'exit_index': len(self.exit_positions) - 1,
            'confidence': confidence,
        }
