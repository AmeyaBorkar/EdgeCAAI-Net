"""
Baseline D: Tiny Transformer — same family as EdgeCAAI-Net but without
early exit or artist invariance. Isolates the architectural contribution.
Approximate parameters: 1-2M.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyTransformer(nn.Module):
    """
    Minimal transformer baseline for spectrogram classification.

    Args:
        num_classes: Number of genre classes.
        n_mels: Number of mel bins.
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        n_blocks: Number of transformer blocks.
        ffn_dim: FFN intermediate dimension.
        dropout: Dropout rate.
    """

    def __init__(self, num_classes=8, n_mels=64, d_model=128, n_heads=4,
                 n_blocks=4, ffn_dim=256, dropout=0.15):
        super().__init__()
        # Linear projection of mel bins to d_model
        self.input_proj = nn.Linear(n_mels, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)
        self.norm_in = nn.LayerNorm(d_model)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)

        # Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F)
        x = self.input_proj(x)                     # (B, T, d_model)
        T = x.size(1)
        x = x + self.pos_encoding[:, :T, :]
        x = self.norm_in(x)
        x = self.encoder(x)                        # (B, T, d_model)
        x = x.transpose(1, 2)                      # (B, d_model, T)
        x = self.pool(x).squeeze(-1)               # (B, d_model)
        return self.classifier(x)
