"""
Baseline B: Small CRNN — CNN stem + BiGRU for temporal modeling.
Approximate parameters: 1-2M.
"""

import torch
import torch.nn as nn


class SmallCRNN(nn.Module):
    """
    Small CNN-RNN baseline with BiGRU temporal modeling.

    Args:
        num_classes: Number of genre classes.
        n_mels: Number of mel bins.
        gru_hidden: GRU hidden dimension.
    """

    def __init__(self, num_classes=8, n_mels=64, gru_hidden=128):
        super().__init__()
        # CNN stem
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # pool frequency only
        )

        # Frequency dimension after 3 pools: n_mels // 8
        rnn_input = 128 * (n_mels // 8)

        # BiGRU
        self.gru = nn.GRU(
            rnn_input, gru_hidden,
            batch_first=True, bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(gru_hidden * 2, num_classes),
        )

    def forward(self, x):
        # x: (B, T, F) -> (B, 1, T, F)
        x = x.unsqueeze(1)
        x = self.cnn(x)  # (B, C, T', F')
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)  # (B, T', C*F')
        x, _ = self.gru(x)       # (B, T', 2*hidden)
        x = x.mean(dim=1)        # (B, 2*hidden) - temporal average
        return self.classifier(x)
