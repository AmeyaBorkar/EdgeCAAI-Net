"""
Baseline A: Tiny CNN — depthwise-separable CNN on log-mel spectrograms.
Approximate parameters: 0.5-1M.
"""

import torch.nn as nn


class TinyCNN(nn.Module):
    """
    Lightweight CNN baseline using depthwise-separable convolutions.

    Args:
        num_classes: Number of genre classes.
        n_mels: Number of mel bins.
    """

    def __init__(self, num_classes=8, n_mels=64):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: standard conv for initial feature extraction
            nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Block 2
            self._dsconv_block(64, 128, stride=2),
            # Block 3
            self._dsconv_block(128, 256, stride=2),
            # Block 4
            self._dsconv_block(256, 256, stride=2),
            # Block 5
            self._dsconv_block(256, 512, stride=2),
            # Block 6
            self._dsconv_block(512, 512, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    @staticmethod
    def _dsconv_block(in_ch, out_ch, stride=1):
        return nn.Sequential(
            # Depthwise
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        # x: (B, T, F) -> (B, 1, T, F)
        x = x.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
