"""
Baseline C: MobileNetV3-Small adapted for spectrogram input.
Approximate parameters: ~1.5M.
"""

import torch.nn as nn
import torchvision.models as models


class MobileNetBaseline(nn.Module):
    """
    MobileNetV3-Small treating spectrogram as a single-channel image.

    Args:
        num_classes: Number of genre classes.
        pretrained: Whether to load ImageNet pretrained weights.
    """

    def __init__(self, num_classes=8, pretrained=False):
        super().__init__()
        self.mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        )
        # Replace first conv to accept 1-channel input
        old_conv = self.mobilenet.features[0][0]
        self.mobilenet.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # Replace classifier
        in_features = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: (B, T, F) -> (B, 1, T, F)
        x = x.unsqueeze(1)
        return self.mobilenet(x)
