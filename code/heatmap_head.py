# coding: utf-8
"""
heatmap_head.py
HeatmapHead 네트워크 모듈.
출처: LAA-Net pose_efficientNet.py (head != 'cls' 분기)
"""

import torch
import torch.nn as nn


class HeatmapHead(nn.Module):
    """
    b_3 feature map → Heatmap Logit (B, 1, H', W')

    구조: Conv(in_ch→head_conv, 3×3) → BN → ReLU → Conv(head_conv→1, 1×1)
    """

    def __init__(self, in_channels: int, head_conv: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_conv, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(head_conv),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, 1, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Conv2d) and m.weight.shape[0] == 1:
                nn.init.constant_(m.bias, -2.19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H', W') — EfficientNet-B4 b_3 feature map
        Returns:
            hm_logit: (B, 1, H', W') — sigmoid 적용 전 raw logit
        """
        return self.head(x)
