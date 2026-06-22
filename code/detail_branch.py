# coding: utf-8
"""
detail_branch.py
DenseNet121 기반 Detail Crop Branch.
K개 crop 각각을 처리하는 shared 분류기.
"""

import torch
import torch.nn as nn
from torchvision import models


class DetailCropBranch(nn.Module):
    """
    DenseNet121 기반 2-class 분류기.

    입력: (B, 3, 224, 224) — 단일 crop 또는 K개 crop을 배치로 묶은 것
    출력: (B, 2)            — real/fake logit
    """

    def __init__(self):
        super().__init__()
        base         = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.encoder = base.features
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.head    = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            logit: (B, 2)
        """
        feat = self.pool(self.encoder(x))   # (B, 1024, 1, 1)
        feat = feat.view(x.size(0), -1)     # (B, 1024)
        return self.head(feat)              # (B, 2)
