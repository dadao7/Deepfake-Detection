"""S9: 저자 baseline 상단을 캐시(CLIP CLS + b_3)에서 재구성해 학습 가능하게 한 모듈 (PROTOCOL §15).
logit = output(cat[α_v·vision_proj(CLS), adapter 상수, deepfake_proj(avgpool(f8(f7(b_3))))]) — 원본 M2F2Det.forward와 같은 순서·연산.
가중치는 current_model_180.pth에서 읽는다(읽기 전용). 풀린 모듈은 항상 eval 모드(BN 통계 고정, stochastic depth 끔)."""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

PREFIX = {'deepfake_encoder.7.': 'f7.', 'deepfake_encoder.8.': 'f8.', 'deepfake_proj.': 'deepfake_proj.',
          'vision_proj.': 'vision_proj.', 'output.': 'output.', 'clip_vision_alpha': 'clip_vision_alpha'}


class BaselineTop(nn.Module):
    def __init__(self):
        super().__init__()
        f = torchvision.models.efficientnet_b4().features
        self.f7, self.f8 = f[7], f[8]
        self.deepfake_proj = nn.Sequential(nn.Linear(1792, 1792), nn.LayerNorm(1792))
        self.vision_proj = nn.Sequential(nn.Linear(1024, 1792), nn.LayerNorm(1792))
        self.clip_vision_alpha = nn.Parameter(torch.tensor(0.5))
        self.output = nn.Linear(2 * 1792 + 128, 2)
        self.register_buffer('adapter', torch.from_numpy(np.load(os.path.join(C.BASE_DIR, 'adapter_const.npy'))).float())
        super().train(False)

    def load_author(self):
        C._patch_torch_load()
        ck = torch.load(os.path.join(C.WEIGHTS, 'current_model_180.pth'), map_location='cpu')['model_state_dict']
        sd = {}
        for k, v in ck.items():
            k = C.remap(k)
            for a, b in PREFIX.items():
                if k.startswith(a):
                    sd[b + k[len(a):]] = v
        r = self.load_state_dict(sd, strict=False)
        missing = [k for k in r.missing_keys if k != 'adapter']
        assert not missing and not r.unexpected_keys, (missing, r.unexpected_keys)
        del ck
        return f'baseline 상단 {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M 로드'

    def train(self, mode=True):
        return super().train(False)          # 항상 eval: BN 고정, stochastic depth 끔

    def forward(self, cls, b3):
        x = self.f8(self.f7(b3))
        df = self.deepfake_proj(F.adaptive_avg_pool2d(x, 1).flatten(1))
        v = self.clip_vision_alpha * self.vision_proj(cls.float())
        ad = self.adapter.unsqueeze(0).expand(b3.shape[0], -1)
        return self.output(torch.cat([v, ad, df], dim=-1))
