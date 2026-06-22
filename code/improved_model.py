"""
improved_model.py
M2F2Det + HeatmapHead + NMS TopK Crop + DenseNet121 Detail Branch
[★ 완전체 최종 수정본: 하이브리드 제어(Dynamic/Static) + Method 4 탑재]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_t
import random

from heatmap_head   import HeatmapHead
from detail_branch  import DetailCropBranch
from topk_crop      import crop_by_heatmap_topk, aggregate_topk_preds


class M2F2DetImproved(nn.Module):
    """
    Args:
        baseline_model : nn.DataParallel(M2F2Det) 상태
        k              : crop 개수 (default 3)
        crop_size      : 각 crop 크기 (default 64)
        alpha          : 'dynamic' 입력 시 동적 게이팅(Method 3) 활성화 / 숫자 입력 시 고정 가중치 모드
    """

    def __init__(
        self,
        baseline_model,
        k: int = 3,
        crop_size: int = 64,
        alpha = 'dynamic',  # [핵심] 기본값을 'dynamic'으로 설정하여 연구 명확성 확보
        aggregate_mode: str = 'weighted',
        min_distance: int = 2,
    ):
        super().__init__()
        self.baseline       = baseline_model
        self.k              = k
        self.crop_size      = crop_size
        self.alpha          = alpha  # ◀ AttributeError 방지를 위해 확실하게 멤버 변수 등록
        self.aggregate_mode = aggregate_mode
        self.min_distance   = min_distance

        # EfficientNet-B4 고정 채널 수
        b3_ch = 272
        print(f'[M2F2DetImproved] b_3 ch={b3_ch}, 활성화된 가중치 모드(alpha)={self.alpha}')

        self.heatmap_head  = HeatmapHead(in_channels=b3_ch, head_conv=64)
        self.detail_branch = DetailCropBranch()

        # 동적 게이팅 네트워크 초기화 (상시 대기)
        self.gating_network = nn.Sequential(
            nn.Linear(2 + self.k, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, images: torch.Tensor, return_dict: bool = False, **kwargs):
        device = images.device

        # ── 1. Baseline (전역 맥락 추출) ─────────────────────────────────
        out            = self.baseline(images, return_dict=True, **kwargs)
        baseline_logit = out['pred']
        b3             = self.baseline.module.block_outputs['b_3'].to(device)

        # ── 2. Top-K Crop (위조 의심 국소 패치 및 신뢰도 추출) ────────────────
        all_crops, hm_logit, peak_vals = crop_by_heatmap_topk(
            images, self.heatmap_head, b3,
            k=self.k,
            crop_size=self.crop_size,
            min_distance=self.min_distance,
        )

        # ── 3. Method 4: Targeted Local Patch Regularization (학습용 패치 교란) ──
        if self.training:
            orig_shape = all_crops.shape
            if len(orig_shape) == 5:
                B, K, C_im, H, W = orig_shape
                all_crops_flat = all_crops.view(B * K, C_im, H, W)
            else:
                all_crops_flat = all_crops

            if random.random() < 0.5:
                kernel_size = random.choice([3, 5])
                sigma = random.uniform(0.1, 2.0)
                all_crops_flat = F_t.gaussian_blur(all_crops_flat, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])

            if random.random() < 0.5:
                noise_std = random.uniform(0.01, 0.04)
                noise = torch.randn_like(all_crops_flat) * noise_std
                all_crops_flat = torch.clamp(all_crops_flat + noise, 0.0, 1.0)

            if random.random() < 0.3:
                all_crops_flat = F.dropout(all_crops_flat, p=0.1, training=True)

            if len(orig_shape) == 5:
                all_crops = all_crops_flat.view(B, K, C_im, H, W)
            else:
                all_crops = all_crops_flat

        # ── 4. Detail Branch (국소 패치 분류 및 가중 합산) ─────────────────
        detail_logit = aggregate_topk_preds(
            self.detail_branch, all_crops, peak_vals,
            mode=self.aggregate_mode,
        )

        # ── 5. Method 3: 하이브리드 가중치 분기 로직 ────────────────────────
        # alpha가 문자열 'dynamic'일 때만 동적 게이팅 수행, 숫자일 때는 지정된 static 가중치 적용!
        if isinstance(self.alpha, str) and self.alpha == 'dynamic':
            gating_input  = torch.cat([baseline_logit, peak_vals], dim=1)  # (B, 2 + K)
            alpha_to_use  = self.gating_network(gating_input)             # (B, 1)
        else:
            alpha_to_use  = self.alpha  # 입력받은 고정 숫자값 사용 (예: 0.5, 0.85)

        fused = alpha_to_use * baseline_logit + (1.0 - alpha_to_use) * detail_logit

        if return_dict:
            return {
                'pred':          fused,
                'baseline_pred': baseline_logit,
                'detail_pred':   detail_logit,
                'heatmap':       hm_logit,
                'dynamic_alpha': alpha_to_use if isinstance(self.alpha, str) else torch.full_like(baseline_logit[:, :1], self.alpha),
            }
        return fused

    def all_parameters(self):
        return list(self.parameters())