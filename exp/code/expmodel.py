"""옵션형 실험 모델 ExpModel (PROTOCOL §4~5). baseline은 캐시 출력(b_3, logit)을 입력으로 받는다.
원본 모듈(HeatmapHead, DetailCropBranch, nms_heatmap)은 C:\\m2f2\\code에서 import만 하고 수정하지 않는다."""
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

if C.ORIG_CODE not in sys.path:
    sys.path.append(C.ORIG_CODE)          # 뒤에 붙여 우리 모듈 이름을 가리지 않게
from detail_branch import DetailCropBranch  # noqa: E402  (원본, 읽기 전용)
from heatmap_head import HeatmapHead  # noqa: E402
from topk_crop import nms_heatmap  # noqa: E402

HM_SIZE = 11
IMG_SIZE = 224
SBI1_CKPT = os.path.join(C.WEIGHTS, 'pretrained_heatmap_detail.pth')       # 1학기 (S2 시범 전용, 읽기 전용)
SBI_CKPT = os.path.join(C.RUNS, 'sbi_pretrain', 'sbi_best.pt')             # 새 SBI 사전학습 (S3)

# PROTOCOL §5 설정 표. map: heatmap|feat, agg: single|mean|weighted, alpha: 0.5|'dynamic'
SETTINGS = {
    'P1':  dict(map='feat',    k=1, agg='single',   alpha=0.5,       m4=False, detach=True,  init='imagenet'),
    'P1p': dict(map='feat',    k=1, agg='single',   alpha=0.5,       m4=False, detach=True,  init='sbi'),
    'P2a': dict(map='heatmap', k=1, agg='single',   alpha=0.5,       m4=False, detach=True,  init='sbi'),
    'P2b': dict(map='feat',    k=3, agg='mean',     alpha=0.5,       m4=False, detach=True,  init='sbi'),
    'P2':  dict(map='heatmap', k=3, agg='weighted', alpha=0.5,       m4=False, detach=True,  init='sbi'),
    'P3a': dict(map='heatmap', k=3, agg='weighted', alpha=0.5,       m4=True,  detach=True,  init='sbi'),
    'P3b': dict(map='heatmap', k=3, agg='weighted', alpha='dynamic', m4=False, detach=True,  init='sbi'),
    'P3':  dict(map='heatmap', k=3, agg='weighted', alpha='dynamic', m4=True,  detach=True,  init='sbi'),
    'P4a': dict(map='heatmap', k=3, agg='weighted', alpha='dynamic', m4=True,  detach=False, init='sbi'),
}
COMMON = dict(crop=64, min_distance=2)
MODULE_SEED = {'heatmap_head': 11, 'detail_branch': 22, 'gating_network': 33}


def setting_cfg(name, **over):
    cfg = {**COMMON, **SETTINGS[name], **over}
    cfg['setting'] = name
    return cfg


# ── crop (원본 crop_by_heatmap_topk의 좌표 규칙을 벡터화) ─────────────
def peaks_to_boxes(peaks_yx, img_hw=(IMG_SIZE, IMG_SIZE), hm_hw=(HM_SIZE, HM_SIZE), crop=64):
    """peaks_yx (B,k,2) long → 상자 왼쪽 위 (y1, x1) (B,k) long. 원본과 같은 float32 계산·버림·경계 처리"""
    img_H, img_W = img_hw
    scale_y = img_H / hm_hw[0]
    scale_x = img_W / hm_hw[1]
    cy = (peaks_yx[:, :, 0].float() * scale_y).long()
    cx = (peaks_yx[:, :, 1].float() * scale_x).long()
    half = crop // 2
    y1 = (cy - half).clamp(min=0)
    y1 = torch.where(y1 + crop > img_H, torch.full_like(y1, max(0, img_H - crop)), y1)
    x1 = (cx - half).clamp(min=0)
    x1 = torch.where(x1 + crop > img_W, torch.full_like(x1, max(0, img_W - crop)), x1)
    return y1, x1


def crop_boxes(images, y1, x1, crop=64):
    """images (B,3,H,W), y1/x1 (B,k) → (B,k,3,crop,crop). 상자가 항상 crop×crop이라 원본의 같은 크기
    bilinear 리사이즈는 항등 → 그대로 잘라냄 (게이트 ②에서 원본과 같은지 검증)"""
    B, Cc, H, W = images.shape
    k = y1.shape[1]
    ar = torch.arange(crop, device=images.device)
    rows = (y1[:, :, None] + ar)[:, :, :, None]           # (B,k,crop,1)
    cols = (x1[:, :, None] + ar)[:, :, None, :]           # (B,k,1,crop)
    bi = torch.arange(B, device=images.device)[:, None, None, None]
    hw = images.permute(0, 2, 3, 1)                       # (B,H,W,3)
    out = hw[bi, rows, cols]                              # (B,k,crop,crop,3)
    return out.permute(0, 1, 4, 2, 3).contiguous()


# ── 증강 ──────────────────────────────────────────────────────────────
def color_jitter(x, bf, cf, sf):
    """torchvision adjust_brightness→contrast→saturation과 같은 수식, 샘플마다 다른 계수.
    x (B,3,H,W) float32 [0,1], bf/cf/sf: numpy float64 (B,) — (1−f)를 float64로 계산해 원본 스칼라 연산과 맞춤"""
    dev = x.device

    def t(a):
        return torch.as_tensor(np.asarray(a, np.float64), device=dev).float().view(-1, 1, 1, 1)

    def gray(z):
        return (0.2989 * z[:, 0] + 0.587 * z[:, 1] + 0.114 * z[:, 2]).unsqueeze(1)
    x = (t(bf) * x + t(1.0 - np.asarray(bf)) * torch.zeros_like(x)).clamp(0, 1)
    m = gray(x).mean(dim=(-3, -2, -1), keepdim=True)
    x = (t(cf) * x + t(1.0 - np.asarray(cf)) * m).clamp(0, 1)
    x = (t(sf) * x + t(1.0 - np.asarray(sf)) * gray(x)).clamp(0, 1)
    return x


def method4(crops):
    """원본 improved_model.py Method 4와 같은 논리·같은 random 호출 순서 (배치 전체에 한 번씩 결정)"""
    shp = crops.shape
    flat = crops.view(shp[0] * shp[1], *shp[2:]) if crops.dim() == 5 else crops
    if random.random() < 0.5:
        ks = random.choice([3, 5])
        sigma = random.uniform(0.1, 2.0)
        flat = F_t.gaussian_blur(flat, kernel_size=[ks, ks], sigma=[sigma, sigma])
    if random.random() < 0.5:
        noise_std = random.uniform(0.01, 0.04)
        flat = torch.clamp(flat + torch.randn_like(flat) * noise_std, 0.0, 1.0)
    if random.random() < 0.3:
        flat = F.dropout(flat, p=0.1, training=True)
    return flat.view(shp) if crops.dim() == 5 else flat


# ── 모델 ──────────────────────────────────────────────────────────────
class ExpModel(nn.Module):
    def __init__(self, cfg, seed=0):
        super().__init__()
        self.cfg = dict(cfg)
        self.k = cfg['k']
        torch.manual_seed(seed * 1000 + MODULE_SEED['heatmap_head'])
        self.heatmap_head = HeatmapHead(in_channels=272, head_conv=64) if cfg['map'] == 'heatmap' else None
        torch.manual_seed(seed * 1000 + MODULE_SEED['detail_branch'])
        self.detail_branch = DetailCropBranch()
        torch.manual_seed(seed * 1000 + MODULE_SEED['gating_network'])
        self.gating_network = nn.Sequential(nn.Linear(2 + self.k, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()) \
            if cfg['alpha'] == 'dynamic' else None

    def load_init(self, which):
        """which: imagenet(아무것도 안 함) | sbi | sbi1. 반환: 로드한 항목 설명"""
        if which == 'imagenet':
            return 'detail: ImageNet DenseNet121 + 무작위 head'
        path = SBI_CKPT if which == 'sbi' else SBI1_CKPT
        saved = torch.load(path, map_location='cpu', weights_only=False)
        msg = []
        if self.heatmap_head is not None and 'heatmap_head' in saved:
            r = self.heatmap_head.load_state_dict(saved['heatmap_head'], strict=True)
            msg.append(f'heatmap_head({r})')
        if 'detail_branch' in saved:
            r = self.detail_branch.load_state_dict(saved['detail_branch'], strict=True)
            msg.append(f'detail_branch({r})')
        return f'{which}: ' + ', '.join(msg)

    def location_map(self, b3):
        """→ (hm_logit or None, hm_prob (B,1,11,11))"""
        if self.cfg['map'] == 'heatmap':
            hm_logit = self.heatmap_head(b3)
            src = hm_logit.detach() if self.cfg['detach'] else hm_logit
            return hm_logit, torch.sigmoid(src)
        return None, torch.sigmoid(b3.float().mean(1, keepdim=True))

    def forward(self, images, b3, base_logit):
        """images (B,3,224,224) float [0,1] (학습 때는 이미 반전·색 변형된 것), b3 (B,272,11,11), base_logit (B,2) fp32"""
        hm_logit, hm_prob = self.location_map(b3)
        peaks_yx, peak_vals = nms_heatmap(hm_prob, k=self.k, min_distance=self.cfg['min_distance'])
        y1, x1 = peaks_to_boxes(peaks_yx, crop=self.cfg['crop'])
        crops = crop_boxes(images, y1, x1, crop=self.cfg['crop'])
        if self.training and self.cfg['m4']:
            crops = method4(crops)
        B, k = crops.shape[:2]
        k_preds = self.detail_branch(crops.view(B * k, *crops.shape[2:])).view(B, k, 2)
        agg = self.cfg['agg']
        if agg == 'single':
            detail = k_preds[:, 0]
        elif agg == 'mean':
            detail = k_preds.mean(dim=1)
        else:                                   # 원본 aggregate_topk_preds 'weighted'와 같은 식
            weights = peak_vals.unsqueeze(-1)
            weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
            detail = (k_preds * weights / weight_sum).sum(dim=1)
        # 원본에서 baseline logit은 autocast면 fp16(캐시는 그 값을 fp32로 저장) → α·logit 곱이 fp16에서 반올림되는 것까지 맞춤.
        # (detail은 autocast의 sum이 fp32라 fp32)
        bl = base_logit.half() if torch.is_autocast_enabled('cuda') else base_logit
        if self.gating_network is not None:
            alpha = self.gating_network(torch.cat([bl, peak_vals.to(bl.dtype)], dim=1))
        else:
            alpha = self.cfg['alpha']
        fused = alpha * bl + (1.0 - alpha) * detail
        a_out = alpha if torch.is_tensor(alpha) else torch.full_like(bl[:, :1], alpha)
        return {'pred': fused, 'detail_pred': detail, 'heatmap': hm_logit, 'alpha': a_out,
                'peaks': peaks_yx, 'peak_vals': peak_vals, 'boxes': torch.stack([y1, x1], -1)}

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
