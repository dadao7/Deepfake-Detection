# coding: utf-8
"""
heatmap_loss.py
Focal Loss + GT Heatmap 생성 함수.
출처: LAA-Net losses.py, common.py, image_utils.py
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# GT Heatmap 생성
# ─────────────────────────────────────────────────────────────

def get_boundary_mask(sbi_mask: np.ndarray) -> np.ndarray:
    """
    SBI raw mask → boundary 강도 맵.
    B = (1-M) * M * 4  (M=0.5에서 최대값 1)

    Args:
        sbi_mask: gen_SBI()의 blending mask, 값 [0,1], shape (H,W) or (H,W,1)
    Returns:
        boundary_mask: (H, W, 1) float32
    """
    if sbi_mask.ndim == 2:
        sbi_mask = sbi_mask[..., np.newaxis]
    return ((1 - sbi_mask) * sbi_mask * 4).astype(np.float32)


def _cal_mask_wh(center, mask_2d: np.ndarray):
    j, i = center
    nz_cols = np.where(mask_2d[j, :] > 0)[0]
    nz_rows = np.where(mask_2d[:, i] > 0)[0]
    w = int(nz_cols[-1] - nz_cols[0]) if len(nz_cols) > 1 else 1
    h = int(nz_rows[-1] - nz_rows[0]) if len(nz_rows) > 1 else 1
    return w, h


def _gaussian_radius(det_size, min_overlap: float = 0.7) -> float:
    h, w = det_size
    a1 = 1; b1 = h + w; c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(max(b1**2 - 4*a1*c1, 0))) / 2
    a2 = 4; b2 = 2*(h+w); c2 = (1-min_overlap)*w*h
    r2 = (b2 + math.sqrt(max(b2**2 - 4*a2*c2, 0))) / 2
    a3 = 4*min_overlap; b3 = -2*min_overlap*(h+w); c3 = w*h*(min_overlap-1)
    r3 = (b3 + math.sqrt(max(b3**2 - 4*a3*c3, 0))) / 2
    return min(r1, r2, r3)


def generate_heatmap_gt(
    blending_mask: np.ndarray,
    heatmap_size: tuple,
    sigma_adaptive: bool = True,
    sigma: float = 3.0,
) -> np.ndarray:
    """
    SBI 블렌딩 마스크 → GT Heatmap (가우시안 피크).
    출처: LAA-Net common.py _new_encode_target()

    Args:
        blending_mask: (H,W) or (H,W,C) — boundary mask (get_boundary_mask 출력)
        heatmap_size:  (hm_H, hm_W)
        sigma_adaptive: True면 마스크 크기 기반 sigma 자동 계산
        sigma:         sigma_adaptive=False일 때 고정 sigma

    Returns:
        heatmap: (1, H, W) float32, 값 [0, 1]
    """
    mask_2d = (blending_mask[..., 0] if blending_mask.ndim == 3
               else blending_mask).astype(np.float32)
    if mask_2d.max() <= 1.0:
        mask_2d = mask_2d * 255.0

    hm_h, hm_w = heatmap_size
    target_H, target_W = mask_2d.shape
    heatmap = np.zeros((1, target_H, target_W), dtype=np.float32)

    for patch in [[0, 0], [0, 0.5], [0.5, 0], [0.5, 0.5]]:
        px1, py1 = int(target_W*patch[0]), int(target_H*patch[1])
        px2, py2 = int(target_W*(patch[0]+0.5)), int(target_H*(patch[1]+0.5))
        region   = mask_2d[py1:py2, px1:px2]
        max_val  = region.max()
        if max_val <= 0:
            continue

        pts = np.where(region == max_val)
        for j, i in zip(pts[0]+py1, pts[1]+px1):
            if sigma_adaptive:
                w_s, h_s = _cal_mask_wh((j, i), mask_2d)
                sigma_cur = _gaussian_radius((h_s, w_s)) / 3 + 1e-4
            else:
                sigma_cur = sigma

            tmp  = sigma_cur * 3
            size = tmp * 2 + 1
            if size <= 0:
                continue

            xs = np.arange(0, size, 1, np.float32)
            ys = xs[:, np.newaxis]
            x0 = y0 = size // 2
            g  = np.exp(-((xs-x0)**2 + (ys-y0)**2) / (2*sigma_cur**2))

            ul = [int(i-tmp), int(j-tmp)]
            br = [int(i+tmp+1), int(j+tmp+1)]
            g_x  = (max(0,-ul[0]), min(br[0],hm_w)-ul[0])
            g_y  = (max(0,-ul[1]), min(br[1],hm_h)-ul[1])
            img_x = (max(0,ul[0]), min(br[0],hm_w))
            img_y = (max(0,ul[1]), min(br[1],hm_h))

            if (g_x[1]>g_x[0] and g_y[1]>g_y[0]
                    and img_x[1]>img_x[0] and img_y[1]>img_y[0]):
                heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]],
                    heatmap[0][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                )

    return heatmap


# ─────────────────────────────────────────────────────────────
# Focal Loss
# ─────────────────────────────────────────────────────────────

def heatmap_focal_loss(
    pred_logit: torch.Tensor,
    gt: torch.Tensor,
    alpha: float = 0.25,
    epsilon: float = 0.35,
    noise_distribution: float = 0.2,
) -> torch.Tensor:
    """
    CornerNet modified focal loss.
    출처: LAA-Net losses.py _neg_loss()

    Args:
        pred_logit: (B, 1, H, W) — HeatmapHead 출력 (sigmoid 전)
        gt:         (B, 1, H, W) — generate_heatmap_gt() 출력, 값 [0,1]
        alpha:      전체 loss 스케일
    Returns:
        scalar loss tensor
    """
    pred = torch.clamp(pred_logit.sigmoid(), min=1e-4, max=1-1e-4)
    if gt.dim() == 3:
        gt = gt.unsqueeze(1)

    pos_inds   = gt.eq(1.0).float()
    neg_inds   = gt.lt(1.0).float()
    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = ((1-epsilon) * torch.log(pred) * torch.pow(1-pred, 2) * pos_inds
                + epsilon   * torch.log(pred) * torch.pow(1-pred, 2) * noise_distribution * pos_inds)
    neg_loss = torch.log(1-pred) * torch.pow(pred, 2) * neg_inds * neg_weights

    num_pos = pos_inds.float().sum()
    loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos if num_pos > 0 else -neg_loss.sum()
    return loss * alpha
