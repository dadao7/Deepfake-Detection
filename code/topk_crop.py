# coding: utf-8
"""
topk_crop.py
NMS 기반 Top-K 피크 추출 + Multi-Crop + Weighted Aggregation.
(기존 topk_heatmap_crop.py → 이름만 변경, 내용 동일)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def nms_heatmap(hm_prob: torch.Tensor, k: int, min_distance: int = 5) -> tuple:
    """
    Heatmap에서 NMS로 well-separated K개 피크 추출.

    Args:
        hm_prob:      (B, 1, H, W) sigmoid 적용된 heatmap
        k:            추출할 피크 수
        min_distance: 피크 간 최소 픽셀 거리 (heatmap 공간 기준)
    Returns:
        peaks_yx:  (B, k, 2) 피크 (y, x) 좌표
        peak_vals: (B, k)    각 피크의 heatmap 값 (가중치용)
    """
    B, _, H, W = hm_prob.shape
    kernel = max(3, 2 * min_distance + 1)
    if kernel % 2 == 0:
        kernel += 1

    hm_max        = F.max_pool2d(hm_prob, kernel_size=kernel, stride=1, padding=kernel//2)
    local_max     = (hm_prob == hm_max).float()
    hm_filtered   = hm_prob * local_max

    flat          = hm_filtered.view(B, -1)
    actual_k      = min(k, flat.shape[1])
    top_vals, top_idxs = torch.topk(flat, actual_k, dim=1)

    if actual_k < k:
        pad = k - actual_k
        top_vals = torch.cat([top_vals, top_vals[:, -1:].expand(-1, pad)], dim=1)
        top_idxs = torch.cat([top_idxs, top_idxs[:, -1:].expand(-1, pad)], dim=1)

    peak_y   = top_idxs // W
    peak_x   = top_idxs %  W
    peaks_yx = torch.stack([peak_y, peak_x], dim=2)
    return peaks_yx, top_vals


def crop_by_heatmap_topk(
    images: torch.Tensor,
    heatmap_head: nn.Module,
    b3: torch.Tensor,
    k: int = 3,
    crop_size: int = 224,
    min_distance: int = 2,
) -> tuple:
    """
    b3 → HeatmapHead → NMS → K개 crop.

    Args:
        images:       (B, 3, H, W) 원본 입력 이미지
        heatmap_head: HeatmapHead 인스턴스
        b3:           (B, C, H', W') EfficientNet-B4 b_3
        k:            crop 개수
        crop_size:    각 crop 크기
        min_distance: NMS 최소 거리
    Returns:
        all_crops:  (B, k, 3, crop_size, crop_size)
        hm_logit:   (B, 1, H', W') — focal loss 계산용 (gradient 유지)
        peak_vals:  (B, k)          — aggregation 가중치
    """
    B, _, img_H, img_W = images.shape

    hm_logit  = heatmap_head(b3)
    hm_prob   = torch.sigmoid(hm_logit.detach())
    hm_up     = F.interpolate(hm_prob, size=(img_H, img_W), mode='bilinear', align_corners=False)

    hm_sp_H, hm_sp_W = hm_logit.shape[2], hm_logit.shape[3]
    scale_y   = img_H / hm_sp_H
    scale_x   = img_W / hm_sp_W

    peaks_yx_hm, peak_vals = nms_heatmap(hm_prob, k=k, min_distance=min_distance)

    peaks_img = peaks_yx_hm.float().clone()
    peaks_img[:, :, 0] = (peaks_yx_hm[:, :, 0].float() * scale_y).long().float()
    peaks_img[:, :, 1] = (peaks_yx_hm[:, :, 1].float() * scale_x).long().float()

    half   = crop_size // 2
    crops  = []
    for ki in range(k):
        ki_crops = []
        for b in range(B):
            cy = int(peaks_img[b, ki, 0].item())
            cx = int(peaks_img[b, ki, 1].item())
            y1 = max(0, cy - half); y2 = y1 + crop_size
            if y2 > img_H: y2 = img_H; y1 = max(0, y2 - crop_size)
            x1 = max(0, cx - half); x2 = x1 + crop_size
            if x2 > img_W: x2 = img_W; x1 = max(0, x2 - crop_size)
            crop = F.interpolate(
                images[b:b+1, :, y1:y2, x1:x2],
                size=(crop_size, crop_size), mode='bilinear', align_corners=False
            )
            ki_crops.append(crop[0])
        crops.append(torch.stack(ki_crops, dim=0))

    all_crops = torch.stack(crops, dim=1)
    return all_crops, hm_logit, peak_vals


def aggregate_topk_preds(
    detail_branch: nn.Module,
    all_crops: torch.Tensor,
    peak_vals: torch.Tensor,
    mode: str = 'weighted',
) -> torch.Tensor:
    """
    K개 crop → DenseNet → 가중 집계 → pred_detail.

    Args:
        detail_branch: DetailCropBranch 인스턴스
        all_crops:     (B, k, 3, H, W)
        peak_vals:     (B, k) heatmap 피크값
        mode:          'weighted' | 'mean' | 'max'
    Returns:
        pred_detail: (B, 2)
    """
    B, k, C, H, W = all_crops.shape
    flat_crops = all_crops.view(B * k, C, H, W)
    flat_preds = detail_branch(flat_crops)
    k_preds    = flat_preds.view(B, k, 2)

    if mode == 'weighted':
        weights     = peak_vals.unsqueeze(-1)
        weight_sum  = weights.sum(dim=1, keepdim=True).clamp(min=1e-6)
        pred_detail = (k_preds * weights / weight_sum).sum(dim=1)
    elif mode == 'mean':
        pred_detail = k_preds.mean(dim=1)
    elif mode == 'max':
        fake_probs  = F.softmax(k_preds, dim=-1)[:, :, 1]
        best_k      = fake_probs.argmax(dim=1)
        pred_detail = k_preds[torch.arange(B), best_k]
    else:
        raise ValueError(f"mode must be 'weighted'|'mean'|'max', got {mode}")

    return pred_detail
