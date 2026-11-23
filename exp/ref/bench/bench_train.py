"""학습 속도 측정 (코드 수정 없음): 현재 모델 / detach 제거 / seg head 추가 시 1스텝 시간과 GPU 메모리"""
import sys, time, types
sys.path.insert(0, r'C:\m2f2\code')
import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import autocast, GradScaler

import config as C
import train_improved as T
import topk_crop
from heatmap_head import HeatmapHead

B = C.TRAIN_BATCH
model = T.load_model('dynamic')
for p in model.baseline.parameters():
    p.requires_grad = False
seg_head = HeatmapHead(in_channels=272, head_conv=64).to(C.device)

orig_crop = topk_crop.crop_by_heatmap_topk


def crop_no_detach(images, heatmap_head, b3, k=3, crop_size=224, min_distance=2):
    """detach 만 뺀 같은 함수 (측정용 사본)"""
    Bn, _, H, W = images.shape
    hm_logit = heatmap_head(b3)
    hm_prob = torch.sigmoid(hm_logit)
    peaks, peak_vals = topk_crop.nms_heatmap(hm_prob, k=k, min_distance=min_distance)
    sy, sx = H / hm_logit.shape[2], W / hm_logit.shape[3]
    half, crops = crop_size // 2, []
    for ki in range(k):
        row = []
        for b in range(Bn):
            cy, cx = int(peaks[b, ki, 0].item() * sy), int(peaks[b, ki, 1].item() * sx)
            y1 = max(0, cy - half); y2 = y1 + crop_size
            if y2 > H: y2 = H; y1 = max(0, y2 - crop_size)
            x1 = max(0, cx - half); x2 = x1 + crop_size
            if x2 > W: x2 = W; x1 = max(0, x2 - crop_size)
            row.append(F.interpolate(images[b:b + 1, :, y1:y2, x1:x2], size=(crop_size, crop_size), mode='bilinear', align_corners=False)[0])
        crops.append(torch.stack(row))
    return torch.stack(crops, 1), hm_logit, peak_vals


def bench(name, use_seg, n=25):
    params = [p for p in model.parameters() if p.requires_grad] + (list(seg_head.parameters()) if use_seg else [])
    opt = torch.optim.AdamW(params, lr=C.LR)
    scaler = GradScaler('cuda')
    crit = nn.CrossEntropyLoss()
    model.train(); model.baseline.eval()
    x = torch.rand(B, 3, 224, 224, device=C.device)
    y = torch.randint(0, 2, (B,), device=C.device)
    mask = (torch.rand(B, 1, 11, 11, device=C.device) > 0.8).float()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(n + 5):
        torch.cuda.synchronize(); t = time.time()
        opt.zero_grad()
        with autocast('cuda'):
            out = model(x, return_dict=True)
            loss = crit(out['pred'].float(), y) + C.LOSS_W_DETAIL * crit(out['detail_pred'].float(), y)
            if use_seg:
                b3 = model.baseline.module.block_outputs['b_3'].detach().float()
                loss = loss + F.binary_cross_entropy_with_logits(seg_head(b3).float(), mask)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        if i >= 5:
            times.append(time.time() - t)
    hm_grad = sum(float(p.grad.abs().sum()) for p in model.heatmap_head.parameters() if p.grad is not None)
    print(f'{name:28s} 1스텝 {sum(times)/len(times)*1000:6.0f} ms (배치 {B}) | 최대 GPU 메모리 {torch.cuda.max_memory_allocated()/2**30:4.1f} GB | heatmap_head gradient 합 {hm_grad:.4g}')


def bench_eval(n=25):
    model.eval()
    x = torch.rand(32, 3, 224, 224, device=C.device)
    times = []
    with torch.no_grad(), autocast('cuda'):
        for i in range(n + 5):
            torch.cuda.synchronize(); t = time.time()
            model(x, return_dict=True)
            torch.cuda.synchronize()
            if i >= 5:
                times.append(time.time() - t)
    print(f'{"평가 (배치 32)":28s} 1스텝 {sum(times)/len(times)*1000:6.0f} ms')


import improved_model
bench('현재 모델 (detach 있음)', False)
improved_model.crop_by_heatmap_topk = crop_no_detach
bench('detach 제거', False)
bench('detach 제거 + seg head', True)
improved_model.crop_by_heatmap_topk = orig_crop
bench_eval()
print('GPU:', torch.cuda.get_device_name(0))
