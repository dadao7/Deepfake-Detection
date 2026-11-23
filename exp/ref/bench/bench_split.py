"""학습 1스텝 중 '고정된 기본 모델 계산'이 차지하는 시간 측정 (캐시 방식의 이득 추정, 코드 수정 없음)"""
import sys, time
sys.path.insert(0, r'C:\m2f2\code')
import torch, torch.nn as nn
from torch.amp import autocast, GradScaler
import config as C
import train_improved as T
from topk_crop import nms_heatmap, aggregate_topk_preds

model = T.load_model('dynamic')
for p in model.baseline.parameters():
    p.requires_grad = False
model.train(); model.baseline.eval()
B = 16
x = torch.rand(B, 3, 224, 224, device=C.device)
y = torch.randint(0, 2, (B,), device=C.device)
crit = nn.CrossEntropyLoss()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=C.LR); scaler = GradScaler('cuda')


def t(fn, n=20):
    for _ in range(3): fn()
    torch.cuda.synchronize(); s = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time() - s) / n * 1000


def base_only():
    with torch.no_grad(), autocast('cuda'):
        out = model.baseline(x, return_dict=True)
        _ = model.baseline.module.block_outputs['b_3']


with torch.no_grad(), autocast('cuda'):
    out = model.baseline(x, return_dict=True)
    cached_logit = out['pred'].float().clone()
    cached_b3 = model.baseline.module.block_outputs['b_3'].half().clone()


def heads_only():
    """기본 모델 출력은 저장된 값을 쓰고, 나머지(heatmap·crop·DenseNet·결합)만 학습"""
    opt.zero_grad()
    with autocast('cuda'):
        hm = model.heatmap_head(cached_b3.float())
        peaks, pv = nms_heatmap(torch.sigmoid(hm.detach()), k=3, min_distance=2)
        cy = (peaks[:, :, 0].float() * (224 / 11)).long().clamp(32, 224 - 32)
        cx = (peaks[:, :, 1].float() * (224 / 11)).long().clamp(32, 224 - 32)
        idx = torch.arange(64, device=x.device) - 32
        ys = (cy[..., None] + idx).clamp(0, 223); xs = (cx[..., None] + idx).clamp(0, 223)
        bi = torch.arange(B, device=x.device)[:, None, None, None]
        crops = x.permute(0, 2, 3, 1)[bi, ys[:, :, :, None], xs[:, :, None, :]].permute(0, 1, 4, 2, 3)
        det = aggregate_topk_preds(model.detail_branch, crops, pv)
        a = model.gating_network(torch.cat([cached_logit, pv], 1))
        fused = a * cached_logit + (1 - a) * det
        loss = crit(fused.float(), y) + 0.15 * crit(det.float(), y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()


def full_step():
    opt.zero_grad()
    with autocast('cuda'):
        out = model(x, return_dict=True)
        loss = crit(out['pred'].float(), y) + 0.15 * crit(out['detail_pred'].float(), y)
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update()


print(f'현재 방식 1스텝          {t(full_step):6.0f} ms')
print(f'  그중 기본 모델 계산      {t(base_only):6.0f} ms')
print(f'캐시 방식 1스텝(추정)     {t(heads_only):6.0f} ms  (crop 추출도 반복문 대신 한 번에)')
print(f'b_3 1장 크기 {cached_b3[0].numel() * 2 / 1024:.1f} KB')
