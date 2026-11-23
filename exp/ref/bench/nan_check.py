"""detach 제거 시 heatmap_head gradient NaN 원인 확인 (코드 수정 없음)
- 혼합정밀도(fp16) + GradScaler 에서 스텝마다 gradient 가 유한한지, scale 이 어떻게 변하는지
- fp32 로 돌리면 사라지는지"""
import sys
sys.path.insert(0, r'C:\m2f2\code')
sys.path.insert(0, r'C:\Users\COM\AppData\Local\Temp\claude\c--m2f2\0e1dfcd2-bc72-4433-a053-953ab69c855d\scratchpad')
import h5py, numpy as np, torch, torch.nn as nn
from torch.amp import autocast, GradScaler

import config as C
import train_improved as T
import improved_model
import bench_train_lite as BL  # detach 없는 crop 함수 사본

model = T.load_model('dynamic')
for p in model.baseline.parameters():
    p.requires_grad = False
improved_model.crop_by_heatmap_topk = BL.crop_no_detach

# 실제 FF++ 프레임 16장 (진짜 8 + Deepfakes 8)
frames, labels = [], []
for fname, lab in (('FF++_original_c40.h5', 0), ('FF++_Deepfakes_c40.h5', 1)):
    with h5py.File(rf'C:\m2f2\data\FF++\{fname}', 'r') as f:
        for k in sorted(f.keys())[:8]:
            frames.append(f[k][0]); labels.append(lab)
x = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.).permute(0, 3, 1, 2).to(C.device)
y = torch.tensor(labels, device=C.device)
crit = nn.CrossEntropyLoss()


def run(name, amp, n=12):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=C.LR)
    scaler = GradScaler('cuda', enabled=amp)
    model.train(); model.baseline.eval()
    res = []
    for i in range(n):
        opt.zero_grad()
        with autocast('cuda', enabled=amp):
            out = model(x, return_dict=True)
            loss = crit(out['pred'].float(), y) + C.LOSS_W_DETAIL * crit(out['detail_pred'].float(), y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        g = [p.grad for p in model.heatmap_head.parameters() if p.grad is not None]
        finite = all(torch.isfinite(t).all() for t in g)
        norm = float(torch.sqrt(sum((t.float() ** 2).sum() for t in g))) if finite else float('nan')
        hm = out['heatmap'].float()
        res.append(f'{i}: loss {float(loss):.3f} | hm logit 범위 [{float(hm.min()):.1f}, {float(hm.max()):.1f}] | grad 유한 {finite} norm {norm:.3g} | scale {scaler.get_scale() if amp else 1:.0f}')
        scaler.step(opt); scaler.update()
    print(f'== {name}'); print('\n'.join(res))


run('혼합정밀도(fp16) — 현재 학습 설정', True)
model = T.load_model('dynamic')
for p in model.baseline.parameters():
    p.requires_grad = False
run('fp32', False)
