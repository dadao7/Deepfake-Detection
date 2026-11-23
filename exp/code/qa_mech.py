"""질문 방어 ⑧·⑨ 메커니즘 확인 (2026-10-21). 학습 없음, 기존 코드는 함수만 불러 쓴다. m2f2 환경, GPU.
(A) ⑥ S3/S4 crop 불일치: SBI val에서 사전학습 detail 분기를 정답 위치 crop vs heatmap 예측 위치 crop으로 판정 → crop 정확도·영상(이미지) AUC
(B) ④ U3s의 동적 α를 SBI val에서 측정(테스트 세트 값과 비교) — seed 0~4
(C) ③ 상단으로 가는 gradient 크기: 같은 SBI 학습 배치에서 U0(분기 없음) vs U3(분기 있음), 출발점 θ0와 학습 후(best) — seed 0~2
출력: reports/qa_mech.json
"""
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import common as C  # noqa: E402
import expmodel as E  # noqa: E402
import pretrain_sbi as PS  # noqa: E402
import run_train_unfreeze as U  # noqa: E402
from topk_crop import nms_heatmap  # noqa: E402

out = {}

# ── (A) crop 불일치 ──────────────────────────────────────────────────
em = E.ExpModel(E.setting_cfg('P3'), seed=0).cuda().eval()
em.load_init('sbi')
D = PS.SBISet('val')
acc_gt, acc_pr, img_y, img_gt, img_pr = [], [], [], [], []
with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):     # b_3 캐시가 fp16 → 학습·평가와 같은 autocast
    for s in range(0, D.N, 16):
        idx = np.arange(s, min(D.N, s + 16))
        x, b3, gt, y, g = D.batch(idx)
        crops, cl = PS.gt_crops(x, g)                                   # S3 학습과 같은 정답 위치 crop
        lo = em.detail_branch(crops).float()
        acc_gt.append((lo.argmax(1) == cl).float().cpu().numpy())
        _, hm = em.location_map(b3)
        peaks, pv = nms_heatmap(hm, k=3, min_distance=2)
        y1, x1 = E.peaks_to_boxes(peaks, crop=64)
        cp = E.crop_boxes(x, y1, x1)
        Bk = cp.shape[:2]
        kp = em.detail_branch(cp.view(-1, *cp.shape[2:])).float().view(*Bk, 2)
        acc_pr.append((kp.argmax(2) == y.view(-1, 1)).float().cpu().numpy().ravel())
        w = pv.float() / pv.float().sum(1, keepdim=True).clamp(min=1e-6)
        img_pr.append(torch.softmax((kp * w.unsqueeze(-1)).sum(1), 1)[:, 1].cpu().numpy())
        img_y.append(y.cpu().numpy())
        # 정답 위치 crop의 이미지 단위 점수: 가짜의 정답 peak 위치를 진짜·가짜에 같이 적용(S3와 같음), 위치별 평균
        pk, pvg = nms_heatmap(g, k=3, min_distance=2)
        gy, gx = E.peaks_to_boxes(pk, crop=64)
        cg = E.crop_boxes(x, gy.repeat_interleave(2, 0), gx.repeat_interleave(2, 0))
        kg = em.detail_branch(cg.view(-1, *cg.shape[2:])).float().view(*cg.shape[:2], 2)
        m = (pvg >= PS.GT_POS).float().repeat_interleave(2, 0)
        img_gt.append(torch.softmax((kg * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1), 1)[:, 1].cpu().numpy())
yy = np.concatenate(img_y)
out['crop_mismatch'] = {'crop_acc_gt': float(np.concatenate(acc_gt).mean() * 100),
                        'crop_acc_pred': float(np.concatenate(acc_pr).mean() * 100),
                        'img_auc_gt': float(roc_auc_score(yy, np.concatenate(img_gt)) * 100),
                        'img_auc_pred': float(roc_auc_score(yy, np.concatenate(img_pr)) * 100)}
print('crop 불일치', out['crop_mismatch'], flush=True)
del em
torch.cuda.empty_cache()

# ── (B) SBI val에서의 α ─────────────────────────────────────────────
DV = U.SBIData('val')
alpha = {}
for seed in range(5):
    m = U.UModel('U3', seed)
    m.load_state_dict(torch.load(os.path.join(C.RUNS, f'U3s_s{seed}', 'ckpt_best.pt'), map_location='cpu', weights_only=False)['model'])
    o = U.predict([m.cuda()], DV)[0]
    lab = DV.ix['label'].astype(bool)
    alpha[seed] = {'alpha_real': float(o['alpha'][~lab].astype(np.float32).mean()), 'alpha_fake': float(o['alpha'][lab].astype(np.float32).mean()),
                   'auc_fused': float(roc_auc_score(lab, o['prob']) * 100), 'auc_detail': float(roc_auc_score(lab, o['dprob'].astype(np.float32)) * 100),
                   'auc_top': float(roc_auc_score(lab, o['base_prob']) * 100)}
    del m
    torch.cuda.empty_cache()
out['u3s_alpha_sbival'] = alpha
print('α (SBI val)', alpha, flush=True)

# ── (C) 상단 gradient 크기 ──────────────────────────────────────────
DT = U.SBIData('train')
rows = np.random.default_rng(0).choice(DT.N, 640, replace=False)


def top_grad(model, cfg):
    model.cuda().train()
    norms = []
    for s in range(0, len(rows), 16):
        torch.manual_seed(s)
        random.seed(s)
        x, b3, lg, y, cls = DT.batch(np.sort(rows[s:s + 16]))
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            o = model(x, b3, cls)
        loss = F.cross_entropy(o['pred'].float(), y)
        if cfg == 'U3':
            loss = loss + 0.15 * F.cross_entropy(o['detail_pred'].float(), y)
        loss.backward()
        sq = sum(float(p.grad.float().pow(2).sum()) for p in model.top.parameters() if p.grad is not None)
        norms.append(sq ** 0.5)
    return float(np.mean(norms))


grads = {}
for seed in range(3):
    row = {}
    for cfg in ('U0', 'U3'):
        torch.manual_seed(0)
        m0 = U.UModel(cfg, seed)
        row[f'{cfg}_init'] = top_grad(m0, cfg)
        del m0
        m1 = U.UModel(cfg, seed)
        m1.load_state_dict(torch.load(os.path.join(C.RUNS, f'{cfg}s_s{seed}', 'ckpt_best.pt'), map_location='cpu', weights_only=False)['model'])
        row[f'{cfg}_best'] = top_grad(m1, cfg)
        del m1
        torch.cuda.empty_cache()
    grads[seed] = row
    print('grad', seed, row, flush=True)
out['top_grad_norm'] = grads
json.dump(out, open(os.path.join(C.REPORTS, 'qa_mech.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('MECH_OK', flush=True)
