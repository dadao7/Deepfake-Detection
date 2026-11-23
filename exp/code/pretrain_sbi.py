"""S3-2: SBI 사전학습 — heatmap head(11×11 경계 GT, 원본 focal loss) + detail 분기(GT peak 위치 crop: SBI vs 같은 위치 진짜)
한 번만 수행(seed 0)하고 모든 설정·seed가 같은 SBI 초기화를 쓴다. seg head = best epoch의 heatmap head 복사본(고정).
출력: runs/sbi_pretrain/{sbi_best.pt, sbi_last.pt, seg_head.pt, curves.csv, metrics.json, config.json}"""
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import expmodel as E  # noqa: E402
from heatmap_loss import heatmap_focal_loss  # noqa: E402  (원본, 읽기 전용)
from topk_crop import nms_heatmap  # noqa: E402

SBI_DIR = os.path.join(C.CACHE, 'sbi')
OUT = os.path.join(C.RUNS, 'sbi_pretrain')
SEED, EPOCHS, LR, WD, BS_PAIRS, CLIP, W_DETAIL = 0, 10, 1e-4, 1e-3, 16, 1.0, 1.0
K_GT, GT_POS, V_TRAIN = 3, 0.5, 4


class SBISet:
    def __init__(self, split):
        self.img = np.load(os.path.join(SBI_DIR, f'{split}_img.npy'), mmap_mode='r')
        self.b3 = np.load(os.path.join(SBI_DIR, f'{split}_b3.npy'), mmap_mode='r')
        self.gt = np.load(os.path.join(SBI_DIR, f'{split}_gt.npy'), mmap_mode='r')
        self.mask = np.load(os.path.join(SBI_DIR, f'{split}_mask.npy'), mmap_mode='r')
        self.N = self.img.shape[0]

    def batch(self, idx):
        """→ x (2B,3,224,224) [진짜0, 가짜0, 진짜1, …], b3 (2B,…), gt (2B,1,11,11)(진짜는 0), y (2B,), gt_fake (B,1,11,11)"""
        idx = np.asarray(idx)
        B = len(idx)
        x = torch.from_numpy(self.img[idx]).cuda().view(2 * B, 224, 224, 3).permute(0, 3, 1, 2).float().div_(255.)
        b3 = torch.from_numpy(self.b3[idx]).cuda().view(2 * B, 272, 11, 11)
        g = torch.from_numpy(self.gt[idx].astype(np.float32)).cuda().view(B, 1, 11, 11)
        gt = torch.zeros(2 * B, 1, 11, 11, device='cuda')
        gt[1::2] = g
        y = torch.tensor([0, 1] * B, device='cuda')
        return x, b3, gt, y, g


def gt_crops(x, g):
    """GT peak(NMS top-3, 값 ≥ 0.5) 위치에서 가짜·진짜 crop 쌍 → (crops (2n,3,64,64), labels (2n,))"""
    peaks, pv = nms_heatmap(g, k=K_GT, min_distance=2)
    y1, x1 = E.peaks_to_boxes(peaks, crop=64)
    real, fake = x[0::2], x[1::2]
    cf = E.crop_boxes(fake, y1, x1)
    cr = E.crop_boxes(real, y1, x1)
    ok = pv >= GT_POS
    cf, cr = cf[ok], cr[ok]
    crops = torch.cat([cf, cr])
    lab = torch.cat([torch.ones(len(cf), dtype=torch.long), torch.zeros(len(cr), dtype=torch.long)]).cuda()
    return crops, lab


def boundary11(mask_u8):
    m = mask_u8.astype(np.float32) / 255.
    b = 4 * m * (1 - m)
    return np.stack([cv2.resize(bb, (11, 11), interpolation=cv2.INTER_AREA) for bb in b])


@torch.no_grad()
def evaluate(head, det, S):
    """SBI val: 손실 + 위치 지표 (PROTOCOL §11과 같은 정의를 SBI 마스크에 적용)"""
    head.eval()
    det.eval()
    fl, ce, n_c, correct, n_b = 0.0, 0.0, 0, 0, 0
    score, lab, hit_gt, b_at_peak, crop_hit, center_b, center_gt = [], [], [], [], [], [], []
    for s in range(0, S.N, 32):
        idx = np.arange(s, min(S.N, s + 32))
        x, b3, gt, y, g = S.batch(idx)
        with torch.autocast('cuda', dtype=torch.float16):
            hm = head(b3)
            crops, cl = gt_crops(x, g)
            lo = det(crops)
        fl += float(heatmap_focal_loss(hm.float(), gt)) * len(idx)
        ce += float(F.cross_entropy(lo.float(), cl, reduction='sum'))
        n_c += len(cl)
        correct += int((lo.argmax(1) == cl).sum())
        n_b += len(idx)
        prob = torch.sigmoid(hm.float())
        score.append(prob.flatten(1).max(1).values.cpu())
        lab.append(y.cpu())
        pf = prob[1::2]
        peaks, _ = nms_heatmap(pf, k=3, min_distance=2)
        p1 = peaks[:, 0]
        gg = g[:, 0]
        hit_gt.append((gg[torch.arange(len(idx)), p1[:, 0], p1[:, 1]] >= GT_POS).cpu())
        b11 = torch.from_numpy(boundary11(S.mask[idx])).cuda()
        b_at_peak.append(b11[torch.arange(len(idx)), p1[:, 0], p1[:, 1]].cpu())
        center_b.append(b11[:, 5, 5].cpu())
        center_gt.append((gg[:, 5, 5] >= GT_POS).cpu())
        # crop 적중: 예측 top-3 상자 중 하나라도 경계(B>0.5) 픽셀을 포함
        y1, x1 = E.peaks_to_boxes(peaks, crop=64)
        m = torch.from_numpy(S.mask[idx].astype(np.float32) / 255.).cuda()
        bfull = (4 * m * (1 - m) > 0.5).float()[:, None]
        cb = E.crop_boxes(bfull, y1, x1)                       # (B,3,1,64,64)
        crop_hit.append((cb.flatten(2).max(2).values.max(1).values > 0).cpu())
    head.train()
    det.train()
    score, lab = torch.cat(score).numpy(), torch.cat(lab).numpy()
    return {'val_focal': fl / n_b, 'val_ce_detail': ce / n_c, 'val_total': fl / n_b + W_DETAIL * ce / n_c,
            'val_detail_acc': correct / n_c * 100, 'val_hm_auc': roc_auc_score(lab, score) * 100,
            'val_peak_hit_gt': float(torch.cat(hit_gt).float().mean()) * 100,
            'val_boundary_at_peak': float(torch.cat(b_at_peak).mean()),
            'val_crop_hit': float(torch.cat(crop_hit).float().mean()) * 100,
            'center_boundary': float(torch.cat(center_b).mean()),
            'center_hit_gt': float(torch.cat(center_gt).float().mean()) * 100}


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    TR, VA = SBISet('train'), SBISet('val')
    n_frames = TR.N // V_TRAIN
    torch.manual_seed(SEED * 1000 + E.MODULE_SEED['heatmap_head'])
    head = E.HeatmapHead(272, 64).cuda()
    torch.manual_seed(SEED * 1000 + E.MODULE_SEED['detail_branch'])
    det = E.DetailCropBranch().cuda()
    params = list(head.parameters()) + list(det.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda')
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    torch.manual_seed(SEED)
    random.seed(SEED)
    cfg = {'seed': SEED, 'epochs': EPOCHS, 'lr': LR, 'wd': WD, 'bs_pairs': BS_PAIRS, 'clip': CLIP,
           'w_detail': W_DETAIL, 'k_gt': K_GT, 'gt_pos': GT_POS, 'v_train': V_TRAIN, 'n_train_pairs': TR.N,
           'n_val_pairs': VA.N, 'code_sha': C.code_sha(), 'versions': C.env_versions(),
           'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    json.dump(cfg, open(os.path.join(OUT, 'config.json'), 'w'), indent=1)
    v0 = evaluate(head, det, VA)
    curves = [{'epoch': 0, **v0}]
    print('ep 0 (초기)', {k: round(v, 4) for k, v in v0.items()}, flush=True)
    best = (float('inf'), 0)
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        head.train()
        det.train()
        order = np.random.default_rng([SEED, ep]).permutation(n_frames)
        rows = order * V_TRAIN + (ep - 1) % V_TRAIN
        acc = torch.zeros(2, device='cuda', dtype=torch.float64)
        nb = 0
        for s in range(0, len(rows), BS_PAIRS):
            idx = np.sort(rows[s:s + BS_PAIRS])
            x, b3, gt, y, g = TR.batch(idx)
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                hm = head(b3)
                crops, cl = gt_crops(x, g)
                lo = det(crops)
            lf = heatmap_focal_loss(hm.float(), gt)
            lc = F.cross_entropy(lo.float(), cl)
            loss = lf + W_DETAIL * lc
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            scaler.step(opt)
            scaler.update()
            acc += torch.stack([lf.detach(), lc.detach()]).double()
            nb += 1
        lr_now = opt.param_groups[0]['lr']
        sch.step()
        tf, tc = (acc / nb).tolist()
        if not (np.isfinite(tf) and np.isfinite(tc)):
            raise RuntimeError('비유한 손실')
        v = evaluate(head, det, VA)
        row = {'epoch': ep, 'lr': lr_now, 'train_focal': tf, 'train_ce_detail': tc, **v, 'sec': time.time() - t0}
        curves.append(row)
        state = {'heatmap_head': head.state_dict(), 'detail_branch': det.state_dict(), 'epoch': ep, 'val': v}
        if v['val_total'] < best[0]:
            best = (v['val_total'], ep)
            torch.save(state, os.path.join(OUT, 'sbi_best.pt'))
        torch.save(state, os.path.join(OUT, 'sbi_last.pt'))
        with open(os.path.join(OUT, 'curves.csv'), 'w') as f:
            keys = list(curves[-1].keys())
            f.write(','.join(keys) + '\n')
            for r in curves:
                f.write(','.join(f'{r.get(k, float("nan")):.6g}' for k in keys) + '\n')
        print(f'ep {ep} train focal {tf:.4f} ce {tc:.4f} | val total {v["val_total"]:.4f} focal {v["val_focal"]:.4f} '
              f'ce {v["val_ce_detail"]:.4f} acc {v["val_detail_acc"]:.1f} hmAUC {v["val_hm_auc"]:.1f} '
              f'peakhit {v["val_peak_hit_gt"]:.1f} crophit {v["val_crop_hit"]:.1f} | {row["sec"]:.0f}s', flush=True)
    bst = torch.load(os.path.join(OUT, 'sbi_best.pt'), map_location='cpu', weights_only=False)
    torch.save({'heatmap_head': bst['heatmap_head'], 'epoch': bst['epoch'],
                'note': 'seg head = SBI 사전학습 best epoch의 heatmap head 복사본, 가중치·BN 통계 고정(항상 eval)'},
               os.path.join(OUT, 'seg_head.pt'))
    json.dump({'best_epoch': best[1], 'best_val_total': best[0], 'best_val': bst['val'], 'init_val': v0,
               'finished': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(os.path.join(OUT, 'metrics.json'), 'w'), indent=1)
    print('끝: best epoch', best[1], flush=True)


if __name__ == '__main__':
    main()
