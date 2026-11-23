"""S6-3: 위치 평가 (PROTOCOL §11). FF++ 실제 마스크(정렬 통과 프레임)에서 heatmap 위치 정확도.
대상: seg head(P4b·P4 공통) / 분류용 heatmap(P2a·P2·P3a·P3b·P3·P4a, seed별 best·last) / 얼굴 중앙 기준선 / (참고) 1학기 SBI head
지표(11×11 셀, GT = 셀 조작 비율 ≥ 0.5): 셀 AUROC, 경계 셀(0.1~0.9) AUROC, IoU(τ는 모델마다 val 평균 IoU 최대), 최댓값 적중, 경계 적중(GT11∈[0.1,0.9]),
      crop 적중(top-k 64 crop 중 조작 픽셀 ≥ 50%가 하나라도) k=1·3, crop 평균 조작 비율. numpy 행렬곱(BLAS) 안 씀.
출력: reports/s6_loc.csv (모델·체크포인트·seed·방식별), reports/s6_loc_summary.json
  python code/loc_eval.py"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import expmodel as E  # noqa: E402
from topk_crop import nms_heatmap  # noqa: E402

MD = os.path.join(C.CACHE, 'masks')
HM_SETTINGS = ['P2a', 'P2', 'P3a', 'P3b', 'P3', 'P4a']
S8_SETTINGS = ['P2', 'P3', 'P4a']
ANCHOR_SETTINGS = ['P4c_l1', 'P4c_l10', 'P4c_l100']   # TASKS4 T4-6 heatmap anchor
TAUS = np.concatenate([np.linspace(0.001, 0.05, 50), np.linspace(0.06, 0.99, 94)])


def load_split(sp):
    z = dict(np.load(os.path.join(MD, f'{sp}_loc.npz')))
    z['gt224'] = np.unpackbits(z['gt224'], axis=-1)[..., :224].astype(bool)
    if sp == 'val':
        z['b3'] = np.load(os.path.join(MD, 'val_b3.npy'), mmap_mode='r')
        z['rows'] = np.arange(len(z['meth']))
    else:
        z['b3'] = np.load(os.path.join(C.BASE_DIR, 'test_b3.npy'), mmap_mode='r')
        z['rows'] = z['cache_row']
    return z


@torch.no_grad()
def heatmaps(head, z):
    """→ σ(heatmap) (N,11,11) float32, eval 모드, autocast fp16 (분류 평가와 같은 정밀도)"""
    head.eval()
    out = []
    for s in range(0, len(z['rows']), 256):
        r = z['rows'][s:s + 256]
        b3 = torch.from_numpy(np.ascontiguousarray(z['b3'][r])).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            hm = head(b3)
        out.append(torch.sigmoid(hm.float())[:, 0].cpu().numpy())
    return np.concatenate(out)


def center_map(n):
    yy, xx = np.mgrid[0:11, 0:11]
    g = np.exp(-((yy - 5) ** 2 + (xx - 5) ** 2) / (2 * 2.0 ** 2)).astype(np.float32)
    return np.repeat(g[None], n, 0)


def cell_auc(p, g):
    """프레임별 셀 AUROC(Mann-Whitney), 양·음 셀이 모두 있는 프레임만. 반환 (N,) nan 포함"""
    out = np.full(len(p), np.nan)
    pf, gf = p.reshape(len(p), -1), g.reshape(len(g), -1)
    for j in range(len(p)):
        pos, neg = pf[j][gf[j]], pf[j][~gf[j]]
        if len(pos) and len(neg):
            d = pos[:, None] - neg[None, :]
            out[j] = ((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size
    return out


def iou_at(p, g, tau):
    pr = p >= tau
    inter = (pr & g).sum((1, 2))
    union = (pr | g).sum((1, 2))
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def crop_stats(p, gt224, k):
    pt = torch.from_numpy(p[:, None]).float()
    peaks, _ = nms_heatmap(pt, k=k, min_distance=2)
    y1, x1 = E.peaks_to_boxes(peaks, crop=64)
    gm = torch.from_numpy(gt224[:, None].astype(np.float32))
    cov = E.crop_boxes(gm, y1, x1).mean((2, 3, 4)).numpy()         # (N,k) 조작 픽셀 비율
    top1 = peaks[:, 0].numpy()
    return cov, top1


def frame_metrics(p, z, tau, k_list=(1, 3)):
    g11 = z['gt11'].astype(np.float32)
    g = g11 >= 0.5
    gb = (g11 >= 0.1) & (g11 <= 0.9)                     # 경계 셀(조작 영역 가장자리) — FF++ 마스크는 얼굴 대부분을 덮어 '안쪽' 지표가 중앙 기준선에 포화됨
    m = {'cell_auc': cell_auc(p, g), 'bnd_auc': cell_auc(p, gb), 'iou': iou_at(p, g, tau)}
    for k in k_list:
        cov, top1 = crop_stats(p, z['gt224'], k)
        m[f'crop_hit_k{k}'] = (cov >= 0.5).any(1).astype(float)
        m[f'crop_cov_k{k}'] = cov.mean(1)
        if k == 1:
            g11 = z['gt11'].astype(np.float32)[np.arange(len(p)), top1[:, 0], top1[:, 1]]
            m['max_hit'] = (g11 >= 0.5).astype(float)
            m['boundary_hit'] = ((g11 >= 0.1) & (g11 <= 0.9)).astype(float)
    return m


def summarize(m, meth):
    out = {}
    for name in ['all'] + list(C.METHODS):
        sel = np.ones(len(meth), bool) if name == 'all' else (meth == name)
        out[name] = {k: float(np.nanmean(v[sel])) * (1 if k == 'iou' or k.startswith('crop_cov') else 100)
                     for k, v in m.items()}
    return out


def eval_model(name, ckpt, seed, tag, p_val, p_test, val, test, rows, k_list=(1, 3)):
    g_val = val['gt11'].astype(np.float32) >= 0.5
    ious = [iou_at(p_val, g_val, t).mean() for t in TAUS]
    tau = float(TAUS[int(np.argmax(ious))])
    s = summarize(frame_metrics(p_test, test, tau, k_list), test['meth'])
    for meth, d in s.items():
        rows.append({'model': name, 'ckpt': ckpt, 'seed': seed, 'tag': tag, 'method': meth, 'tau': tau,
                     'val_iou_at_tau': float(max(ious)), **d})
    print(f'{name:10s} {ckpt:5s} s{seed} τ={tau:.3f} | AUROC {s["all"]["cell_auc"]:.1f} bndAUROC {s["all"]["bnd_auc"]:.1f} IoU {s["all"]["iou"]:.3f} '
          f'max {s["all"]["max_hit"]:.1f} bnd {s["all"]["boundary_hit"]:.1f} crop1 {s["all"]["crop_hit_k1"]:.1f} '
          f'crop3 {s["all"].get("crop_hit_k3", float("nan")):.1f}', flush=True)


def main():
    val, test = load_split('val'), load_split('test')
    rows = []
    # 얼굴 중앙 기준선
    eval_model('center', '-', '-', 'baseline', center_map(len(val['rows'])), center_map(len(test['rows'])),
               val, test, rows, k_list=(1,))
    # seg head (P4b·P4 공통) + 1학기 SBI head(참고)
    for name, path in (('seg_head', E.SBI_CKPT.replace('sbi_best.pt', 'seg_head.pt')), ('sbi1_head', E.SBI1_CKPT)):
        if not os.path.exists(path):
            print('없음:', path)
            continue
        head = E.HeatmapHead(272, 64).cuda()
        head.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['heatmap_head'])
        eval_model(name, '-', '-', 'fixed', heatmaps(head, val), heatmaps(head, test), val, test, rows)
    # 분류용 heatmap (S4: FF++ 학습) + S8: SBI 학습 (PROTOCOL §14)
    jobs = [(st, f'{st}_d100_s{{}}') for st in HM_SETTINGS] + [(f'{st}_sbi', f'{st}_sbi_s{{}}') for st in S8_SETTINGS] + \
        [(st, f'{st}_s{{}}') for st in ANCHOR_SETTINGS]
    for st, pat in jobs:
        for seed in (0, 1, 2):
            rdir = os.path.join(C.RUNS, pat.format(seed))
            for ck in ('best', 'last'):
                p = os.path.join(rdir, f'ckpt_{ck}.pt')
                if not os.path.exists(os.path.join(rdir, 'DONE')) or not os.path.exists(p):
                    continue
                sd = torch.load(p, map_location='cpu', weights_only=False)['model']
                head = E.HeatmapHead(272, 64).cuda()
                head.load_state_dict({k[len('heatmap_head.'):]: v for k, v in sd.items() if k.startswith('heatmap_head.')})
                eval_model(st, ck, seed, 'cls', heatmaps(head, val), heatmaps(head, test), val, test, rows)
    os.makedirs(C.REPORTS, exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(C.REPORTS, 's6_loc.csv'), 'w', encoding='utf-8') as f:
        f.write(','.join(keys) + '\n')
        for r in rows:
            f.write(','.join(f'{r[k]:.4f}' if isinstance(r.get(k), float) else str(r.get(k, '')) for k in keys) + '\n')
    # 요약: 모델·체크포인트별 seed 평균±표준편차 (전체)
    summ = {}
    for r in rows:
        if r['method'] != 'all':
            continue
        key = f"{r['model']}|{r['ckpt']}"
        summ.setdefault(key, []).append(r)
    out = {}
    for key, rs in summ.items():
        out[key] = {m: [float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs])), len(rs)]
                    for m in ('cell_auc', 'bnd_auc', 'iou', 'max_hit', 'boundary_hit', 'crop_hit_k1', 'crop_hit_k3',
                              'crop_cov_k1', 'crop_cov_k3') if m in rs[0]}
    json.dump({'n_val_frames': int(len(val['rows'])), 'n_test_frames': int(len(test['rows'])), 'summary': out},
              open(os.path.join(C.REPORTS, 's6_loc_summary.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('저장 reports/s6_loc.csv, s6_loc_summary.json')


if __name__ == '__main__':
    main()
