"""S2 게이트 ⑤: 시범 P3 캐시 방식 vs 원본 방식(같은 seed·데이터 순서·증강 계수, 1학기 SBI 초기화) 차이 기록.
출력: reports/s2_pilot_compare.json, reports/s2_pilot_curves.csv"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

A, B = 'pilot_P3_cache_s0', 'pilot_P3_orig_s0'


def read_curves(rid):
    with open(os.path.join(C.RUNS, rid, 'curves.csv')) as f:
        keys = f.readline().strip().split(',')
        return [dict(zip(keys, map(float, ln.strip().split(',')))) for ln in f if ln.strip()]


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def main():
    ca, cb = read_curves(A), read_curves(B)
    keys = ['train_loss', 'train_loss_fused', 'train_loss_detail', 'train_auc', 'val_loss_bal', 'val_loss_raw',
            'val_loss_detail', 'val_auc_frame', 'val_auc_video', 'val_bacc05']
    with open(os.path.join(C.REPORTS, 's2_pilot_curves.csv'), 'w') as f:
        f.write('epoch,' + ','.join(f'{k}_cache,{k}_orig' for k in keys) + '\n')
        for ra, rb in zip(ca, cb):
            f.write(f'{int(ra["epoch"])},' + ','.join(f'{ra[k]:.5f},{rb[k]:.5f}' for k in keys) + '\n')
    rep = {'curve_maxabsdiff': {k: max(abs(ra[k] - rb[k]) for ra, rb in zip(ca, cb)) for k in keys},
           'curve_meanabsdiff': {k: float(np.mean([abs(ra[k] - rb[k]) for ra, rb in zip(ca, cb)])) for k in keys},
           'best_epoch': {}, 'test': {}, 'pred_diff': {}, 'heatmap_bn': {}}
    for rid in (A, B):
        m = json.load(open(os.path.join(C.RUNS, rid, 'metrics.json')))
        rep['best_epoch'][rid] = m['best']['epoch']
        rep['test'][rid] = {t: {k: round(v, 3) for k, v in m[t].items()} for t in ('best', 'last')}
    for t in ('best', 'last'):
        pa = np.load(os.path.join(C.RUNS, A, f'pred_{t}.npz'))
        pb = np.load(os.path.join(C.RUNS, B, f'pred_{t}.npz'))
        d = {}
        for ds in ('ffpp', 'cdf'):
            x, y = pa[f'{ds}_prob'].astype(np.float64), pb[f'{ds}_prob'].astype(np.float64)
            vx, vy = pa[f'{ds}_video_score'], pb[f'{ds}_video_score']
            d[ds] = {'frame_mean_absdiff': float(np.abs(x - y).mean()), 'frame_max_absdiff': float(np.abs(x - y).max()),
                     'frame_pearson': pearson(x, y), 'video_mean_absdiff': float(np.abs(vx - vy).mean()),
                     'video_pearson': pearson(vx, vy),
                     'alpha_mean_cache': float(pa[f'{ds}_alpha'].astype(np.float64).mean()),
                     'alpha_mean_orig': float(pb[f'{ds}_alpha'].astype(np.float64).mean()),
                     'top1_peak_same_frac': float((pa[f'{ds}_peaks'][:, 0] == pb[f'{ds}_peaks'][:, 0]).all(1).mean())}
        d['metric_diff_cache_minus_orig'] = {k: rep['test'][A][t][k] - rep['test'][B][t][k]
                                             for k in rep['test'][A][t] if k != 'epoch'}
        rep['pred_diff'][t] = d
    for t in ('best', 'last'):
        sa = torch.load(os.path.join(C.RUNS, A, f'ckpt_{t}.pt'), map_location='cpu', weights_only=False)['model']
        sb = torch.load(os.path.join(C.RUNS, B, f'ckpt_{t}.pt'), map_location='cpu', weights_only=False)['model']
        rv = 'heatmap_head.head.1.running_var'
        rm = 'heatmap_head.head.1.running_mean'
        rep['heatmap_bn'][t] = {'running_mean_maxabsdiff': float((sa[rm] - sb[rm]).abs().max()),
                                'running_var_rel_diff_mean': float(((sa[rv] - sb[rv]).abs() / sb[rv].abs().clamp(min=1e-6)).mean()),
                                'heatmap_weights_equal': all(torch.equal(sa[k], sb[k]) for k in sa
                                                             if k.startswith('heatmap_head') and 'running' not in k
                                                             and 'num_batches' not in k)}
    json.dump(rep, open(os.path.join(C.REPORTS, 's2_pilot_compare.json'), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
