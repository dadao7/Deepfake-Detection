"""⑭ 단일 프레임 성능 통계 (2026-10-21). 입력: cache/qa_frame_probs.npz (code/qa_frame.py). 기본 miniconda Python.
프레임 하나를 독립 입력으로 보고 AUC·오탐 5% 탐지율·ACC/bACC(0.5)를 계산한다.
신뢰구간은 같은 영상의 프레임이 서로 닮았으므로 '영상 단위 군집 bootstrap'(영상을 뽑고 그 프레임 전부 사용) 1,000회.
출력: reports/qa_frame.json
"""
import json
import os

import numpy as np
from sklearn.metrics import roc_curve

EXP = r'C:\m2f2\exp'
z = np.load(os.path.join(EXP, 'cache', 'qa_frame_probs.npz'), allow_pickle=True)
names = list(z['names'])
GROUP = {'P0': [names.index('P0')], 'P3': [names.index(f'P3|{k}') for k in range(3)],
         '최종': [names.index(f'final|{k}') for k in range(3)]}
SETS = ['fftest', 'cdf', 'fsh', 'dfdtest']


def auc_fast(y, s):
    r = np.empty(len(s))
    r[np.argsort(s, kind='mergesort')] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def metrics(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    q = p >= 0.5
    return [auc_fast(y, p) * 100, tpr[fpr <= 0.05].max() * 100, (q == y).mean() * 100,
            (q[y == 1].mean() + (~q[y == 0]).mean()) / 2 * 100]


out = {}
rng = np.random.default_rng(0)
for s in SETS:
    P, y, vid = z[f'{s}|p'].astype(np.float64), z[f'{s}|y'].astype(int), z[f'{s}|vid']
    row = {g: dict(zip(['auc', 'tpr@fpr5', 'acc05', 'bacc05'], np.mean([metrics(y, P[j]) for j in J], 0).round(3).tolist()))
           for g, J in GROUP.items()}
    uv = np.unique(vid)
    frames_of = {v: np.where(vid == v)[0] for v in uv}
    ds = {'P0': [], 'P3': []}
    for _ in range(1000):
        idx = np.concatenate([frames_of[v] for v in rng.choice(uv, len(uv))])
        yi = y[idx]
        if yi.min() == yi.max():
            continue
        fin = np.mean([auc_fast(yi, P[j][idx]) for j in GROUP['최종']])
        for b in ('P0', 'P3'):
            ds[b].append((fin - np.mean([auc_fast(yi, P[j][idx]) for j in GROUP[b]])) * 100)
    for b in ('P0', 'P3'):
        d = np.array(ds[b])
        pt = row['최종']['auc'] - row[b]['auc']
        lo, hi = np.quantile(d, [0.025, 0.975])
        row[f'Δauc_vs_{b}'] = [round(float(pt), 3), round(float(lo), 3), round(float(hi), 3)]
    row['n_frames'], row['n_videos'] = int(len(y)), int(len(uv))
    out[s] = row
    print(s, json.dumps(row, ensure_ascii=False), flush=True)
json.dump(out, open(os.path.join(EXP, 'reports', 'qa_frame.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
