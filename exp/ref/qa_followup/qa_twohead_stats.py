"""② 두 head 구조 통계 (2026-10-21). 입력: cache/qa_twohead_scores.npz. 기본 miniconda Python.
TH(두 head) best·last vs P3·P4a(같은 체크포인트 종류)·P0. seed 0~2 평균 AUC, 영상 짝 bootstrap 2,000회. 출력: reports/qa_twohead.json
"""
import json
import os

import numpy as np

EXP = r'C:\m2f2\exp'
SETS = ['fftest', 'cdf', 'fsh', 'dfdtest']
b4 = np.load(os.path.join(EXP, 'cache', 'scorebank.npz'), allow_pickle=True)
z = np.load(os.path.join(EXP, 'cache', 'qa_twohead_scores.npz'), allow_pickle=True)
names = list(z['names'])


def auc_fast(y, s):
    s = np.asarray(s, np.float64)
    r = np.empty(len(s))
    r[np.argsort(s, kind='mergesort')] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def get(model, s, ck):
    if model == 'TH':
        return [z[s][names.index(f'TH|{k}|{ck}')] for k in range(3)]
    if model == 'P0':
        return [b4[f'score|{s}|P0']]
    return [b4[f'score|{s}|S4|{model}|{k}|{ck}'] for k in range(3)]


out = {}
rng = np.random.default_rng(0)
for ck in ('best', 'last'):
    for s in SETS:
        y = b4[f'label|{s}'].astype(int)
        M = {m: [np.asarray(v, np.float64) for v in get(m, s, ck)] for m in ('TH', 'P3', 'P4a', 'P0')}
        pt = {m: np.mean([auc_fast(y, v) for v in L]) * 100 for m, L in M.items()}
        ds = {b: [] for b in ('P3', 'P4a', 'P0')}
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            th = np.mean([auc_fast(y[i], v[i]) for v in M['TH']])
            for b in ds:
                ds[b].append((th - np.mean([auc_fast(y[i], v[i]) for v in M[b]])) * 100)
        row = {'auc': pt}
        for b in ds:
            lo, hi = np.quantile(ds[b], [0.025, 0.975])
            row[f'TH_minus_{b}'] = [pt['TH'] - pt[b], float(lo), float(hi)]
        out[f'{ck}|{s}'] = row
        f = lambda b: f"{row[f'TH_minus_{b}'][0]:+.2f} [{row[f'TH_minus_{b}'][1]:+.2f}, {row[f'TH_minus_{b}'][2]:+.2f}]" + (
            ' ✔' if row[f'TH_minus_{b}'][1] > 0 or row[f'TH_minus_{b}'][2] < 0 else '')
        print(f"{ck:4s} {s:8s} TH {pt['TH']:.2f} P3 {pt['P3']:.2f} P4a {pt['P4a']:.2f} P0 {pt['P0']:.2f} | −P3 {f('P3')} | −P4a {f('P4a')} | −P0 {f('P0')}")
json.dump(out, open(os.path.join(EXP, 'reports', 'qa_twohead.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
