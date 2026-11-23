"""① 재압축 Celeb-DF 통계 (2026-10-21). 입력: cache/qa_cdf/scores.npz (code/qa_cdf_eval.py). 기본 miniconda Python.
(1) 조건별 각 모델 − P0의 AUC 차이 CI, (2) SBI 학습 모델의 손해가 압축으로 줄었는지: [모델−P0]crf40 − [모델−P0]orig 의 CI.
영상 짝 bootstrap 2,000회(같은 영상 표본을 모든 조건·모델에 공통 적용). 출력: reports/qa_cdf.json
"""
import json
import os

import numpy as np

EXP = r'C:\m2f2\exp'
z = np.load(os.path.join(EXP, 'cache', 'qa_cdf', 'scores.npz'), allow_pickle=True)
names = list(z['names'])
y = np.load(os.path.join(EXP, 'cache', 'scorebank.npz'), allow_pickle=True)['label|cdf'].astype(int)
V = ('orig', 'crf23', 'crf40')
G = {g: [i for i, n in enumerate(names) if n.split('|')[0] == g] for g in ('P0', 'P3', 'U0s', 'U3s', 'final')}


def auc_fast(yy, s):
    r = np.empty(len(s))
    r[np.argsort(s, kind='mergesort')] = np.arange(1, len(s) + 1)
    n1 = yy.sum()
    return (r[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(yy) - n1))


def table(i):
    return {v: {g: np.mean([auc_fast(y[i], z[v][j][i]) for j in J]) * 100 for g, J in G.items()} for v in V}


pt = table(np.arange(len(y)))
rng = np.random.default_rng(0)
bs = []
for _ in range(2000):
    i = rng.integers(0, len(y), len(y))
    if y[i].min() != y[i].max():
        bs.append(table(i))
out = {'auc': pt, 'delta_vs_P0': {}, 'gap_change_orig_to_crf40': {}}
for v in V:
    for g in ('P3', 'U0s', 'U3s', 'final'):
        d = np.array([b[v][g] - b[v]['P0'] for b in bs])
        lo, hi = np.quantile(d, [0.025, 0.975])
        out['delta_vs_P0'][f'{v}|{g}'] = [pt[v][g] - pt[v]['P0'], float(lo), float(hi)]
for g in ('P3', 'U0s', 'U3s', 'final'):
    d = np.array([(b['crf40'][g] - b['crf40']['P0']) - (b['orig'][g] - b['orig']['P0']) for b in bs])
    lo, hi = np.quantile(d, [0.025, 0.975])
    p = (pt['crf40'][g] - pt['crf40']['P0']) - (pt['orig'][g] - pt['orig']['P0'])
    out['gap_change_orig_to_crf40'][g] = [p, float(lo), float(hi)]
for k, (p, lo, hi) in out['delta_vs_P0'].items():
    print(f'{k:12s} Δ vs P0 {p:+.2f} [{lo:+.2f}, {hi:+.2f}]{" ✔" if lo > 0 or hi < 0 else ""}')
for g, (p, lo, hi) in out['gap_change_orig_to_crf40'].items():
    print(f'격차 변화(원본→c40) {g:6s} {p:+.2f} [{lo:+.2f}, {hi:+.2f}]{" ✔" if lo > 0 or hi < 0 else ""}')
json.dump(out, open(os.path.join(EXP, 'reports', 'qa_cdf.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
