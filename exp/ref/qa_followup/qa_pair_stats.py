"""원본 vs 열화(SNS 모사) / c40 vs c23 짝 비교 통계 (2026-10-21). 기본 miniconda Python.
  python qa_pair_stats.py sns  → cache/qa_sns/scores.npz  → reports/qa_sns.json
  python qa_pair_stats.py c23  → cache/qa_c23/scores.npz  → reports/qa_c23.json
조건별 모델 AUC(seed 평균), 최종 − P0 차이의 영상 짝 bootstrap 95% CI(2,000회), 조건 간 '향상 폭 변화'의 CI.
"""
import json
import os
import sys

import numpy as np

EXP = r'C:\m2f2\exp'


def auc_fast(y, s):
    s = np.asarray(s, np.float64)
    r = np.empty(len(s))
    r[np.argsort(s, kind='mergesort')] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def groups(names):
    return {g: [i for i, n in enumerate(names) if n.split('|')[0] == g] for g in ('P0', 'P3', 'final')}


def analyse(y, conds, G):
    """conds: {조건: (M, V) 점수} 같은 영상 순서 → 표·CI"""
    def tab(i):
        return {c: {g: np.mean([auc_fast(y[i], S[j][i]) for j in J]) * 100 for g, J in G.items()} for c, S in conds.items()}
    pt = tab(np.arange(len(y)))
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            bs.append(tab(i))
    out = {'auc': pt, 'final_minus_P0': {}, 'P3_minus_P0': {}}
    for c in conds:
        for key, g in (('final_minus_P0', 'final'), ('P3_minus_P0', 'P3')):
            d = np.array([b[c][g] - b[c]['P0'] for b in bs])
            out[key][c] = [pt[c][g] - pt[c]['P0'], *map(float, np.quantile(d, [0.025, 0.975]))]
    c0, c1 = list(conds)
    d = np.array([(b[c1]['final'] - b[c1]['P0']) - (b[c0]['final'] - b[c0]['P0']) for b in bs])
    out['gain_change'] = [(pt[c1]['final'] - pt[c1]['P0']) - (pt[c0]['final'] - pt[c0]['P0']), *map(float, np.quantile(d, [0.025, 0.975]))]
    return out


mode = sys.argv[1]
res = {}
if mode == 'sns':
    z = np.load(os.path.join(EXP, 'cache', 'qa_sns', 'scores.npz'), allow_pickle=True)
    G = groups(list(z['names']))
    for s in ('fftest', 'cdf', 'fsh', 'dfdtest'):
        y = z[f'{s}|label'].astype(int)
        res[s] = analyse(y, {'orig': z[f'{s}|orig'], 'sns': z[f'{s}|sns']}, G)
else:
    z = np.load(os.path.join(EXP, 'cache', 'qa_c23', 'scores.npz'), allow_pickle=True)
    G = groups(list(z['names']))
    k23, k40 = list(z['c23|key']), list(z['c40|key'])
    common = sorted(set(k23) & set(k40))
    i23 = [k23.index(k) for k in common]
    i40 = [k40.index(k) for k in common]
    y = z['c40|label'][i40].astype(int)
    assert (z['c23|label'][i23] == y).all()
    res['ffpp'] = analyse(y, {'c40': z['c40|score'][:, i40], 'c23': z['c23|score'][:, i23]}, G)
    res['ffpp']['n_videos'] = len(common)
for s, r in res.items():
    print(f'== {s}')
    for c, d in r['auc'].items():
        fm = r['final_minus_P0'][c]
        print(f"  {c:5s} P0 {d['P0']:.2f} P3 {d['P3']:.2f} 최종 {d['final']:.2f} | 최종−P0 {fm[0]:+.2f} [{fm[1]:+.2f}, {fm[2]:+.2f}]"
              f"{' ✔' if fm[1] > 0 or fm[2] < 0 else ''}")
    gc = r['gain_change']
    print(f"  향상 폭 변화 {gc[0]:+.2f} [{gc[1]:+.2f}, {gc[2]:+.2f}]{' ✔' if gc[1] > 0 or gc[2] < 0 else ''}")
json.dump(res, open(os.path.join(EXP, 'reports', f'qa_{mode}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
