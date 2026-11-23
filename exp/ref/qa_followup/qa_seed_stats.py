"""⑩ seed 확대 통계 (2026-10-21). 기본 miniconda Python. 학습 없음(점수만 읽음).
seed 0~2: 기존 은행(scorebank·scorebank_t5), seed 3~4: cache/qa_seedext_scores.npz (code/qa_seed_ext.py).
병합 비율 a0.50b0.75는 seed 0~2로 미리 고른 값 그대로 → seed 3~4는 사전 고정 규칙의 재현 검증.
출력: reports/qa_seed.json, 화면 요약
"""
import json
import os

import numpy as np

EXP = r'C:\m2f2\exp'
REPORT = ['fftest', 'cdf', 'fsh', 'dfdtest']
b4 = np.load(os.path.join(EXP, 'cache', 'scorebank.npz'), allow_pickle=True)
b5 = np.load(os.path.join(EXP, 'cache', 'scorebank_t5.npz'), allow_pickle=True)
bx = np.load(os.path.join(EXP, 'cache', 'qa_seedext_scores.npz'), allow_pickle=True)
LAB = {s: b4[f'label|{s}'].astype(int) for s in REPORT}
NEW = (3, 4)


def get(model, s, seed):
    if seed in NEW:
        return bx[f'score|{s}|{ {"최종": "merge", "P3": "P3last", "U3s": "U3sbest"}[model] }|{seed}']
    return {'최종': lambda: b5[f'score|{s}|M1|tv|P3best+U3s|a0.50b0.75|{seed}'],
            'P3': lambda: b4[f'score|{s}|S4|P3|{seed}|last'],
            'U3s': lambda: b4[f'score|{s}|S9|U3s|{seed}|best']}[model]()


def auc_fast(y, s):
    s = np.asarray(s, np.float64)
    r = np.empty(len(s))
    r[np.argsort(s, kind='mergesort')] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


out = {'per_seed': {}, 'tests': []}
P0 = {s: np.asarray(b4[f'score|{s}|P0'], np.float64) for s in REPORT}
for s in REPORT:
    y = LAB[s]
    row = {'P0': auc_fast(y, P0[s]) * 100}
    for m in ('최종', 'P3', 'U3s'):
        row[m] = [auc_fast(y, get(m, s, k)) * 100 for k in range(5)]
    out['per_seed'][s] = row
    d = np.array(row['최종']) - row['P0']
    print(f"{s:8s} P0 {row['P0']:.2f} | 최종 seed별 " + ' '.join(f'{v:.2f}' for v in row['최종'])
          + f" | 평균 {np.mean(row['최종']):.2f} (seed0~2 {np.mean(row['최종'][:3]):.2f}, seed3~4 {np.mean(row['최종'][3:]):.2f})"
          + f" | P0 대비 +인 seed {int((d > 0).sum())}/5, Δ {d.mean():+.2f} ± {d.std(ddof=1):.2f}")

rng = np.random.default_rng(0)
for base in ('P0', 'P3'):
    for s in REPORT:
        y = LAB[s]
        new = [np.asarray(get('최종', s, k), np.float64) for k in range(5)]
        bas = [P0[s]] if base == 'P0' else [np.asarray(get('P3', s, k), np.float64) for k in range(5)]
        stat = lambda i: np.mean([auc_fast(y[i], v[i]) for v in new]) - np.mean([auc_fast(y[i], v[i]) for v in bas])
        pt = stat(np.arange(len(y))) * 100
        ds = []
        for _ in range(2000):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                ds.append(stat(i) * 100)
        ds = np.array(ds)
        p = max(2 * min((ds <= 0).mean(), (ds >= 0).mean()), 1 / len(ds))
        out['tests'].append({'base': base, 'set': s, 'delta': float(pt), 'lo': float(np.quantile(ds, 0.025)),
                             'hi': float(np.quantile(ds, 0.975)), 'p': float(p)})
for fam, idx in (('8개(P0·P3 대비)', range(8)), ('4개(P0 대비)', range(4))):
    T = [out['tests'][i] for i in idx]
    order = np.argsort([t['p'] for t in T])
    run = 0.0
    for rank, j in enumerate(order):
        run = max(run, min(1.0, (len(T) - rank) * T[j]['p']))
        T[j][f'p_holm_{fam}'] = float(run)
for t in out['tests']:
    print(f"최종−{t['base']} {t['set']:8s} {t['delta']:+.2f} [{t['lo']:+.2f}, {t['hi']:+.2f}] p {t['p']:.4f} "
          + ' '.join(f'{k[7:]} {v:.3f}' for k, v in t.items() if k.startswith('p_holm')))
json.dump(out, open(os.path.join(EXP, 'reports', 'qa_seed.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
