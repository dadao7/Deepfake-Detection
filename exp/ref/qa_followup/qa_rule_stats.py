"""⑦ 사전 고정 규칙 재검증 통계 (2026-10-21). 입력: cache/qa_rule_scores.npz (code/qa_rule_check.py). 기본 miniconda Python.
새 seed 3·4에서 세 후보(기존 병합·Xp3 병합·Xlr1 병합)의 선택 세트 점수로 규칙 A·B를 적용하고, 보고 세트 결과를 본다.
seed 0~2(TASKS6, scorebank_t5) 값도 같은 표에 둔다. 출력: reports/qa_rule.json
"""
import json
import os

import numpy as np

EXP = r'C:\m2f2\exp'
SETS = ['ffval', 'dfdval', 'fftest', 'cdf', 'fsh', 'dfdtest']
b4 = np.load(os.path.join(EXP, 'cache', 'scorebank.npz'), allow_pickle=True)
b5 = np.load(os.path.join(EXP, 'cache', 'scorebank_t5.npz'), allow_pickle=True)
bn = np.load(os.path.join(EXP, 'cache', 'qa_rule_scores.npz'), allow_pickle=True)
LAB = {s: b4[f'label|{s}'].astype(int) for s in SETS}
OLD = {'base': 'M1|tv|P3best+U3s|a0.50b0.75', 'Xp3': 'T6X|Xp3-best|tv|a0.75b0.75', 'Xlr1': 'T6X|Xlr1-best|lin|t0.8'}


def auc_fast(y, s):
    s = np.asarray(s, np.float64)
    r = np.empty(len(s))
    r[np.argsort(s, kind='mergesort')] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def get(c, s, seed):
    return bn[f'score|{s}|{c}|{seed}'] if seed in (3, 4) else b5[f'score|{s}|{OLD[c]}|{seed}']


out = {}
for tag, seeds in (('seed0-2', (0, 1, 2)), ('seed3-4', (3, 4))):
    tab = {c: {s: float(np.mean([auc_fast(LAB[s], get(c, s, k)) for k in seeds]) * 100) for s in SETS} for c in OLD}
    for c in tab:
        tab[c]['select'] = (tab[c]['ffval'] + tab[c]['dfdval']) / 2
    ruleA = max(tab, key=lambda c: tab[c]['select'])
    ok = [c for c in tab if tab[c]['ffval'] >= tab['base']['ffval']]
    ruleB = max(ok, key=lambda c: tab[c]['select'])
    out[tag] = {'table': tab, 'ruleA': ruleA, 'ruleB': ruleB}
    print(f'== {tag}: 규칙 A → {ruleA}, 규칙 B → {ruleB}')
    print('후보   선택   FFval  DFDval | FFtest  CDF    FSH    DFDtest')
    for c, d in tab.items():
        print(f"{c:5s} {d['select']:6.2f} {d['ffval']:6.2f} {d['dfdval']:6.2f} | {d['fftest']:6.2f} {d['cdf']:6.2f} {d['fsh']:6.2f} {d['dfdtest']:6.2f}")

# 새 seed에서 Xp3 병합 − 기존 병합 (보고 세트, 영상 짝 bootstrap, seed 3·4 평균)
rng = np.random.default_rng(0)
out['xp3_minus_base_seed3-4'] = {}
for s in ['fftest', 'cdf', 'fsh', 'dfdtest']:
    y = LAB[s]
    A = [np.asarray(get('Xp3', s, k), np.float64) for k in (3, 4)]
    B = [np.asarray(get('base', s, k), np.float64) for k in (3, 4)]
    st = lambda i: (np.mean([auc_fast(y[i], v[i]) for v in A]) - np.mean([auc_fast(y[i], v[i]) for v in B])) * 100
    pt = st(np.arange(len(y)))
    ds = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            ds.append(st(i))
    lo, hi = np.quantile(ds, [0.025, 0.975])
    out['xp3_minus_base_seed3-4'][s] = [float(pt), float(lo), float(hi)]
    print(f'Xp3 − 기존 병합 (seed 3·4) {s:8s} {pt:+.2f} [{lo:+.2f}, {hi:+.2f}]{" ✔" if lo > 0 or hi < 0 else ""}')
json.dump(out, open(os.path.join(EXP, 'reports', 'qa_rule.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
