"""기준값 0.5(보정 없음)에서 최종 모델 − 논문 모델(P0)의 ACC·bACC 차이 신뢰구간 (2026-10-26). 학습 없음, 기본 miniconda Python.
최종 모델은 seed 3개의 지표 평균, 영상 짝 bootstrap 2,000회(qa_calib_ci.py와 같은 방식, 보정만 하지 않음). 출력: reports/qa_bacc_raw_ci.json
"""
import json
import os

import numpy as np

EXP = r'C:\m2f2\exp'
REPORT = ['fftest', 'cdf', 'fsh', 'dfdtest']
b4 = np.load(os.path.join(EXP, 'cache', 'scorebank.npz'), allow_pickle=True)
b5 = np.load(os.path.join(EXP, 'cache', 'scorebank_t5.npz'), allow_pickle=True)
LAB = {s: b4[f'label|{s}'].astype(int) for s in REPORT}
MODELS = {
    'P0': lambda s: [b4[f'score|{s}|P0']],
    '최종': lambda s: [b5[f'score|{s}|M1|tv|P3best+U3s|a0.50b0.75|{k}'] for k in range(3)],
}


def acc_bacc(y, preds):
    a = np.mean([(p == y).mean() for p in preds])
    b = np.mean([(p[y == 1].mean() + (~p[y == 0]).mean()) / 2 for p in preds])
    return np.array([a, b]) * 100


P = {n: {s: [np.asarray(x, np.float64) >= 0.5 for x in g(s)] for s in REPORT} for n, g in MODELS.items()}
rng = np.random.default_rng(0)
res = {}
for s in REPORT:
    y = LAB[s]
    base, fin = acc_bacc(y, P['P0'][s]), acc_bacc(y, P['최종'][s])
    pt = fin - base
    ds = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        ds.append(acc_bacc(y[i], [p[i] for p in P['최종'][s]]) - acc_bacc(y[i], [p[i] for p in P['P0'][s]]))
    lo, hi = np.quantile(np.array(ds), [0.025, 0.975], axis=0)
    res[s] = {'P0': [float(base[0]), float(base[1])], '최종': [float(fin[0]), float(fin[1])],
              'acc': [float(pt[0]), float(lo[0]), float(hi[0])], 'bacc': [float(pt[1]), float(lo[1]), float(hi[1])]}
    sig = lambda j: ' ✔' if lo[j] > 0 or hi[j] < 0 else ''
    print(f'{s:8s} ACC {base[0]:.1f}→{fin[0]:.1f} {pt[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]{sig(0)}   '
          f'bACC {base[1]:.1f}→{fin[1]:.1f} {pt[1]:+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}]{sig(1)}')
json.dump(res, open(os.path.join(EXP, 'reports', 'qa_bacc_raw_ci.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
