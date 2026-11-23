"""⑨-3 보정 후 정확도 비교의 신뢰구간 (2026-10-21). 학습 없음, 기본 miniconda Python으로 실행.
두 모델(P0, 최종)에 같은 Platt 보정(선택 세트 FF++ val + DFD-val, 두 세트 같은 무게)을 적용한 뒤,
보고 세트에서 기준값 0.5의 ACC·bACC 차이(최종 − P0, seed 평균)를 영상 짝 bootstrap 2,000회로 구한다.
보정 함수는 선택 세트로 한 번만 맞추고 고정한다(보고 세트 라벨은 쓰지 않음). 출력: reports/qa_calib_ci.json
"""
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression

EXP = r'C:\m2f2\exp'
REPORT = ['fftest', 'cdf', 'fsh', 'dfdtest']
SELECT = ['ffval', 'dfdval']
b4 = np.load(os.path.join(EXP, 'cache', 'scorebank.npz'), allow_pickle=True)
b5 = np.load(os.path.join(EXP, 'cache', 'scorebank_t5.npz'), allow_pickle=True)
LAB = {s: b4[f'label|{s}'].astype(int) for s in REPORT + SELECT}
MODELS = {
    'P0': lambda s: [b4[f'score|{s}|P0']],
    '최종': lambda s: [b5[f'score|{s}|M1|tv|P3best+U3s|a0.50b0.75|{k}'] for k in range(3)],
}


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibrated(get):
    """seed마다 선택 세트로 Platt 보정을 맞춰 보고 세트 확률을 돌려준다 → {set: [seed별 확률]}"""
    out = {s: [] for s in REPORT}
    for k in range(len(get('ffval'))):
        xs = np.concatenate([lg(get(s)[k]) for s in SELECT])
        ys = np.concatenate([LAB[s] for s in SELECT])
        w = np.concatenate([np.full(len(LAB[s]), 1 / len(LAB[s])) for s in SELECT])
        cal = LogisticRegression(C=1e6).fit(xs.reshape(-1, 1), ys, sample_weight=w)
        for s in REPORT:
            out[s].append(cal.predict_proba(lg(get(s)[k]).reshape(-1, 1))[:, 1] >= 0.5)
    return out


def acc_bacc(y, preds):
    a = np.mean([(p == y).mean() for p in preds])
    b = np.mean([(p[y == 1].mean() + (~p[y == 0]).mean()) / 2 for p in preds])
    return np.array([a, b]) * 100


P = {n: calibrated(g) for n, g in MODELS.items()}
rng = np.random.default_rng(0)
res = {}
for s in REPORT:
    y = LAB[s]
    pt = acc_bacc(y, P['최종'][s]) - acc_bacc(y, P['P0'][s])
    ds = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        ds.append(acc_bacc(y[i], [p[i] for p in P['최종'][s]]) - acc_bacc(y[i], [p[i] for p in P['P0'][s]]))
    ds = np.array(ds)
    lo, hi = np.quantile(ds, [0.025, 0.975], axis=0)
    res[s] = {'acc': [float(pt[0]), float(lo[0]), float(hi[0])], 'bacc': [float(pt[1]), float(lo[1]), float(hi[1])]}
    sig = lambda j: ' ✔' if lo[j] > 0 or hi[j] < 0 else ''
    print(f'{s:8s} ΔACC {pt[0]:+.2f} [{lo[0]:+.2f}, {hi[0]:+.2f}]{sig(0)}   ΔbACC {pt[1]:+.2f} [{lo[1]:+.2f}, {hi[1]:+.2f}]{sig(1)}')
json.dump(res, open(os.path.join(EXP, 'reports', 'qa_calib_ci.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
