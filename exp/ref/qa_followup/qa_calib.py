"""⑨-2 확률 보정(Platt scaling): 선택 세트(FF++ val + DFD-val)에서 logit(영상 점수)에 1차 로지스틱을 맞춰 보고 세트에 적용.
학습된 모델은 그대로, 점수 → 확률 변환만 바꾼다. 보고 세트 라벨은 쓰지 않는다. 2026-10-21. 기본 miniconda Python으로 실행.
출력: reports/qa_calib.json (요약은 qa_followup.md에 추가)
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


def ece(y, s, nb=10):
    e = 0.0
    for lo in np.linspace(0, 1, nb, endpoint=False):
        m = (s >= lo) & (s < lo + 1 / nb) if lo < 1 - 1 / nb else (s >= lo)
        if m.any():
            e += m.mean() * abs(s[m].mean() - y[m].mean())
    return e * 100


def acc(y, s):
    p = s >= 0.5
    return (p == y).mean() * 100, (p[y == 1].mean() + (~p[y == 0]).mean()) / 2 * 100


out = {}
for name, get in MODELS.items():
    nseed = len(get('ffval'))
    res = {s: [] for s in REPORT}
    for k in range(nseed):
        xs = np.concatenate([lg(get(s)[k]) for s in SELECT])
        ys = np.concatenate([LAB[s] for s in SELECT])
        w = np.concatenate([np.full(len(LAB[s]), 1 / len(LAB[s])) for s in SELECT])   # 두 선택 세트를 같은 무게로
        cal = LogisticRegression(C=1e6).fit(xs.reshape(-1, 1), ys, sample_weight=w)
        for s in REPORT:
            y = LAB[s]
            raw = np.asarray(get(s)[k], np.float64)
            new = cal.predict_proba(lg(raw).reshape(-1, 1))[:, 1]
            res[s].append([ece(y, raw), ece(y, new), *acc(y, raw), *acc(y, new)])
    out[name] = {s: dict(zip(['ece_raw', 'ece_cal', 'acc_raw', 'bacc_raw', 'acc_cal', 'bacc_cal'],
                             [float(v) for v in np.mean(res[s], 0)])) for s in REPORT}
json.dump(out, open(os.path.join(EXP, 'reports', 'qa_calib.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for name in out:
    for s in REPORT:
        d = out[name][s]
        print(f"{name:3s} {s:8s} ECE {d['ece_raw']:5.1f}→{d['ece_cal']:5.1f}  ACC {d['acc_raw']:5.1f}→{d['acc_cal']:5.1f}  bACC {d['bacc_raw']:5.1f}→{d['bacc_cal']:5.1f}")
