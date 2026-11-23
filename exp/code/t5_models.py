"""TASKS5: 비교 대상 모델의 영상 점수 모음 (TASKS4 은행 + TASKS5 은행). 각 항목 = seed별 {set: 확률 점수} 목록.
cost = 영상 1개 판정에 필요한 모델 수(추론 비용). 결합은 logit 공간에서 한다."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import common as C  # noqa: E402
import t34_common as T  # noqa: E402
import t5_common as T5  # noqa: E402

SEEDS = (0, 1, 2)
W_ENS = 0.7          # TASKS4에서 선택 세트로 고른 SBI 전문가 가중치 (P3×0.3 + U3s×0.7)


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def sig(x):
    return 1 / (1 + np.exp(-x))


def _seed(bank, fmt):
    return [{s: bank[f'score|{s}|' + fmt.format(sd=sd)] for s in T5.ALL} for sd in SEEDS]


def _avg(lst):
    return {s: sig(np.mean([logit(d[s]) for d in lst], 0)) for s in T5.ALL}


def _mix(a, b, w):
    return {s: sig((1 - w) * logit(a[s]) + w * logit(b[s])) for s in T5.ALL}


def collect(extra=True):
    """→ {이름: dict(scores=[seed별 dict], cost=모델 수, note=설명)}"""
    bank, b5 = T.load_bank(), T5.load_bank5()
    out = {}
    p3, u3s = _seed(bank, 'S4|P3|{sd}|last'), _seed(bank, 'S9|U3s|{sd}|best')
    out['P0'] = dict(scores=[{s: bank[f'score|{s}|P0'] for s in T5.ALL}], cost=1, note='저자 baseline')
    out['P3'] = dict(scores=p3, cost=1, note='현재 모델(last), seed 3개')
    out['U3s'] = dict(scores=u3s, cost=1, note='SBI 전문가(best), seed 3개')
    out['P3×3'] = dict(scores=[_avg(p3)], cost=3, note='P3 seed 3개 logit 평균')
    out['결합2'] = dict(scores=[_mix(a, b, W_ENS) for a, b in zip(p3, u3s)], cost=2, note='같은 seed P3×0.3 + U3s×0.7')
    out['결합6'] = dict(scores=[_mix(_avg(p3), _avg(u3s), W_ENS)], cost=6, note='TASKS4 최고: seed 평균 전문가 결합')
    if extra:
        ch = os.path.join(C.REPORTS, 't5_merge_choice.json')
        if os.path.exists(ch):
            ch = json.load(open(ch))
            for tag, key in (('병합', 'best_lin'), ('병합TV', 'best_tv')):
                n = ch[key][1]
                if all(f'score|fftest|M1|{n}|{sd}' in b5 for sd in SEEDS):
                    sc = _seed(b5, f'M1|{n}|{{sd}}')
                    out[tag] = dict(scores=sc, cost=1, note=f'M1 {n}, seed 3개')
                    out[tag + '×3'] = dict(scores=[_avg(sc)], cost=3, note=f'M1 {n} seed 3개 logit 평균')
        if all(f'score|fftest|MIX|U3m|{sd}|best' in bank for sd in SEEDS):
            out['혼합학습'] = dict(scores=_seed(bank, 'MIX|U3m|{sd}|best'), cost=1, note='TASKS4 혼합 학습 U3m(best), seed 3개')
        wise = sorted({k.split('|')[3:6] and '|'.join(k.split('|')[3:6]) for k in b5 if k.startswith('score|fftest|M1|wise|')})
        if wise:                                       # WiSE형 대조: 선택 세트로 b 선택
            lab = T5.labels()
            selv = {w: np.mean([np.mean([T.auc(lab[s], b5[f'score|{s}|M1|{w}|{sd}']) for s in T.SELECT_SETS]) for sd in SEEDS])
                    for w in wise if all(f'score|fftest|M1|{w}|{sd}' in b5 for sd in SEEDS)}
            w = max(selv, key=selv.get)
            out['WiSE대조'] = dict(scores=_seed(b5, f'M1|{w}|{{sd}}'), cost=1, note=f'θ0 + b·τ_U3s (FF++ 전문가 없음), 선택 {w}')
        for tag in ('lin', 'tv'):
            if f'score|fftest|M1|soup-{tag}|0' in b5:
                out[f'수프({tag})'] = dict(scores=[{s: b5[f'score|{s}|M1|soup-{tag}|0'] for s in T5.ALL}], cost=1,
                                         note=f'seed 3개 병합({tag}) 가중치 평균 = 단일 모델')
        for tn in ('P0', 'P3', '병합', '병합TV', '증류'):
            sds = [sd for sd in SEEDS if f'score|fftest|TTA|{tn}|{sd}' in b5]
            if sds and (tn == 'P0' or len(sds) == 3):
                base_cost = 1
                out[f'{tn}+TTA'] = dict(scores=_seed(b5, f'TTA|{tn}|{{sd}}') if tn != 'P0' else
                                        [{s: b5[f'score|{s}|TTA|P0|0'] for s in T5.ALL}], cost=2 * base_cost,
                                        note=f'{tn} + 좌우 반전 TTA(프레임 확률 평균)')
        for k in sorted({k.split('|')[3] for k in b5 if k.startswith('score|fftest|M3|')}):
            sds = [sd for sd in SEEDS if f'score|fftest|M3|{k}|{sd}' in b5]
            if len(sds) == 3:
                sc = _seed(b5, f'M3|{k}|{{sd}}')
                out[f'증류:{k}'] = dict(scores=sc, cost=1, note=f'M3 {k}, seed 3개')
                out[f'증류:{k}×3'] = dict(scores=[_avg(sc)], cost=3, note=f'M3 {k} seed 3개 logit 평균')
    return out
