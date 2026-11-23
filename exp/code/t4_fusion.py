"""TASKS4 T4-7: 상보 결합 방식 비교 (점수 은행, seed 평균 전문가). FF++ 전문가 A = S4|P3|last, SBI 전문가 B = S9|U3s|best (+ 대안 쌍)
 (a) logit 평균  (b) 순위 평균(세트 안에서 영상 점수 순위를 [0,1]로, 라벨 없음)  (c) z점수 평균(세트 안 평균·표준편차, 라벨 없음)
 (d) 신뢰도 게이트: |logit_A| ≥ τ면 A만, 아니면 (1−w)A + wB
선택 = FF++ val·DFD-val 평균 AUC. 보고 세트에서 P0 대비 bootstrap. 출력: reports/t4_fusion.md"""
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import t34_common as T  # noqa: E402

ALL = T.SELECT_SETS + T.REPORT_SETS
W = np.round(np.arange(0.1, 1.0, 0.1), 2)


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def expert(bank, g):
    keys = [k for k in T.model_keys(bank) if k.startswith(g.split('|')[0] + '|' + g.split('|')[1] + '|') and k.endswith('|' + g.split('|')[2])]
    return {s: np.mean([logit(bank[f'score|{s}|{k}']) for k in keys], 0) for s in ALL}, len(keys)


def rank01(x):
    r = np.argsort(np.argsort(x, kind='stable'), kind='stable').astype(np.float64)
    return r / max(len(x) - 1, 1)


def main():
    bank = T.load_bank()
    lab = {s: bank[f'label|{s}'] for s in ALL}
    p0 = {s: logit(bank[f'score|{s}|P0']) for s in ALL}
    pairs = [('S4|P3|last', 'S9|U3s|best'), ('P0', 'S9|U3s|best'), ('S4|P3|last', 'S9|U0s|best'), ('S4|P3|last', 'MIX|U3m|best')]
    md = ['# T4-7: 상보 결합 방식 비교 (자동 생성: code/t4_fusion.py)\n',
          '전문가 = seed 3개 logit 평균. 선택 = FF++ val·DFD-val 평균 AUC. Δ = P0 대비 [영상 bootstrap 95% CI], ✔ = 0 밖.\n',
          '| 전문가 쌍 | 방식 | 선택값 | 선택 점수 | ' + ' | '.join(f'Δ {s}' for s in T.REPORT_SETS) + ' |', '|---|---|---|---|' + '---|' * 4]
    for ga, gb in pairs:
        A = p0 if ga == 'P0' else expert(bank, ga)[0]
        B, nb = expert(bank, gb)
        if nb == 0:
            continue
        variants = {}
        for w in W:
            variants[f'logit w={w}'] = {s: (1 - w) * A[s] + w * B[s] for s in ALL}
            variants[f'rank w={w}'] = {s: (1 - w) * rank01(A[s]) + w * rank01(B[s]) for s in ALL}
            variants[f'z w={w}'] = {s: (1 - w) * (A[s] - A[s].mean()) / A[s].std() + w * (B[s] - B[s].mean()) / B[s].std() for s in ALL}
            for tau in (1.0, 2.0, 3.0):
                variants[f'gate τ={tau} w={w}'] = {s: np.where(np.abs(A[s]) >= tau, A[s], (1 - w) * A[s] + w * B[s]) for s in ALL}
        fam = {}
        for name, sc in variants.items():
            sel = float(np.mean([T.auc(lab[s], sc[s]) for s in T.SELECT_SETS]))
            f = name.split()[0]
            if f not in fam or sel > fam[f][0]:
                fam[f] = (sel, name, sc)
        for f, (sel, name, sc) in fam.items():
            cells = []
            for s in T.REPORT_SETS:
                d, lo, hi = T.boot_delta(lab[s], [sc[s]], p0[s], B=1000)
                cells.append(f'{d:+.2f} [{lo:+.2f}, {hi:+.2f}]{" ✔" if (lo > 0 or hi < 0) else ""}')
            md.append(f'| {ga} + {gb} | {f} | {name} | {sel:.2f} | ' + ' | '.join(cells) + ' |')
        print(ga, gb, '완료', flush=True)
    open(os.path.join(C.REPORTS, 't4_fusion.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
