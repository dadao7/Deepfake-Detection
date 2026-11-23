"""TASKS5 M2: 선택 세트 기준값 보정. 모델(seed)마다 FF++ val·DFD-val 평균 균형 ACC가 최대인 기준값 τ를 고정해 보고 세트에 적용.
AUC는 바뀌지 않는다(순위 불변). 보고: ACC·bACC @0.5 vs @τ, P3(@0.5)·P3(@τ) 대비 짝 bootstrap.
  python code/t5_calib.py   → reports/t5_calib.md, reports/t5_calib.json"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import common as C  # noqa: E402
import t34_common as T  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_models as MD  # noqa: E402

GRID = np.round(np.arange(0.01, 0.9901, 0.005), 3)


def pick_tau(d, lab):
    """선택 세트 평균 bACC 최대 τ (같으면 0.5에 가까운 쪽)"""
    sc = np.array([np.mean([T5.acc(lab[s], d[s], t)[1] for s in T.SELECT_SETS]) for t in GRID])
    best = np.flatnonzero(sc >= sc.max() - 1e-9)
    return float(GRID[best[np.argmin(np.abs(GRID[best] - 0.5))]]), float(sc.max())


def main():
    lab = T5.labels()
    models = MD.collect()
    taus = {n: [pick_tau(d, lab) for d in m['scores']] for n, m in models.items()}
    json.dump({n: v for n, v in taus.items()}, open(os.path.join(C.REPORTS, 't5_calib.json'), 'w'), indent=1, ensure_ascii=False)
    base = models['P3']['scores']
    tb = [t for t, _ in taus['P3']]
    md = ['# M2: 선택 세트 기준값 보정 (자동 생성: code/t5_calib.py)\n',
          '기준값 τ = FF++ val·DFD-val 평균 균형 ACC(bACC)가 최대인 값(모델·seed마다, 0.01~0.99 / 0.005 간격). 보고 세트는 τ를 고르는 데 쓰지 않았다.',
          'AUC는 기준값과 무관하다. ACC = 정확도, bACC = (TPR+TNR)/2. seed가 여러 개면 seed 평균. Δ = [영상 짝 bootstrap 95% CI], ✔ = 0 밖.',
          '보고 세트 구성(진짜:가짜 영상): FF++ test 140:560, Celeb-DF 178:340, FaceShifter 140:140, DFD-test 174:991 → 불균형 세트의 ACC는 가짜 쪽 성능에 크게 좌우된다.\n',
          '## 1. 모델별 기준값과 ACC (@0.5 → @τ)\n',
          '| 모델 | 비용 | τ (seed별) | 선택 bACC | ' + ' | '.join(f'{s} ACC' for s in T.REPORT_SETS) + ' | ' + ' | '.join(f'{s} bACC' for s in T.REPORT_SETS) + ' |',
          '|---|---|---|---|' + '---|' * 8]
    for n, m in models.items():
        ts = [t for t, _ in taus[n]]
        cells_a, cells_b = [], []
        for s in T.REPORT_SETS:
            a05 = np.mean([T5.acc(lab[s], d[s])[0] for d in m['scores']])
            at = np.mean([T5.acc(lab[s], d[s], t)[0] for d, t in zip(m['scores'], ts)])
            b05 = np.mean([T5.acc(lab[s], d[s])[1] for d in m['scores']])
            bt = np.mean([T5.acc(lab[s], d[s], t)[1] for d, t in zip(m['scores'], ts)])
            cells_a.append(f'{a05:.1f}→**{at:.1f}**')
            cells_b.append(f'{b05:.1f}→**{bt:.1f}**')
        md.append(f'| {n} | {m["cost"]} | {", ".join(f"{t:.3f}" for t in ts)} | {np.mean([v for _, v in taus[n]]):.2f} | ' + ' | '.join(cells_a) + ' | ' + ' | '.join(cells_b) + ' |')
    md += ['', '## 2. P3 대비 판정 (보정 기준값 적용 모델 vs P3)\n',
           '- "vs P3@0.5": 지금 앱처럼 P3를 기준값 0.5로 쓸 때와 비교. "vs P3@τ": P3도 같은 방식으로 보정했을 때와 비교(공정 비교).\n',
           '| 모델 | 세트 | ΔAUC vs P3 | ΔACC vs P3@0.5 | ΔbACC vs P3@0.5 | ΔACC vs P3@τ | ΔbACC vs P3@τ |', '|---|---|---|---|---|---|---|']
    for n, m in models.items():
        if n == 'P3':
            continue
        ts = [t for t, _ in taus[n]]
        for s in T.REPORT_SETS:
            new = [d[s] for d in m['scores']]
            b = [d[s] for d in base]
            d05 = T5.boot_cmp(lab[s], new, b, B=1000, th_new=ts, th_base=0.5)
            dt = T5.boot_cmp(lab[s], new, b, B=1000, th_new=ts, th_base=tb)
            md.append(f'| {n} | {s} | {T5.fmt(d05)} | {T5.fmt(d05, "acc")} | {T5.fmt(d05, "bacc")} | {T5.fmt(dt, "acc")} | {T5.fmt(dt, "bacc")} |')
    # P3 자신: 보정 효과
    md += ['', '## 3. 보정 자체의 효과 (같은 모델, @τ − @0.5)\n', '| 모델 | ' + ' | '.join(f'{s} ΔACC / ΔbACC' for s in T.REPORT_SETS) + ' |', '|---|' + '---|' * 4]
    for n in ('P0', 'P3'):
        m = models[n]
        ts = [t for t, _ in taus[n]]
        cells = []
        for s in T.REPORT_SETS:
            x = [d[s] for d in m['scores']]
            dd = T5.boot_cmp(lab[s], x, x, B=1000, th_new=ts, th_base=0.5)
            cells.append(f'{T5.fmt(dd, "acc")} / {T5.fmt(dd, "bacc")}')
        md.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    open(os.path.join(C.REPORTS, 't5_calib.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
