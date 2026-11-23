"""TASKS5 M4: 모든 후보의 최종 비교 표 (자동 생성). 서술은 reports/t5_final.md.
선택 점수 = FF++ val·DFD-val 평균 AUC(seed 평균). 보고 세트는 선택에 쓰지 않음. 기준값 τ = 선택 세트 평균 bACC 최대(t5_calib 규칙).
  python code/t5_final.py → reports/t5_final_tables.md, reports/t5_final.json"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import common as C  # noqa: E402
import t34_common as T  # noqa: E402
import t5_calib as CB  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_models as MD  # noqa: E402

KEY = ['병합TV', '병합', '수프(lin)', '수프(tv)', '병합TV+TTA', '병합+TTA', '병합TV×3', '결합2', '결합6', '혼합학습', 'WiSE대조', 'P3+TTA', 'P3×3']
CONTRASTS = [('병합TV', 'WiSE대조', 'FF++ 전문가(P3) task vector의 기여'), ('병합TV', '혼합학습', '병합 vs 한 모델에 섞어 학습'),
             ('병합TV', '결합2', '단일 병합 모델 vs 모델 2개 결합'), ('병합TV', 'U3s', '병합 vs SBI 전문가 단독')]


def main():
    lab = T5.labels()
    models = MD.collect()
    for n in list(models):
        if n.startswith('증류:') and not n.endswith('×3'):
            KEY.insert(4, n)
            CONTRASTS.append(('병합TV', n, '병합 vs 증류'))
    taus = {n: [CB.pick_tau(d, lab)[0] for d in m['scores']] for n, m in models.items()}
    stat = {}
    for n, m in models.items():
        sc = m['scores']
        stat[n] = {'cost': m['cost'], 'note': m['note'], 'tau': taus[n],
                   'sel': float(np.mean([np.mean([T.auc(lab[s], d[s]) for s in T.SELECT_SETS]) for d in sc])),
                   'auc': {s: float(np.mean([T.auc(lab[s], d[s]) for d in sc])) for s in T5.ALL},
                   'acc05': {s: float(np.mean([T5.acc(lab[s], d[s])[0] for d in sc])) for s in T.REPORT_SETS},
                   'bacc05': {s: float(np.mean([T5.acc(lab[s], d[s])[1] for d in sc])) for s in T.REPORT_SETS},
                   'acct': {s: float(np.mean([T5.acc(lab[s], d[s], t)[0] for d, t in zip(sc, taus[n])])) for s in T.REPORT_SETS},
                   'bacct': {s: float(np.mean([T5.acc(lab[s], d[s], t)[1] for d, t in zip(sc, taus[n])])) for s in T.REPORT_SETS}}
    order = sorted(stat, key=lambda n: -stat[n]['sel'])
    md = ['# TASKS5 최종 비교 표 (자동 생성: code/t5_final.py)\n',
          '선택 점수 = FF++ val·DFD-val 평균 AUC(seed 평균) — 보고 세트(FF++ test, Celeb-DF, FaceShifter, DFD-test)는 고르는 데 쓰지 않음.',
          '비용 = 영상 1개 판정에 돌리는 모델 수(TTA는 2배). ACC/bACC@τ = 선택 세트 bACC 최대 기준값(모델·seed마다). Δ = [영상 짝 bootstrap 95% CI], ✔ = 0 밖.\n',
          '## 1. 전체 후보 (선택 점수 순, 보고 세트 AUC)\n',
          '| 순위 | 모델 | 비용 | 선택 점수 | ' + ' | '.join(f'{s} AUC' for s in T.REPORT_SETS) + ' | 설명 |', '|---|---|---|---|' + '---|' * 5]
    for i, n in enumerate(order, 1):
        st = stat[n]
        md.append(f'| {i} | {n} | {st["cost"]} | {st["sel"]:.2f} | ' + ' | '.join(f'{st["auc"][s]:.2f}' for s in T.REPORT_SETS) +
                  f' | {st["note"]} |')
    md += ['', '## 2. 정확도 (기준값 0.5 / 선택 세트 보정 τ)\n',
           '| 모델 | ' + ' | '.join(f'{s} ACC 0.5→τ' for s in T.REPORT_SETS) + ' | ' + ' | '.join(f'{s} bACC 0.5→τ' for s in T.REPORT_SETS) + ' |',
           '|---|' + '---|' * 8]
    for n in ['P0', 'P3'] + [k for k in KEY if k in stat]:
        st = stat[n]
        md.append(f'| {n} | ' + ' | '.join(f'{st["acc05"][s]:.1f}→{st["acct"][s]:.1f}' for s in T.REPORT_SETS) + ' | ' +
                  ' | '.join(f'{st["bacc05"][s]:.1f}→{st["bacct"][s]:.1f}' for s in T.REPORT_SETS) + ' |')
    base = models['P3']['scores']
    md += ['', '## 3. P3(현재 모델, 기준값 0.5) 대비 판정\n',
           '- ΔACC·ΔbACC(0.5): 두 모델 모두 기준값 0.5. ΔACC·ΔbACC(τ): 새 모델은 선택 세트 보정 τ, P3는 0.5(현재 앱 설정).\n',
           '| 모델 | 세트 | ΔAUC | ΔACC(0.5) | ΔbACC(0.5) | ΔACC(τ) | ΔbACC(τ) |', '|---|---|---|---|---|---|---|']
    out = {'stat': stat, 'vsP3': {}}
    for n in [k for k in KEY if k in stat]:
        sc = models[n]['scores']
        for s in T.REPORT_SETS:
            new, b = [d[s] for d in sc], [d[s] for d in base]
            d05 = T5.boot_cmp(lab[s], new, b, B=1000)
            dt = T5.boot_cmp(lab[s], new, b, B=1000, th_new=taus[n], th_base=0.5)
            out['vsP3'][f'{n}|{s}'] = {'05': d05, 'tau': dt}
            md.append(f'| {n} | {s} | {T5.fmt(d05)} | {T5.fmt(d05, "acc")} | {T5.fmt(d05, "bacc")} | {T5.fmt(dt, "acc")} | {T5.fmt(dt, "bacc")} |')
    md += ['', '## 4. 대조 비교 (AUC, 같은 seed끼리)\n', '| 비교 | 질문 | ' + ' | '.join(T.REPORT_SETS) + ' |', '|---|---|' + '---|' * 4]
    for a, b, q in CONTRASTS:
        if a in models and b in models:
            cells = [T5.fmt(T5.boot_cmp(lab[s], [d[s] for d in models[a]['scores']], [d[s] for d in models[b]['scores']], B=1000))
                     for s in T.REPORT_SETS]
            md.append(f'| {a} − {b} | {q} | ' + ' | '.join(cells) + ' |')
    md += ['', '## 5. 좌우 반전 TTA 효과 (같은 모델, TTA − 없음)\n', '| 모델 | ' + ' | '.join(f'{s} ΔAUC' for s in T5.ALL) + ' |',
           '|---|' + '---|' * len(T5.ALL)]
    for n in ('P0', 'P3', '병합', '병합TV'):
        if f'{n}+TTA' in models:
            cells = [T5.fmt(T5.boot_cmp(lab[s], [d[s] for d in models[f'{n}+TTA']['scores']], [d[s] for d in models[n]['scores']], B=1000))
                     for s in T5.ALL]
            md.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    open(os.path.join(C.REPORTS, 't5_final_tables.md'), 'w', encoding='utf-8').write('\n'.join(md))
    json.dump(out, open(os.path.join(C.REPORTS, 't5_final.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print('\n'.join(md))


if __name__ == '__main__':
    main()
