"""TASKS6 최종 비교 표 (자동 생성). 후보: TASKS5 기준(P0, P3, 병합TV) + TASKS6 각 방법의 선택 1위(+ 분해 대조).
선택 점수 = FF++ val·DFD-val 평균 AUC(seed 평균). 보고 세트는 선택에 쓰지 않음. Δ = 영상 짝 bootstrap 95% CI, ✔ = 0 밖.
선택 과적합 점검: 후보들의 선택 점수와 각 보고 세트 AUC의 순위 상관(Spearman)을 함께 보고.
  python code/t6_final.py → reports/t6_final_tables.md"""
import glob
import json
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')     # numpy BLAS 크래시 방지(PROGRESS 환경 주의)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import common as C  # noqa: E402
import t34_common as T  # noqa: E402
import t5_calib as CB  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_models as MD  # noqa: E402

SEEDS = (0, 1, 2)
LABEL = {'mod': 'N1 모듈별 비율', 'ties': 'N2 TIES', 'dare': 'N2 DARE', 'mod_top_only': '(분해) 상단만 병합',
         'mod_branch_only': '(분해) 분기만 병합'}


def spearman(x, y):
    """순위 상관(BLAS 없이 원소별 계산)"""
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def candidates():
    b5 = T5.load_bank5()
    base = MD.collect()
    out = {'P0': base['P0'], 'P3': base['P3'], '병합TV(TASKS5)': base['병합TV']}
    pick = json.load(open(os.path.join(C.REPORTS, 't6_merge_choice.json')))
    for fam, (_, n) in pick.items():
        out[LABEL.get(fam, fam)] = dict(scores=[{s: b5[f'score|{s}|T6|{n}|{sd}'] for s in T5.ALL} for sd in SEEDS], cost=1, note=n)
    hyb = os.path.join(C.REPORTS, 't6_hyb_choice.json')
    if os.path.exists(hyb):
        n = json.load(open(hyb))['best'][1]
        out['N1b 하이브리드'] = dict(scores=[{s: b5[f'score|{s}|T6|{n}|{sd}'] for s in T5.ALL} for sd in SEEDS], cost=1, note=n)
    for f in sorted(glob.glob(os.path.join(C.REPORTS, 't6_texp_choice_*.json'))):
        d = json.load(open(f))
        if all(f'score|fftest|T6X|{d["best"]}|{sd}' in b5 for sd in SEEDS):
            out[f'T {d["x"]} 병합'] = dict(scores=[{s: b5[f'score|{s}|T6X|{d["best"]}|{sd}'] for s in T5.ALL} for sd in SEEDS],
                                         cost=1, note=d['best'])
    return out


def main():
    lab = T5.labels()
    cand = candidates()
    st = {}
    for n, m in cand.items():
        sc = m['scores']
        taus = [CB.pick_tau(d, lab)[0] for d in sc]
        st[n] = {'sel': np.mean([np.mean([T.auc(lab[s], d[s]) for s in T.SELECT_SETS]) for d in sc]),
                 'auc': {s: np.mean([T.auc(lab[s], d[s]) for d in sc]) for s in T5.ALL}, 'tau': taus}
    order = sorted(st, key=lambda n: -st[n]['sel'])
    md = ['# TASKS6 최종 비교 (자동 생성: code/t6_final.py)\n',
          '선택 점수 = FF++ val·DFD-val 평균 AUC(seed 3개 평균). 보고 세트는 고르는 데 쓰지 않음. Δ = [영상 짝 bootstrap 95% CI], ✔ = 0 밖.\n',
          '## 1. 후보 (선택 점수 순, 보고 세트 AUC)\n', '| 모델 | 선택 점수 | ffval | dfdval | ' + ' | '.join(T.REPORT_SETS) + ' | 구성 |', '|---|---|---|---|' + '---|' * 5]
    for n in order:
        a = st[n]['auc']
        md.append(f'| {n} | {st[n]["sel"]:.2f} | {a["ffval"]:.2f} | {a["dfdval"]:.2f} | ' + ' | '.join(f'{a[s]:.2f}' for s in T.REPORT_SETS) +
                  f' | {cand[n].get("note", "")} |')
    for ref in ('P3', '병합TV(TASKS5)'):
        md += ['', f'## 2. {ref} 대비 AUC\n', '| 모델 | ' + ' | '.join(T.REPORT_SETS) + ' |', '|---|' + '---|' * 4]
        for n in order:
            if n in ('P0', ref):
                continue
            cells = [T5.fmt(T5.boot_cmp(lab[s], [d[s] for d in cand[n]['scores']], [d[s] for d in cand[ref]['scores']], B=1000))
                     for s in T.REPORT_SETS]
            md.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    md += ['', '## 3. 정확도 (P3 기준값 0.5 대비, 새 모델은 선택 세트 보정 τ)\n', '| 모델 | ' + ' | '.join(f'{s} ΔACC / ΔbACC' for s in T.REPORT_SETS) + ' |',
           '|---|' + '---|' * 4]
    for n in order:
        if n in ('P0', 'P3'):
            continue
        cells = []
        for s in T.REPORT_SETS:
            d = T5.boot_cmp(lab[s], [x[s] for x in cand[n]['scores']], [x[s] for x in cand['P3']['scores']], B=1000,
                            th_new=st[n]['tau'], th_base=0.5)
            cells.append(f'{T5.fmt(d, "acc")} / {T5.fmt(d, "bacc")}')
        md.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    names = [n for n in order if n != 'P0']
    sel = [st[n]['sel'] for n in names]
    md += ['', '## 4. 선택 과적합 점검: 선택 점수와 보고 세트 AUC의 순위 상관 (P0 제외 후보들)\n', '| 보고 세트 | Spearman ρ | 해석 |', '|---|---|---|']
    for s in T.REPORT_SETS:
        r = spearman(np.array(sel), np.array([st[n]['auc'][s] for n in names]))
        md.append(f'| {s} | {r:+.2f} | {"선택 점수가 오르면 함께 오름" if r > 0.5 else ("관계 약함" if r > -0.3 else "선택 점수가 오르면 오히려 내려감")} |')
    open(os.path.join(C.REPORTS, 't6_final_tables.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
