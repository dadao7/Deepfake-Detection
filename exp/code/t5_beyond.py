"""TASKS5 D: AUC·ACC 외 성능 지표 표 (자동 생성). 서술 문서는 reports/t5_beyond_auc.md.
A 위치(설명) 지표 — reports/s6_loc_summary.json (FF++ test 정렬 마스크 5,472프레임)
B 오탐·미탐 균형(기준값 0.5, 영상) — 점수 은행
C seed 안정성(seed 3개 AUC 표준편차) — 점수 은행
D 과적합 저항(val 균형 손실 상승폭, best→last 변화) — runs/*/curves.csv, registry.csv
E 위조 방식별 AUC(FF++ test 4방식 + FaceShifter) — registry.csv, 점수 은행
  python code/t5_beyond.py → reports/t5_beyond_tables.md"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import common as C  # noqa: E402
import t34_common as T  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_models as MD  # noqa: E402

LOC_ROWS = [('center|-', '얼굴 중앙 고정 지도(기준선)'), ('seg_head|-', 'SBI 경계 탐지기(seg head, P4b·P4)'),
            ('P3|last', 'P3 분류 heatmap (detach 유지)'), ('P3|best', 'P3 (best)'), ('P2|last', 'P2'),
            ('P4a|best', 'P4a (detach 제거)'), ('P4a|last', 'P4a (last)'), ('P4c_l1|best', 'anchor λ=1'),
            ('P4c_l10|best', 'anchor λ=10'), ('P4c_l100|best', 'anchor λ=100 (seed 3)'), ('P4c_l100|last', 'anchor λ=100 (last)'),
            ('P3_sbi|last', 'P3, SBI로 학습'), ('P4a_sbi|last', 'P4a, SBI로 학습')]


def reg():
    return {r['run_id']: r for r in csv.DictReader(open(os.path.join(C.EXP, 'registry.csv'), encoding='utf-8'))}


def main():
    md = ['# AUC·ACC 외 성능 지표 표 (자동 생성: code/t5_beyond.py)\n']
    # A
    loc = json.load(open(os.path.join(C.REPORTS, 's6_loc_summary.json'), encoding='utf-8'))['summary']
    md += ['## A. 위치(설명) 지표 — FF++ test 5,472프레임, 실제 조작 마스크 대비\n',
           '| heatmap | 경계 적중(%) | 경계 셀 AUROC | top-3 crop 적중(%) | crop 조작 비율 | 영역 셀 AUROC | IoU | 최댓값 적중(%) |',
           '|---|---|---|---|---|---|---|---|']
    for k, name in LOC_ROWS:
        if k not in loc:
            continue
        v = loc[k]
        g = lambda m, f='{:.1f}': (f.format(v[m][0]) + (f'±{v[m][1]:.1f}' if v[m][2] > 1 and m != 'iou' and m != 'crop_cov_k3' else '')) if m in v else '–'  # noqa: E731
        md.append(f'| {name} | {g("boundary_hit")} | {g("bnd_auc")} | {g("crop_hit_k3")} | {g("crop_cov_k3", "{:.2f}")} | '
                  f'{g("cell_auc")} | {g("iou", "{:.2f}")} | {g("max_hit")} |')
    t5loc = os.path.join(C.REPORTS, 't5_loc.json')
    if os.path.exists(t5loc):                      # TASKS5 병합·수프·증류 모델 (code/t5_loc.py)
        for k, v in json.load(open(t5loc, encoding='utf-8')).items():
            g = lambda m, f='{:.1f}': f.format(v[m][0]) + (f'±{v[m][1]:.1f}' if v[m][2] > 1 and m not in ('iou', 'crop_cov_k3') else '')  # noqa: E731
            md.append(f'| (TASKS5) {k.replace("|-", "")} | {g("boundary_hit")} | {g("bnd_auc")} | {g("crop_hit_k3")} | '
                      f'{g("crop_cov_k3", "{:.2f}")} | {g("cell_auc")} | {g("iou", "{:.2f}")} | {g("max_hit")} |')
    md.append('\n※ FF++ 마스크는 얼굴 대부분을 덮어 "영역 셀 AUROC·최댓값 적중·crop 적중"은 얼굴 중앙만 찍어도 높다(기준선이 1위). '
              '조작 경계를 찾는 능력은 **경계 적중·경계 셀 AUROC**로 본다(우연 수준: 경계 AUROC 50, 중앙 기준선 경계 적중 0%).\n')
    # B
    lab = T5.labels()
    models = MD.collect()
    show = [n for n in ('P0', 'P3', 'U3s', '결합2', '결합6', '병합', '병합TV', '수프(lin)', '수프(tv)', '혼합학습', 'WiSE대조')
            if n in models] + \
        [n for n in models if n.startswith('증류:') and not n.endswith('×3')]
    md += ['## B. 오탐·미탐 균형 (기준값 0.5, 영상 단위, seed 평균)\n',
           '| 모델 | ' + ' | '.join(f'{s} 오탐/미탐/bACC' for s in T.REPORT_SETS) + ' |', '|---|' + '---|' * 4]
    for n in show:
        cells = []
        for s in T.REPORT_SETS:
            l = lab[s].astype(bool)
            fp = np.mean([((d[s] >= 0.5) & ~l).sum() / (~l).sum() * 100 for d in models[n]['scores']])
            fn = np.mean([((d[s] < 0.5) & l).sum() / l.sum() * 100 for d in models[n]['scores']])
            cells.append(f'{fp:.1f} / {fn:.1f} / {100 - (fp + fn) / 2:.1f}')
        md.append(f'| {n} | ' + ' | '.join(cells) + ' |')
    md += ['', '### B′. bACC 짝 비교 (기준값 0.5)\n', '| 비교 | ' + ' | '.join(T.REPORT_SETS) + ' |', '|---|' + '---|' * 4]
    for a, b in [('P3', 'P0')] + [(n, 'P3') for n in show if n not in ('P0', 'P3')]:
        cells = [T5.fmt(T5.boot_cmp(lab[s], [d[s] for d in models[a]['scores']], [d[s] for d in models[b]['scores']], B=1000), 'bacc')
                 for s in T.REPORT_SETS]
        md.append(f'| {a} − {b} | ' + ' | '.join(cells) + ' |')
    # C
    bank = T.load_bank()
    seeds3 = [('P2 last', 'S4|P2|{}|last'), ('P3 last', 'S4|P3|{}|last'), ('P3 best', 'S4|P3|{}|best'),
              ('P4a best', 'S4|P4a|{}|best'), ('P4a last', 'S4|P4a|{}|last'), ('U3s best', 'S9|U3s|{}|best'),
              ('U3s last', 'S9|U3s|{}|last'), ('U3m best', 'MIX|U3m|{}|best'), ('anchor λ=100 best', 'ANC|P4c_l100|{}|best')]
    md += ['', '## C. seed 안정성 — seed 3개 AUC 표준편차 (작을수록 재현성 높음)\n',
           '| 모델 | ' + ' | '.join(T5.ALL) + ' | 평균 |', '|---|' + '---|' * (len(T5.ALL) + 1)]
    for name, fmt in seeds3:
        if not all(f'score|fftest|{fmt.format(s)}' in bank for s in (0, 1, 2)):
            continue
        sd = [np.std([T.auc(lab[s], bank[f'score|{s}|{fmt.format(k)}']) for k in (0, 1, 2)]) for s in T5.ALL]
        md.append(f'| {name} | ' + ' | '.join(f'{x:.2f}' for x in sd) + f' | {np.mean(sd):.2f} |')
    for n in show:
        if len(models[n]['scores']) == 3:
            sd = [np.std([T.auc(lab[s], d[s]) for d in models[n]['scores']]) for s in T5.ALL]
            md.append(f'| {n} | ' + ' | '.join(f'{x:.2f}' for x in sd) + f' | {np.mean(sd):.2f} |')
    # D
    R = reg()
    md += ['', '## D. 과적합 저항 — 학습을 오래 했을 때 무너지는 정도 (seed 3개 평균)\n',
           '| 설정 | val 균형 손실 상승폭(last − 최저) | FF++ AUC best→last | Celeb-DF AUC best→last |', '|---|---|---|---|']
    for name, pat in [('P1', 'P1_d100_s{}'), ('P2 (고정 α=0.5)', 'P2_d100_s{}'), ('P3 (동적 결합 + 패치 정규화)', 'P3_d100_s{}'),
                      ('P4a (detach 제거)', 'P4a_d100_s{}'), ('P2, SBI 학습', 'P2_sbi_s{}'), ('P3, SBI 학습', 'P3_sbi_s{}'),
                      ('U3s', 'U3s_s{}'), ('anchor λ=100', 'P4c_l100_s{}')]:
        ids = [pat.format(s) for s in (0, 1, 2)]
        if not all(i in R for i in ids):
            continue
        rise = []
        for i in ids:
            v = [float(r['val_loss_bal']) for r in csv.DictReader(open(os.path.join(C.RUNS, i, 'curves.csv'), encoding='utf-8'))]
            rise.append(v[-1] - min(v))
        f = lambda k: np.mean([float(R[i][k]) for i in ids])  # noqa: E731
        md.append(f'| {name} | {np.mean(rise):.2f} | {f("best_ffpp_auc"):.2f} → {f("last_ffpp_auc"):.2f} | '
                  f'{f("best_cdf_auc"):.2f} → {f("last_cdf_auc"):.2f} |')
    # E
    md += ['', '## E. 위조 방식별 AUC (FF++ test 4방식은 학습에 있던 방식, FaceShifter는 처음 보는 방식)\n',
           '| 모델 | DF | F2F | FS | NT | FaceShifter |', '|---|---|---|---|---|---|']
    p0 = R['P0_ref']
    md.append(f'| P0 | {float(p0["best_auc_DF"]):.2f} | {float(p0["best_auc_F2F"]):.2f} | {float(p0["best_auc_FS"]):.2f} | '
              f'{float(p0["best_auc_NT"]):.2f} | {T.auc(lab["fsh"], bank["score|fsh|P0"]):.2f} |')
    ids = [f'P3_d100_s{s}' for s in (0, 1, 2)]
    f = lambda k: np.mean([float(R[i][k]) for i in ids])  # noqa: E731
    md.append(f'| P3 (last) | {f("last_auc_DF"):.2f} | {f("last_auc_F2F"):.2f} | {f("last_auc_FS"):.2f} | {f("last_auc_NT"):.2f} | '
              f'{np.mean([T.auc(lab["fsh"], bank[f"score|fsh|S4|P3|{s}|last"]) for s in (0, 1, 2)]):.2f} |')
    open(os.path.join(C.REPORTS, 't5_beyond_tables.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
