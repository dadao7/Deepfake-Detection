"""S8 분석 (PROTOCOL §14): SBI 학습 vs FF++ 학습(S4) vs P0. 출력: reports/s8_tables.md, s8_fig_compare.png, s8_fig_curves.png
numpy BLAS 우회(MKL_THREADING_LAYER=SEQUENTIAL)는 analyze.py와 같다."""
import json
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402
import common as C  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

SETS = ['P2', 'P3', 'P4a']
SEEDS = (0, 1, 2)
M = ['ffpp_auc', 'ffpp_bacc05', 'ffpp_fpr', 'ffpp_fnr', 'ffpp_frame_auc', 'auc_DF', 'auc_F2F', 'auc_FS', 'auc_NT',
     'cdf_auc', 'cdf_bacc05', 'cdf_fpr', 'cdf_fnr']
DM = ['ffpp_auc', 'cdf_auc', 'ffpp_bacc05', 'cdf_bacc05']


def get(df, kind, st, seed, col):
    if kind == 'ref':
        r = df[df['kind'] == 'ref']
    else:
        r = df[(df['kind'] == kind) & (df['setting'] == st) & (df['seed'] == seed) & (df['data_pct'] == 100)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def table(df):
    md = []
    for ck in ('best', 'last'):
        md += [f'\n### {ck} 체크포인트 (seed 평균±SD; S4 best=FF++ val 선택, S8 best=SBI val 선택)\n',
               '| 설정 | 학습 데이터 | ' + ' | '.join(A.LABEL.get(m, m) for m in M) + ' |', '|' + '---|' * (len(M) + 2)]
        md.append('| P0 | – | ' + ' | '.join(f'{get(df, "ref", None, None, f"{ck}_{m}"):.2f}' for m in M) + ' |')
        for st in SETS:
            for kind, lab in (('main', 'FF++ (S4)'), ('sbi', 'SBI (S8)')):
                cells = [A.fmt(*A.ms([get(df, kind, st, s, f'{ck}_{m}') for s in SEEDS])) for m in M]
                md.append(f'| {st} | {lab} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(md)


def deltas(df):
    comps = [('ref', None, 'sbi', st, f'P0→{st}(SBI)') for st in SETS] + \
            [('main', st, 'sbi', st, f'{st}: FF++→SBI 학습') for st in SETS] + \
            [('sbi', 'P2', 'sbi', 'P3', 'SBI 학습 안에서 P2→P3(동적 결합+M4)'),
             ('sbi', 'P3', 'sbi', 'P4a', 'SBI 학습 안에서 P3→P4a(detach 제거)')]
    md = ['\n| 비교 | 체크포인트 | 지표 | 평균 Δ | seed별 Δ | 부호 일치 | 2SD | 판정 |', '|---|---|---|---|---|---|---|---|']
    rows = []
    for ka, sa, kb, sb, desc in comps:
        for ck in ('best', 'last'):
            for m in DM:
                col = f'{ck}_{m}'
                va = np.array([get(df, ka, sa, s, col) for s in SEEDS])
                vb = np.array([get(df, kb, sb, s, col) for s in SEEDS])
                ok = np.isfinite(va) & np.isfinite(vb)
                if not ok.any():
                    continue
                d = (vb - va)[ok]
                sd = max(0.0 if ka == 'ref' else A.ms(va)[1], A.ms(vb)[1])
                mean = float(d.mean())
                sign = int((d > 0).sum()) if mean >= 0 else int((d < 0).sum())
                cons = len(d) == 3 and sign == 3 and np.all(d != 0)
                big = len(d) == 3 and abs(mean) >= 2 * sd
                verdict = ('향상' if mean > 0 else '저하') if (cons and big) else ('n<3' if len(d) < 3 else '차이 불분명')
                rows.append({'desc': desc, 'ckpt': ck, 'metric': m, 'mean_delta': mean,
                             'seed_deltas': ' '.join(f'{x:+.2f}' for x in d), 'sign': f'{sign}/{len(d)}',
                             'two_sd': 2 * sd, 'verdict': verdict})
                md.append(f'| {desc} | {ck} | {A.LABEL.get(m, m)} | {mean:+.2f} | {rows[-1]["seed_deltas"]} | '
                          f'{sign}/{len(d)} | {2 * sd:.2f} | **{verdict}** |')
    pd.DataFrame(rows).to_csv(os.path.join(C.REPORTS, 's8_deltas.csv'), index=False)
    return '\n'.join(md)


def diagnostics():
    te, cd = C.load_index('test'), C.load_index('cdf')
    from sklearn.metrics import roc_auc_score
    base = {}
    for ds, ix, f in (('ffpp', te, 'test'), ('cdf', cd, 'cdf')):
        lg = np.load(os.path.join(C.BASE_DIR, f'{f}_logit.npy'))
        p = 1 / (1 + np.exp(lg[:, 0] - lg[:, 1]))
        vs = C.video_scores(p, ix['vid'], 'ffpp' if ds == 'ffpp' else 'frac')
        vl = np.zeros(len(vs), int)
        vl[ix['vid']] = ix['label']
        base[ds] = (vs, vl)
    rows = []
    for st in SETS:
        for tag, pat in (('FF++ (S4)', f'{st}_d100_s{{}}'), ('SBI (S8)', f'{st}_sbi_s{{}}')):
            for s in SEEDS:
                for ck in ('best', 'last'):
                    pth = os.path.join(C.RUNS, pat.format(s), f'pred_{ck}.npz')
                    if not os.path.exists(pth):
                        continue
                    z = np.load(pth)
                    r = {'setting': st, 'train': tag, 'seed': s, 'ckpt': ck}
                    for ds, ix in (('ffpp', te), ('cdf', cd)):
                        bvs, vl = base[ds]
                        dvs = C.video_scores(z[f'{ds}_dprob'].astype(np.float64), ix['vid'], 'ffpp' if ds == 'ffpp' else 'frac')
                        fvs = z[f'{ds}_video_score']
                        r[f'{ds}_detail_auc'] = roc_auc_score(vl, dvs) * 100
                        r[f'{ds}_flip'] = float(((fvs >= 0.5) != (bvs >= 0.5)).mean() * 100)
                        r[f'{ds}_corr'] = A.pearson(fvs.astype(np.float64), bvs.astype(np.float64))
                        a = z[f'{ds}_alpha'].astype(np.float64)
                        av = np.bincount(ix['vid'], weights=a) / np.bincount(ix['vid'])
                        r[f'{ds}_alpha_real'], r[f'{ds}_alpha_fake'] = float(av[vl == 0].mean()), float(av[vl == 1].mean())
                    rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(C.REPORTS, 's8_diag.csv'), index=False)
    if not len(d):
        return ''
    g = d.groupby(['setting', 'train', 'ckpt']).mean(numeric_only=True).reset_index()
    md = ['\n| 설정 | 학습 | 체크포인트 | detail 단독 AUC (FF++/CDF) | 판정 바뀜% (FF++/CDF) | 결합-baseline 상관 (FF++/CDF) | '
          'α 진짜/가짜 (FF++) | α 진짜/가짜 (CDF) |', '|---|---|---|---|---|---|---|---|']
    for _, x in g.iterrows():
        md.append(f'| {x.setting} | {x.train} | {x.ckpt} | {x.ffpp_detail_auc:.1f} / {x.cdf_detail_auc:.1f} | '
                  f'{x.ffpp_flip:.1f} / {x.cdf_flip:.1f} | {x.ffpp_corr:.3f} / {x.cdf_corr:.3f} | '
                  f'{x.ffpp_alpha_real:.2f} / {x.ffpp_alpha_fake:.2f} | {x.cdf_alpha_real:.2f} / {x.cdf_alpha_fake:.2f} |')
    return '\n'.join(md)


def loc_part():
    p = os.path.join(C.REPORTS, 's6_loc_summary.json')
    if not os.path.exists(p):
        return ''
    s = json.load(open(p, encoding='utf-8'))['summary']
    md = ['\n| 대상 | 경계 AUROC | 경계 적중 % | 셀 AUROC | IoU | 최댓값 적중 % |', '|---|---|---|---|---|---|']
    for key in ('center|-', 'seg_head|-', 'P2|best', 'P3|best', 'P4a|best', 'P2_sbi|best', 'P3_sbi|best',
                'P4a_sbi|best', 'P3|last', 'P4a|last', 'P3_sbi|last', 'P4a_sbi|last'):
        if key not in s:
            continue
        v = s[key]
        f = lambda m, d=1: f'{v[m][0]:.{d}f}±{v[m][1]:.{d}f}' if v[m][2] > 1 else f'{v[m][0]:.{d}f}'  # noqa: E731
        md.append(f'| {key} | {f("bnd_auc")} | {f("boundary_hit")} | {f("cell_auc")} | {f("iou", 3)} | {f("max_hit")} |')
    return '\n'.join(md)


def fig_compare(df):
    A.style()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    ys = np.arange(len(SETS))[::-1]
    for ax, m, title in ((axs[0], 'ffpp_auc', 'FF++ c40 test 영상 AUC'), (axs[1], 'cdf_auc', 'Celeb-DF 영상 AUC')):
        p0 = get(df, 'ref', None, None, f'last_{m}')
        ax.axvline(p0, color=A.MUTED, lw=1, ls='--')
        ax.text(p0, len(SETS) - 0.45, f' P0 {p0:.2f}', color=A.INK2, fontsize=8)
        for j, (kind, lab, col) in enumerate((('main', 'FF++ 학습 (S4)', A.SER[0]), ('sbi', 'SBI 학습 (S8)', A.SER[1]))):
            off = 0.12 if j == 0 else -0.12
            for y, st in zip(ys, SETS):
                v = [get(df, kind, st, s, f'last_{m}') for s in SEEDS]
                mm, sd, n = A.ms(v)
                if not n:
                    continue
                ax.scatter([x for x in v if np.isfinite(x)], [y + off] * n, s=14, facecolors='none', edgecolors=col, lw=0.9)
                ax.errorbar(mm, y + off, xerr=sd, fmt='o', ms=6, color=col, elinewidth=1.5, label=lab if y == ys[0] else None)
        ax.set_yticks(ys)
        ax.set_yticklabels(SETS, color=A.INK2)
        ax.set_title(title, loc='left', color=A.INK)
        ax.set_xlabel('AUC (%)')
    axs[1].legend(fontsize=8, loc='lower left')
    fig.suptitle('학습 데이터에 따른 비교 (last 체크포인트 — 선택 방식과 무관, 속 빈 점 = seed)', x=0.01, ha='left',
                 color=A.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORTS, 's8_fig_compare.png'), dpi=150)
    plt.close(fig)


def fig_curves():
    A.style()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    keys = (('train_loss_fused', A.SER[0], 'train 결합 CE (SBI)'), ('val_loss_bal', A.SER[1], 'SBI val 균형 CE (선택)'),
            ('ffval_loss_bal', A.SER[2], 'FF++ val 균형 CE (기록만)'))
    for ax, st in zip(axs, SETS):
        for key, col, lab in keys:
            cs = [A.read_curves(f'{st}_sbi_s{s}') for s in SEEDS]
            cs = [c for c in cs if c is not None and key in c]
            for c in cs:
                ax.plot(c['epoch'], c[key], color=col, lw=0.8, alpha=0.35)
            if cs:
                L = min(len(c) for c in cs)
                ax.plot(np.arange(1, L + 1), np.mean([c[key].values[:L] for c in cs], 0), color=col, lw=2, label=lab)
        ax.set_title(f'{st} (SBI 학습)', loc='left', color=A.INK)
        ax.set_xlabel('epoch')
    axs[0].set_ylabel('손실')
    if axs[0].get_legend_handles_labels()[0]:
        axs[0].legend(fontsize=7)
    fig.suptitle('S8 손실 곡선 (굵은 선 = seed 평균)', x=0.01, ha='left', color=A.INK)
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORTS, 's8_fig_curves.png'), dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(C.REGISTRY)
    parts = ['# S8 분석 표 (자동 생성: code/analyze_s8.py)\n',
             f'kind=sbi 줄 수: {(df.kind == "sbi").sum()}\n', '## 1. 성능 비교', table(df),
             '\n## 2. 짝 비교 (같은 seed)', deltas(df), '\n## 3. 효과 진단', diagnostics(), '\n## 4. 위치 평가', loc_part()]
    open(os.path.join(C.REPORTS, 's8_tables.md'), 'w', encoding='utf-8').write('\n'.join(parts))
    fig_compare(df)
    fig_curves()
    print('reports/s8_tables.md, s8_fig_compare.png, s8_fig_curves.png 저장')


if __name__ == '__main__':
    main()
