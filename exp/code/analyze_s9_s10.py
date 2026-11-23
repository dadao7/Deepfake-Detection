"""S9(baseline 일부 풀기)·S10(DFD·FaceShifter) 분석 → reports/s9_tables.md, s10_tables.md, s9_fig_compare.png, s10_fig_compare.png
numpy BLAS 우회 내장."""
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

SEEDS = (0, 1, 2)
M9 = ['ffpp_auc', 'ffpp_bacc05', 'ffpp_fpr', 'ffpp_fnr', 'auc_DF', 'auc_F2F', 'auc_FS', 'auc_NT', 'cdf_auc', 'cdf_bacc05',
      'cdf_fpr', 'cdf_fnr']
# (표시 이름, registry kind, setting)
ROWS9 = [('P3 (분기만, FF++)', 'main', 'P3'), ('U0f (baseline 상단만, FF++)', 'unfreeze', 'U0f'),
         ('U3f (P3 + baseline 상단, FF++)', 'unfreeze', 'U3f'), ('P3 (분기만, SBI)', 'sbi', 'P3'),
         ('U0s (baseline 상단만, SBI)', 'unfreeze', 'U0s'), ('U3s (P3 + baseline 상단, SBI)', 'unfreeze', 'U3s')]
COMP9 = [(('ref', None), ('unfreeze', 'U0f'), 'P0 → U0f: baseline 상단 미세조정(FF++)'),
         (('ref', None), ('unfreeze', 'U0s'), 'P0 → U0s: baseline 상단 미세조정(SBI)'),
         (('unfreeze', 'U0f'), ('unfreeze', 'U3f'), 'U0f → U3f: 풀린 baseline에 P3 분기 추가(FF++)'),
         (('main', 'P3'), ('unfreeze', 'U3f'), 'P3 → U3f: P3에서 baseline 상단까지 풀기(FF++)'),
         (('unfreeze', 'U0s'), ('unfreeze', 'U3s'), 'U0s → U3s: 풀린 baseline에 P3 분기 추가(SBI)'),
         (('sbi', 'P3'), ('unfreeze', 'U3s'), 'P3(SBI) → U3s: baseline 상단까지 풀기(SBI)'),
         (('unfreeze', 'U0f'), ('unfreeze', 'U0s'), 'U0f → U0s: baseline 미세조정 데이터 FF++ → SBI')]
DM = ['ffpp_auc', 'cdf_auc', 'ffpp_bacc05', 'cdf_bacc05']


def get(df, kind, st, seed, col):
    if kind == 'ref':
        r = df[df['kind'] == 'ref']
    else:
        r = df[(df['kind'] == kind) & (df['setting'] == st) & (df['seed'] == seed) & (df['data_pct'] == 100)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def paired(va, vb, a_is_ref):
    ok = np.isfinite(va) & np.isfinite(vb)
    d = (vb - va)[ok]
    if not len(d):
        return None
    sd = max(0.0 if a_is_ref else A.ms(va)[1], A.ms(vb)[1])
    mean = float(d.mean())
    sign = int((d > 0).sum()) if mean >= 0 else int((d < 0).sum())
    cons = len(d) == 3 and sign == 3 and np.all(d != 0)
    big = len(d) == 3 and abs(mean) >= 2 * sd
    verdict = ('향상' if mean > 0 else '저하') if (cons and big) else ('n<3' if len(d) < 3 else '차이 불분명')
    return mean, d, sign, 2 * sd, verdict


def s9(df):
    md = ['# S9 분석 표 (자동 생성: code/analyze_s9_s10.py)\n', f'kind=unfreeze 줄 수: {(df.kind == "unfreeze").sum()}\n']
    for ck in ('best', 'last'):
        md += [f'\n## {ck} 체크포인트 (seed 평균±SD)\n', '| 설정 | ' + ' | '.join(A.LABEL.get(m, m) for m in M9) + ' |',
               '|' + '---|' * (len(M9) + 1),
               '| P0 | ' + ' | '.join(f'{get(df, "ref", None, None, f"{ck}_{m}"):.2f}' for m in M9) + ' |']
        for lab, kind, st in ROWS9:
            md.append(f'| {lab} | ' + ' | '.join(A.fmt(*A.ms([get(df, kind, st, s, f"{ck}_{m}") for s in SEEDS]))
                                                 for m in M9) + ' |')
    md += ['\n## 짝 비교 (같은 seed)\n', '| 비교 | 체크포인트 | 지표 | 평균 Δ | seed별 Δ | 부호 | 2SD | 판정 |',
           '|---|---|---|---|---|---|---|---|']
    for (ka, sa), (kb, sb), desc in COMP9:
        for ck in ('best', 'last'):
            for m in DM:
                va = np.array([get(df, ka, sa, s, f'{ck}_{m}') for s in SEEDS])
                vb = np.array([get(df, kb, sb, s, f'{ck}_{m}') for s in SEEDS])
                r = paired(va, vb, ka == 'ref')
                if r:
                    mean, d, sign, two_sd, v = r
                    md.append(f'| {desc} | {ck} | {A.LABEL.get(m, m)} | {mean:+.2f} | {" ".join(f"{x:+.2f}" for x in d)} | '
                              f'{sign}/{len(d)} | {two_sd:.2f} | **{v}** |')
    # U3에서 baseline logit만 따로 본 성능 (metrics.json의 base_only_*)
    md += ['\n## U3에서 학습된 baseline logit만 쓴 성능 (분기 제외)\n', '| 설정 | 체크포인트 | FF++ AUC (baseline만) | CDF AUC (baseline만) |',
           '|---|---|---|---|']
    for st in ('U0f', 'U3f', 'U0s', 'U3s'):
        for ck in ('best', 'last'):
            ff, cd = [], []
            for s in SEEDS:
                p = os.path.join(C.RUNS, f'{st}_s{s}', 'metrics.json')
                if os.path.exists(p):
                    mj = json.load(open(p))[ck]
                    ff.append(mj.get('base_only_ffpp_auc', np.nan))
                    cd.append(mj.get('base_only_cdf_auc', np.nan))
            md.append(f'| {st} | {ck} | {A.fmt(*A.ms(ff))} | {A.fmt(*A.ms(cd))} |')
    open(os.path.join(C.REPORTS, 's9_tables.md'), 'w', encoding='utf-8').write('\n'.join(md))
    # 그림: last 기준 FF++·CDF AUC
    A.style()
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    labs = ['P0'] + [r[0] for r in ROWS9]
    ys = np.arange(len(labs))[::-1]
    for ax, m, title in ((axs[0], 'ffpp_auc', 'FF++ c40 test 영상 AUC'), (axs[1], 'cdf_auc', 'Celeb-DF 영상 AUC')):
        p0 = get(df, 'ref', None, None, f'last_{m}')
        ax.axvline(p0, color=A.MUTED, lw=1, ls='--')
        ax.scatter([p0], [ys[0]], marker='D', s=36, color=A.INK2, zorder=4)
        for y, (lab, kind, st) in zip(ys[1:], ROWS9):
            col = A.SER[0] if 'FF++' in lab else A.SER[1]
            v = [get(df, kind, st, s, f'last_{m}') for s in SEEDS]
            mm, sd, n = A.ms(v)
            if n:
                ax.scatter([x for x in v if np.isfinite(x)], [y] * n, s=14, facecolors='none', edgecolors=col, lw=0.9)
                ax.errorbar(mm, y, xerr=sd, fmt='o', ms=6, color=col, elinewidth=1.5)
        ax.set_yticks(ys)
        ax.set_yticklabels(labs, color=A.INK2, fontsize=8)
        ax.set_title(title, loc='left', color=A.INK)
        ax.set_xlabel('AUC (%)')
    fig.suptitle('S9: baseline 상단 풀기 (last 체크포인트, 파랑 = FF++ 학습, 주황 = SBI 학습, 점선 = P0)', x=0.01, ha='left',
                 color=A.INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORTS, 's9_fig_compare.png'), dpi=150)
    plt.close(fig)


S10_SETS = [('S4', 'P1'), ('S4', 'P2'), ('S4', 'P3'), ('S4', 'P4a'), ('S8', 'P3_sbi'), ('S8', 'P4a_sbi'),
            ('S9', 'U0f'), ('S9', 'U3f'), ('S9', 'U0s'), ('S9', 'U3s')]
DSNAME = {'ffv': 'FF++ test(우리 정렬, 검증용)', 'fsh': 'FaceShifter(학습에 없던 방식)', 'dfd': 'DFD'}


def s10():
    p = os.path.join(C.REPORTS, 's10_eval.csv')
    if not os.path.exists(p):
        return
    ev = pd.read_csv(p)
    md = ['# S10 분석 표 (자동 생성: code/analyze_s9_s10.py)\n']
    g = os.path.join(C.REPORTS, 's10_gate.json')
    if os.path.exists(g):
        gj = json.load(open(g, encoding='utf-8'))
        md += ['## 재현 검증 게이트 (FF++ test, 우리 정렬 crop vs h5 crop)\n',
               f'- 비교 프레임 {gj["frames_compared"]}, NCC 중앙값 {gj["ncc_median"]:.4f}, 평균 {gj["ncc_mean"]:.4f}, '
               f'10%분위 {gj["ncc_p10"]:.4f}, 0.95 이상 {gj["ncc_ge_095"] * 100:.1f}%',
               f'- P0 영상 AUC: 우리 crop {gj["P0_ours"]["auc"]:.2f} vs h5 crop {gj["P0_h5"]["auc"]:.2f} '
               f'(Δ {gj["delta_auc_ours_minus_h5"]:+.2f}, 영상 {gj["n_videos_both"]}) → **{"통과" if gj["pass"] else "실패"}**\n']

    def cell(ds, stage, st, ck, col):
        r = ev[(ev.dataset == ds) & (ev.stage == stage) & (ev.setting == st) & (ev.ckpt == ck)]
        return A.ms(r[col].values.astype(float)) if len(r) else (np.nan, np.nan, 0)
    for ds in ('ffv', 'fsh', 'dfd'):
        if not (ev.dataset == ds).any():
            continue
        p0 = ev[(ev.dataset == ds) & (ev.stage == 'P0')].iloc[0]
        md += [f'\n## {DSNAME[ds]} — 영상 {int(p0.n_videos)}개(제외 {int(p0.n_excluded)})\n',
               '| 설정 | AUC best | AUC last | 균형ACC best | 오탐률 best | 미탐률 best |', '|---|---|---|---|---|---|',
               f'| P0 | {p0.auc:.2f} | {p0.auc:.2f} | {p0.bacc05:.2f} | {p0.fpr:.2f} | {p0.fnr:.2f} |']
        for stage, st in S10_SETS:
            md.append(f'| {st} | {A.fmt(*cell(ds, stage, st, "best", "auc"))} | {A.fmt(*cell(ds, stage, st, "last", "auc"))} | '
                      f'{A.fmt(*cell(ds, stage, st, "best", "bacc05"))} | {A.fmt(*cell(ds, stage, st, "best", "fpr"))} | '
                      f'{A.fmt(*cell(ds, stage, st, "best", "fnr"))} |')
        # P0 대비 짝 비교(AUC)
        md += ['\n| P0 대비 AUC (같은 seed) | 체크포인트 | 평균 Δ | seed별 Δ | 부호 | 판정 |', '|---|---|---|---|---|---|']
        for stage, st in S10_SETS:
            for ck in ('best', 'last'):
                r = ev[(ev.dataset == ds) & (ev.stage == stage) & (ev.setting == st) & (ev.ckpt == ck)].sort_values('seed')
                if len(r) == 0:
                    continue
                res = paired(np.full(len(r), p0.auc), r.auc.values.astype(float), True)
                if res:
                    mean, d, sign, two_sd, v = res
                    md.append(f'| P0 → {st} | {ck} | {mean:+.2f} | {" ".join(f"{x:+.2f}" for x in d)} | {sign}/{len(d)} | **{v}** |')
    open(os.path.join(C.REPORTS, 's10_tables.md'), 'w', encoding='utf-8').write('\n'.join(md))
    # 그림: 데이터셋별 AUC (last)
    A.style()
    dss = [d for d in ('fsh', 'dfd') if (ev.dataset == d).any()]
    if not dss:
        return
    fig, axs = plt.subplots(1, len(dss), figsize=(5.2 * len(dss), 4.2), sharey=True)
    axs = np.atleast_1d(axs)
    labs = ['P0'] + [st for _, st in S10_SETS]
    ys = np.arange(len(labs))[::-1]
    for ax, ds in zip(axs, dss):
        p0 = float(ev[(ev.dataset == ds) & (ev.stage == 'P0')].auc.iloc[0])
        ax.axvline(p0, color=A.MUTED, lw=1, ls='--')
        ax.scatter([p0], [ys[0]], marker='D', s=36, color=A.INK2, zorder=4)
        for y, (stage, st) in zip(ys[1:], S10_SETS):
            col = {'S4': A.SER[0], 'S8': A.SER[1], 'S9': A.SER[2]}[stage]
            r = ev[(ev.dataset == ds) & (ev.stage == stage) & (ev.setting == st) & (ev.ckpt == 'last')]
            if len(r):
                v = r.auc.values.astype(float)
                ax.scatter(v, [y] * len(v), s=14, facecolors='none', edgecolors=col, lw=0.9)
                ax.errorbar(v.mean(), y, xerr=v.std(ddof=1) if len(v) > 1 else 0, fmt='o', ms=6, color=col, elinewidth=1.5)
        ax.set_yticks(ys)
        ax.set_yticklabels(labs, color=A.INK2)
        ax.set_title(f'{DSNAME[ds]} 영상 AUC (last)', loc='left', color=A.INK, fontsize=9)
        ax.set_xlabel('AUC (%)')
    fig.suptitle('S10: 새 교차 데이터셋 (파랑 = S4 FF++ 학습 분기, 주황 = S8 SBI 학습, 청록 = S9 baseline 상단 풀기, 점선 = P0)',
                 x=0.01, ha='left', color=A.INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORTS, 's10_fig_compare.png'), dpi=150)
    plt.close(fig)


def main():
    df = pd.read_csv(C.REGISTRY)
    if (df.kind == 'unfreeze').any():
        s9(df)
    s10()
    print('s9_tables.md / s10_tables.md / 그림 저장')


if __name__ == '__main__':
    main()
