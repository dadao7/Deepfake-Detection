"""S7 분석: registry.csv + runs/*/curves.csv + pred_*.npz → reports/ 표·그림 (부분 결과에도 동작)
표: s7_main_table.{csv,md}, s7_deltas.{csv,md}, s7_datasize.{csv,md}, s7_diag.csv, s7_leakage.json
그림: s7_fig_ablation.png, s7_fig_curves_100.png, s7_fig_datasize.png, s7_fig_datasize_curves.png
numpy BLAS: 이 PC는 MKL 멀티스레드 계층(libiomp5md.dll)이 깨져 행렬곱에서 즉사 → numpy를 불러오기 전에
MKL_THREADING_LAYER=SEQUENTIAL로 우회(프로세스 단위, 환경 수정 아님). matplotlib 좌표 변환이 행렬곱을 쓰기 때문."""
import json
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

SETS = ['P0', 'P1', 'P1p', 'P2a', 'P2b', 'P2', 'P3a', 'P3b', 'P3', 'P4a']
TRAINED = SETS[1:]
NAME = {'P1p': "P1′"}
SEEDS = (0, 1, 2)
MAIN_M = ['ffpp_auc', 'ffpp_acc05', 'ffpp_bacc05', 'ffpp_fpr', 'ffpp_fnr', 'ffpp_frame_auc', 'auc_DF', 'auc_F2F',
          'auc_FS', 'auc_NT', 'cdf_auc', 'cdf_bacc05', 'cdf_fpr', 'cdf_fnr', 'ffpp_accbest']
LABEL = {'ffpp_auc': 'FF++ AUC', 'ffpp_acc05': 'ACC@0.5', 'ffpp_bacc05': '균형ACC', 'ffpp_fpr': '오탐률',
         'ffpp_fnr': '미탐률', 'ffpp_frame_auc': '프레임AUC', 'auc_DF': 'DF', 'auc_F2F': 'F2F', 'auc_FS': 'FS',
         'auc_NT': 'NT', 'cdf_auc': 'CDF AUC', 'cdf_bacc05': 'CDF 균형ACC', 'cdf_fpr': 'CDF 오탐률',
         'cdf_fnr': 'CDF 미탐률', 'ffpp_accbest': '(최적기준ACC)', 'cdf_accbest': '(CDF 최적기준ACC)'}
COMPARE = [('P0', 'P1', 'detail 분기 추가'), ('P1', 'P1p', 'SBI 초기화'), ('P1p', 'P2a', 'heatmap 위치(k=1)'),
           ('P1p', 'P2b', 'Top-3 다중 crop(heatmap 없음)'), ('P1p', 'P2', 'heatmap + Top-3'),
           ('P2a', 'P2', 'heatmap에 Top-3 추가'), ('P2b', 'P2', 'Top-3에 heatmap 추가'),
           ('P2', 'P3a', '패치 정규화(M4)'), ('P2', 'P3b', '동적 결합'), ('P2', 'P3', 'M4 + 동적 결합'),
           ('P3', 'P4a', 'detach 제거'), ('P0', 'P3', '현재 모델 vs 논문 모델'), ('P0', 'P4a', 'P4a vs 논문 모델')]
DELTA_M = ['ffpp_auc', 'cdf_auc', 'ffpp_bacc05', 'ffpp_fpr', 'cdf_bacc05', 'ffpp_frame_auc']
# 그림 토큰 (dataviz 기준 팔레트, 라이트)
INK, INK2, MUTED, GRID, BASE, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#fcfcfb'
SER = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']


def nm(s):
    return NAME.get(s, s)


def style():
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'figure.facecolor': SURF,
                         'axes.facecolor': SURF, 'axes.edgecolor': BASE, 'axes.labelcolor': INK2,
                         'xtick.color': MUTED, 'ytick.color': MUTED, 'text.color': INK, 'axes.grid': True,
                         'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 9, 'axes.titlesize': 10, 'legend.frameon': False})


def load_registry():
    df = pd.read_csv(C.REGISTRY)
    return df[df['kind'].isin(['main', 'ref'])].copy()


def val(df, setting, pct, seed, col):
    if setting == 'P0':
        r = df[df['setting'] == 'P0']
    else:
        r = df[(df['setting'] == setting) & (df['data_pct'] == pct) & (df['seed'] == seed)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def ms(x):
    x = np.asarray([v for v in x if np.isfinite(v)], float)
    if not len(x):
        return np.nan, np.nan, 0
    return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0, len(x)


def fmt(m, s, n):
    if n == 0 or not np.isfinite(m):
        return '–'
    return f'{m:.2f}' if n == 1 else f'{m:.2f}±{s:.2f}'


def main_table(df):
    rows, md = [], []
    for ck in ('best', 'last'):
        md.append(f'\n### 100% 데이터, {ck} 체크포인트 (seed 평균±표준편차, n=3; P0은 학습 없음)\n')
        md.append('| 설정 | ' + ' | '.join(LABEL[m] for m in MAIN_M) + ' |')
        md.append('|' + '---|' * (len(MAIN_M) + 1))
        for st in SETS:
            cells = []
            for m in MAIN_M:
                vals = [val(df, st, 100, s, f'{ck}_{m}') for s in (SEEDS if st != 'P0' else (None,))]
                mm, sd, n = ms(vals)
                rows.append({'setting': st, 'ckpt': ck, 'metric': m, 'mean': mm, 'sd': sd, 'n': n})
                cells.append(fmt(mm, sd, n))
            md.append(f'| {nm(st)} | ' + ' | '.join(cells) + ' |')
    pd.DataFrame(rows).to_csv(os.path.join(C.REPORTS, 's7_main_table.csv'), index=False)
    return '\n'.join(md)


def deltas(df):
    rows, md = [], ['\n| 비교 | 체크포인트 | 지표 | 평균 Δ | 짝 SD | seed별 Δ | 부호 일치 | 2SD 기준 | 판정 |',
                    '|---|---|---|---|---|---|---|---|---|']
    for a, b, desc in COMPARE:
        for ck in ('best', 'last'):
            for m in DELTA_M:
                col = f'{ck}_{m}'
                va = np.array([val(df, a, 100, s, col) for s in SEEDS])
                vb = np.array([val(df, b, 100, s, col) for s in SEEDS])
                ok = np.isfinite(va) & np.isfinite(vb)
                if ok.sum() == 0:
                    continue
                d = (vb - va)[ok]
                sd_a = 0.0 if a == 'P0' else ms(va)[1]
                sd_b = ms(vb)[1]
                sd_seed = max(sd_a if np.isfinite(sd_a) else 0, sd_b if np.isfinite(sd_b) else 0)
                mean_d = float(d.mean())
                sign = int((d > 0).sum()) if mean_d >= 0 else int((d < 0).sum())
                consistent = sign == len(d) and len(d) == 3 and np.all(d != 0)
                big = abs(mean_d) >= 2 * sd_seed if len(d) == 3 else False
                better = (mean_d > 0) if 'fpr' not in m and 'fnr' not in m else (mean_d < 0)
                verdict = ('향상' if better else '저하') if (consistent and big) else ('n<3' if len(d) < 3 else '차이 불분명')
                rows.append({'from': a, 'to': b, 'desc': desc, 'ckpt': ck, 'metric': m, 'mean_delta': mean_d,
                             'paired_sd': float(d.std(ddof=1)) if len(d) > 1 else np.nan,
                             'seed_deltas': ' '.join(f'{x:+.2f}' for x in d), 'n': int(len(d)),
                             'sign_consistent': bool(consistent), 'sd_seed_max': sd_seed, 'meets_2sd': bool(big),
                             'verdict': verdict})
                if m in ('ffpp_auc', 'cdf_auc', 'ffpp_bacc05'):
                    md.append(f'| {nm(a)}→{nm(b)} {desc} | {ck} | {LABEL[m]} | {mean_d:+.2f} | '
                              f'{rows[-1]["paired_sd"]:.2f} | {rows[-1]["seed_deltas"]} | {sign}/{len(d)} | '
                              f'{"충족" if big else "미달"} (2SD={2 * sd_seed:.2f}) | **{verdict}** |')
    pd.DataFrame(rows).to_csv(os.path.join(C.REPORTS, 's7_deltas.csv'), index=False)
    return '\n'.join(md)


def read_curves(rid):
    p = os.path.join(C.RUNS, rid, 'curves.csv')
    return pd.read_csv(p) if os.path.exists(p) else None


def datasize(df):
    rows, md = [], ['\n| 설정 | 데이터 | FF++ AUC best | FF++ AUC last | CDF AUC best | CDF AUC last | best epoch | '
                    'val손실 최저 | val손실 last | 상승폭 | train 결합손실 last | 일반화 간격(last) |',
                    '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for st in ('P1', 'P2', 'P3', 'P4a'):
        for pct in (25, 50, 100):
            r = {'setting': st, 'pct': pct}
            for ck in ('best', 'last'):
                for m in ('ffpp_auc', 'cdf_auc', 'ffpp_bacc05'):
                    r[f'{ck}_{m}'] = ms([val(df, st, pct, s, f'{ck}_{m}') for s in SEEDS])
            cv = [read_curves(f'{st}_d{pct}_s{s}') for s in SEEDS]
            cv = [c for c in cv if c is not None and len(c) == 20]
            if cv:
                r['best_epoch'] = ms([c['val_loss_bal'].idxmin() + 1 for c in cv])
                r['vl_min'] = ms([c['val_loss_bal'].min() for c in cv])
                r['vl_last'] = ms([c['val_loss_bal'].iloc[-1] for c in cv])
                r['vl_rise'] = ms([c['val_loss_bal'].iloc[-1] - c['val_loss_bal'].min() for c in cv])
                r['tl_last'] = ms([c['train_loss_fused'].iloc[-1] for c in cv])
                r['gap_last'] = ms([c['val_loss_raw'].iloc[-1] - c['train_loss_fused'].iloc[-1] for c in cv])
            rows.append(r)
            g = lambda k: fmt(*r[k]) if k in r else '–'  # noqa: E731
            md.append(f'| {nm(st)} | {pct}% | {g("best_ffpp_auc")} | {g("last_ffpp_auc")} | {g("best_cdf_auc")} | '
                      f'{g("last_cdf_auc")} | {g("best_epoch")} | {g("vl_min")} | {g("vl_last")} | {g("vl_rise")} | '
                      f'{g("tl_last")} | {g("gap_last")} |')
    flat = [{k: (v[0] if isinstance(v, tuple) else v) for k, v in r.items()} for r in rows]
    pd.DataFrame(flat).to_csv(os.path.join(C.REPORTS, 's7_datasize.csv'), index=False)
    return '\n'.join(md)


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def diagnostics(df):
    """detail 단독 AUC, 동적 α, baseline 대비 판정 바뀜, 결합-baseline 상관 (FF++·CDF, 100% 데이터)"""
    te, cd = C.load_index('test'), C.load_index('cdf')
    base = {}
    for ds, ix in (('ffpp', te), ('cdf', cd)):
        lg = np.load(os.path.join(C.BASE_DIR, f'{"test" if ds == "ffpp" else "cdf"}_logit.npy'))
        p = 1 / (1 + np.exp(lg[:, 0] - lg[:, 1]))
        vs = C.video_scores(p, ix['vid'], 'ffpp' if ds == 'ffpp' else 'frac')
        vl = np.zeros(len(vs), int)
        vl[ix['vid']] = ix['label']
        base[ds] = (vs, vl)
    rows = []
    for st in TRAINED:
        for s in SEEDS:
            rid = f'{st}_d100_s{s}'
            for ck in ('best', 'last'):
                pth = os.path.join(C.RUNS, rid, f'pred_{ck}.npz')
                if not os.path.exists(pth):
                    continue
                z = np.load(pth)
                r = {'setting': st, 'seed': s, 'ckpt': ck}
                for ds, ix in (('ffpp', te), ('cdf', cd)):
                    trim = 'ffpp' if ds == 'ffpp' else 'frac'
                    bvs, vl = base[ds]
                    dvs = C.video_scores(z[f'{ds}_dprob'].astype(np.float64), ix['vid'], trim)
                    fvs = z[f'{ds}_video_score']
                    from sklearn.metrics import roc_auc_score
                    r[f'{ds}_detail_auc'] = roc_auc_score(vl, dvs) * 100
                    r[f'{ds}_fused_auc'] = roc_auc_score(vl, fvs) * 100
                    r[f'{ds}_base_auc'] = roc_auc_score(vl, bvs) * 100
                    r[f'{ds}_flip_rate'] = float(((fvs >= 0.5) != (bvs >= 0.5)).mean() * 100)
                    r[f'{ds}_fused_base_corr'] = pearson(fvs.astype(np.float64), bvs.astype(np.float64))
                    a = z[f'{ds}_alpha'].astype(np.float64)
                    av = np.bincount(ix['vid'], weights=a) / np.bincount(ix['vid'])
                    r[f'{ds}_alpha_real'] = float(av[vl == 0].mean())
                    r[f'{ds}_alpha_fake'] = float(av[vl == 1].mean())
                rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(C.REPORTS, 's7_diag.csv'), index=False)
    if not len(d):
        return ''
    g = d.groupby(['setting', 'ckpt']).mean(numeric_only=True).reset_index()
    md = ['\n| 설정 | 체크포인트 | detail 단독 AUC (FF++/CDF) | 결합 AUC (FF++/CDF) | 판정 바뀜% (FF++/CDF) | '
          '결합-baseline 상관 (FF++) | α 진짜/가짜 (FF++) |', '|---|---|---|---|---|---|---|']
    for st in TRAINED:
        for ck in ('best', 'last'):
            x = g[(g['setting'] == st) & (g['ckpt'] == ck)]
            if not len(x):
                continue
            x = x.iloc[0]
            md.append(f'| {nm(st)} | {ck} | {x.ffpp_detail_auc:.1f} / {x.cdf_detail_auc:.1f} | {x.ffpp_fused_auc:.2f} / '
                      f'{x.cdf_fused_auc:.2f} | {x.ffpp_flip_rate:.1f} / {x.cdf_flip_rate:.1f} | '
                      f'{x.ffpp_fused_base_corr:.3f} | {x.ffpp_alpha_real:.2f} / {x.ffpp_alpha_fake:.2f} |')
    md.append(f'\n(baseline 단독 = P0: FF++ AUC {d.ffpp_base_auc.iloc[0]:.2f}, CDF AUC {d.cdf_base_auc.iloc[0]:.2f})')
    return '\n'.join(md)


def leakage():
    """1학기 full_train_loader 분할(파일마다 키 이름순 앞 80% = 학습)과 공식 test의 겹침"""
    test_pairs = C.load_pairs('test')
    out = {}
    test_ids = {x for p in test_pairs for x in p}
    train_ids_seen = set()
    for cat in C.CATS:
        with h5py.File(C.h5_path(cat, 'train'), 'r') as f:
            keys = sorted(f.keys())
        cut = int(len(keys) * 0.8)
        tr = set(keys[:cut])
        tv = set(C.videos_of_pairs(test_pairs, cat))
        out[cat] = {'official_test_videos': len(tv), 'in_1st_sem_train': len(tv & tr)}
        for k in tr:
            train_ids_seen.update(k.split('_'))
    out['identity'] = {'official_test_ids': len(test_ids),
                       'test_ids_seen_in_1st_sem_train(as real or in fake pair)': len(test_ids & train_ids_seen)}
    json.dump(out, open(os.path.join(C.REPORTS, 's7_leakage.json'), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False)
    return out


def fig_ablation(df):
    style()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    ys = np.arange(len(SETS))[::-1]
    for ax, m, title in ((axs[0], 'ffpp_auc', 'FF++ c40 test 영상 AUC'), (axs[1], 'cdf_auc', 'Celeb-DF 영상 AUC')):
        p0 = val(df, 'P0', 100, None, f'best_{m}')
        ax.axvline(p0, color=MUTED, lw=1, ls='--', zorder=1)
        ax.scatter([p0], [ys[0]], marker='D', s=36, color=INK2, zorder=4)
        ax.text(p0, ys[0] + 0.25, f' {p0:.2f}', color=INK2, fontsize=8, va='bottom')
        for j, (ck, col) in enumerate((('best', SER[0]), ('last', SER[1]))):
            off = 0.14 if j == 0 else -0.14
            for y, st in zip(ys, SETS):
                if st == 'P0':
                    continue
                v = [val(df, st, 100, s, f'{ck}_{m}') for s in SEEDS]
                mm, sd, n = ms(v)
                if n == 0:
                    continue
                ax.scatter([x for x in v if np.isfinite(x)], [y + off] * n, s=14, facecolors='none', edgecolors=col,
                           linewidths=0.9, zorder=3)
                ax.errorbar(mm, y + off, xerr=sd, fmt='o', ms=6, color=col, ecolor=col, elinewidth=1.5, capsize=0,
                            zorder=4, label=ck if y == ys[1] else None)
        ax.set_yticks(ys)
        ax.set_yticklabels([nm(s) for s in SETS], color=INK2)
        ax.set_title(title, loc='left', color=INK)
        ax.set_xlabel('AUC (%)')
    axs[0].legend(loc='lower left', title='체크포인트', fontsize=8, title_fontsize=8)
    fig.suptitle('단계별 ablation (100% 데이터, seed 3개: 속 빈 점 = seed, 점·막대 = 평균±SD)', x=0.01, ha='left',
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORTS, 's7_fig_ablation.png'), dpi=150)
    plt.close(fig)


def fig_curves(pct=100, sets=TRAINED, fname='s7_fig_curves_100.png'):
    style()
    n = len(sets)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(11, 3.0 * rows), sharex=True, sharey=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, st in zip(axs, sets):
        for key, col, lab in (('train_loss_fused', SER[0], 'train 결합 CE'), ('val_loss_bal', SER[1], 'val 균형 CE')):
            cs = [read_curves(f'{st}_d{pct}_s{s}') for s in SEEDS]
            cs = [c for c in cs if c is not None]
            for c in cs:
                ax.plot(c['epoch'], c[key], color=col, lw=0.8, alpha=0.35)
            if cs:
                L = min(len(c) for c in cs)
                mean = np.mean([c[key].values[:L] for c in cs], 0)
                ax.plot(np.arange(1, L + 1), mean, color=col, lw=2, label=lab)
        ax.set_title(f'{nm(st)} ({pct}%)', loc='left', color=INK)
    for ax in axs[n:]:
        ax.axis('off')
    axs[0].legend(fontsize=8)
    for ax in axs[(rows - 1) * cols:]:
        ax.set_xlabel('epoch')
    for r in range(rows):
        axs[r * cols].set_ylabel('손실')
    fig.suptitle(f'epoch별 손실 곡선 ({pct}% 데이터, 굵은 선 = seed 평균, 옅은 선 = seed별)', x=0.01, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORTS, fname), dpi=150)
    plt.close(fig)


def fig_datasize(df):
    style()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for ax, m, title in ((axs[0], 'ffpp_auc', 'FF++ AUC (best)'), (axs[1], 'cdf_auc', 'Celeb-DF AUC (best)')):
        ends = []
        for col, st in zip(SER, ('P1', 'P2', 'P3', 'P4a')):
            xs, mean, sd = [], [], []
            for pct in (25, 50, 100):
                mm, s, n = ms([val(df, st, pct, sd_, f'best_{m}') for sd_ in SEEDS])
                if n:
                    xs.append(pct)
                    mean.append(mm)
                    sd.append(s)
            if not xs:
                continue
            ax.errorbar(xs, mean, yerr=sd, color=col, lw=2, marker='o', ms=6, capsize=0, label=nm(st))
            ends.append([mean[-1], nm(st), xs[-1]])
        ax.axhline(val(df, 'P0', 100, None, f'best_{m}'), color=MUTED, lw=1, ls='--')
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.045
        ends.sort()
        for k in range(1, len(ends)):                      # 끝 라벨 겹침 방지: 최소 간격으로 위로 밀기
            ends[k][0] = max(ends[k][0], ends[k - 1][0] + gap)
        for y, lab, x in ends:
            ax.text(x + 3, y, lab, color=INK2, fontsize=8, va='center')
        ax.set_xticks([25, 50, 100])
        ax.set_xlabel('학습 쌍 비율 (%)')
        ax.set_title(title, loc='left', color=INK)
    axs[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('데이터 양별 성능 (seed 평균±SD, 점선 = P0)', x=0.01, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORTS, 's7_fig_datasize.png'), dpi=150)
    plt.close(fig)


def fig_datasize_curves():
    style()
    sets = ('P1', 'P2', 'P3', 'P4a')
    fig, axs = plt.subplots(len(sets), 3, figsize=(10, 2.4 * len(sets)), sharex=True, sharey=True)
    for i, st in enumerate(sets):
        for j, pct in enumerate((25, 50, 100)):
            ax = axs[i, j]
            for key, col, lab in (('train_loss_fused', SER[0], 'train 결합 CE'), ('val_loss_bal', SER[1], 'val 균형 CE')):
                cs = [read_curves(f'{st}_d{pct}_s{s}') for s in SEEDS]
                cs = [c for c in cs if c is not None]
                if cs:
                    L = min(len(c) for c in cs)
                    ax.plot(np.arange(1, L + 1), np.mean([c[key].values[:L] for c in cs], 0), color=col, lw=2, label=lab)
            ax.set_title(f'{nm(st)} · {pct}%', loc='left', color=INK, fontsize=9)
    if axs[0, 0].get_legend_handles_labels()[0]:
        axs[0, 0].legend(fontsize=7)
    fig.suptitle('데이터 양별 손실 곡선 (seed 평균)', x=0.01, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(C.REPORTS, 's7_fig_datasize_curves.png'), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(C.REPORTS, exist_ok=True)
    df = load_registry()
    parts = ['# S7 분석 표 (자동 생성: code/analyze.py)\n', f'registry 줄 수(main+ref): {len(df)}\n',
             '## 1. 단계별 ablation 표', main_table(df), '\n## 2. 단계별 차이 (같은 seed 짝 비교)', deltas(df),
             '\n## 3. 데이터 양별', datasize(df), '\n## 4. 효과 진단', diagnostics(df)]
    lk = leakage()
    parts += ['\n## 5. 1학기 분할 누수 (재계산)', '```', json.dumps(lk, ensure_ascii=False, indent=1), '```']
    open(os.path.join(C.REPORTS, 's7_tables.md'), 'w', encoding='utf-8').write('\n'.join(parts))
    fig_ablation(df)
    fig_curves()
    fig_datasize(df)
    fig_datasize_curves()
    print('reports/s7_tables.md 와 그림 4개 저장')


if __name__ == '__main__':
    main()
