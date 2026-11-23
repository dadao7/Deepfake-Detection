"""S10 보조: 테스트 표본 불확실성 — 영상 단위 짝 bootstrap(2,000회)으로 P0 대비 AUC 차이의 95% 신뢰구간.
seed 일치는 학습 난수 변동만 반영하므로, 같은 영상을 다시 뽑는 bootstrap으로 테스트셋 크기(특히 FaceShifter 280영상)의 불확실성을 더한다.
모델 쪽 AUC는 매 반복마다 seed 3개 AUC의 평균. 대상: Celeb-DF(runs pred), DFD·FaceShifter(s10_frame_probs).
출력: reports/s10_bootstrap.csv, s10_bootstrap.md"""
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import eval_newdata as EV  # noqa: E402

B = 2000
COMPS = [('S4', 'P3', 'best'), ('S4', 'P3', 'last'), ('S4', 'P4a', 'best'), ('S4', 'P4a', 'last'),
         ('S4', 'P2', 'last'), ('S9', 'U0s', 'best'), ('S9', 'U0s', 'last'), ('S9', 'U3s', 'best'),
         ('S8', 'P3_sbi', 'best'), ('S9', 'U0f', 'last')]
RUNPAT = {'S4': '{st}_d100_s{s}', 'S8': '{st}_s{s}', 'S9': '{st}_s{s}'}


def newdata_scores(ds):
    D = EV.NewData(ds)
    z = np.load(os.path.join(C.REPORTS, f's10_frame_probs_{ds}.npz'))
    names = list(z['names'])
    vid, vl = D.meta['vid'], D.meta['video_label']
    V = len(vl)
    cnt = np.bincount(vid, minlength=V)
    ok = cnt >= EV.MIN_FRAMES

    def vs(p):
        out = np.full(V, np.nan)
        for v in np.where(ok)[0]:
            q = p[vid == v]
            out[v] = C.trimmed_mean(q, int(0.1 * len(q)))
        return out[ok]
    scores = {'P0': vs(z['p0'])}
    for stage, st, ck in COMPS:
        scores[(stage, st, ck)] = [vs(z['probs'][names.index(f'{stage}|{st}|{s}|{ck}')]) for s in (0, 1, 2)]
    return vl[ok], scores


def cdf_scores():
    cd = C.load_index('cdf')
    lg = np.load(os.path.join(C.BASE_DIR, 'cdf_logit.npy'))
    p0 = 1 / (1 + np.exp(lg[:, 0] - lg[:, 1]))
    v0 = C.video_scores(p0, cd['vid'], 'frac')
    vl = np.zeros(len(v0), int)
    vl[cd['vid']] = cd['label']
    scores = {'P0': v0}
    for stage, st, ck in COMPS:
        scores[(stage, st, ck)] = [np.load(os.path.join(C.RUNS, RUNPAT[stage].format(st=st.replace('_sbi', '_sbi') if stage != 'S8'
                                                                                         else st, s=s), f'pred_{ck}.npz'))['cdf_video_score']
                                   for s in (0, 1, 2)]
    return vl, scores


def boot(vl, scores, rng):
    n = len(vl)
    idx = [rng.integers(0, n, n) for _ in range(B)]
    idx = [i for i in idx if 0 < vl[i].sum() < len(i)]
    a0 = np.array([roc_auc_score(vl[i], scores['P0'][i]) for i in idx])
    rows = []
    for key in COMPS:
        ms = scores[key]
        point = np.mean([roc_auc_score(vl, m) for m in ms]) - roc_auc_score(vl, scores['P0'])
        d = np.array([np.mean([roc_auc_score(vl[i], m[i]) for m in ms]) for i in idx]) - a0
        lo, hi = np.quantile(d, [0.025, 0.975])
        rows.append({'model': f'{key[1]} ({key[2]})', 'delta': point * 100, 'ci_lo': lo * 100, 'ci_hi': hi * 100,
                     'excludes_0': bool(lo > 0 or hi < 0)})
    return rows


def main():
    rng = np.random.default_rng(0)
    allr = []
    for ds, fn in (('cdf', cdf_scores), ('fsh', lambda: newdata_scores('fsh')), ('dfd', lambda: newdata_scores('dfd'))):
        vl, sc = fn()
        for r in boot(vl, sc, rng):
            allr.append({'dataset': ds, 'n_videos': len(vl), **r})
        print(ds, '완료', flush=True)
    df = pd.DataFrame(allr)
    df.to_csv(os.path.join(C.REPORTS, 's10_bootstrap.csv'), index=False)
    md = ['# 테스트 표본 불확실성: P0 대비 AUC 차이 (영상 단위 짝 bootstrap 2,000회, 95% 신뢰구간, 모델 = seed 3개 평균)\n',
          '| 데이터셋 | 영상 수 | 모델 | Δ AUC | 95% CI | 0 제외 |', '|---|---|---|---|---|---|']
    for _, r in df.iterrows():
        md.append(f'| {r.dataset} | {r.n_videos} | {r.model} | {r.delta:+.2f} | [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] | '
                  f'{"✔" if r.excludes_0 else ""} |')
    open(os.path.join(C.REPORTS, 's10_bootstrap.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
