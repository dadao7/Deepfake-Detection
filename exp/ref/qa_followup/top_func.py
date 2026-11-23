"""상단 단독 출력이 P0(저자 원본) 출력과 얼마나 다른가. 읽기 전용."""
import numpy as np

EXP = 'C:/m2f2/exp'


def spearmanr(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra, rb = ra - ra.mean(), rb - rb.mean()       # np.corrcoef(BLAS)가 이 환경에서 종료되므로 원소 연산으로
    return (float((ra * rb).sum() / np.sqrt((ra * ra).sum() * (rb * rb).sum())),)


def softmax1(l):
    l = l.astype(np.float64)
    e = np.exp(l - l.max(1, keepdims=True))
    return e[:, 1] / e.sum(1)


ref = {ds: softmax1(np.load(f'{EXP}/cache/base/{f}_logit.npy')) for ds, f in [('ffpp', 'test'), ('cdf', 'cdf')]}
print({k: (v.shape, round(float(v.mean()), 3)) for k, v in ref.items()})
for d in ['s', 'f', 'm']:
    for a in ['U0', 'U3']:
        for s in range(3):
            p = np.load(f'{EXP}/runs/{a}{d}_s{s}/pred_last.npz')
            r = []
            for ds in ['ffpp', 'cdf']:
                b, q = p[ds + '_base_prob'], ref[ds]
                r.append(f'{ds}: rho {spearmanr(b, q)[0]:.3f} |dp| {np.abs(b - q).mean():.3f} '
                         f'flip {((b > .5) != (q > .5)).mean() * 100:.1f}%')
            print(f'{a}{d}_s{s}', ' | '.join(r))
