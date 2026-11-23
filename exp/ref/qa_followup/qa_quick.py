"""질문 방어용 빠른 분석 (학습 없음, 기존 점수 은행·캐시만 읽음). 2026-10-21.
⑤ 경계 적중 무작위 기준선  ⑨ 운영점(고정 오탐률 탐지율·배포 기준값·ECE)  ⑪ Holm 다중 비교 보정  ⑮ 병합 비율 민감도 그림
주의: 이 환경에서 numpy 행렬곱(BLAS)은 프로세스를 종료시키므로 쓰지 않는다(원소 연산·정렬만).
출력: reports/qa_followup.md, reports/qa_followup.json, reports/qa_merge_sensitivity.png
"""
import json
import os
import re

import numpy as np
from sklearn.metrics import roc_curve

EXP = r'C:\m2f2\exp'
REP = os.path.join(EXP, 'reports')
REPORT = ['fftest', 'cdf', 'fsh', 'dfdtest']
SELECT = ['ffval', 'dfdval']
NAME = {'fftest': 'FF++ test', 'cdf': 'Celeb-DF', 'fsh': 'FaceShifter', 'dfdtest': 'DFD-test', 'ffval': 'FF++ val', 'dfdval': 'DFD-val'}

b4 = np.load(os.path.join(EXP, 'cache', 'scorebank.npz'), allow_pickle=True)
b5 = np.load(os.path.join(EXP, 'cache', 'scorebank_t5.npz'), allow_pickle=True)
LAB = {s: b4[f'label|{s}'].astype(int) for s in REPORT + SELECT}
MODELS = {
    'P0': lambda s: [b4[f'score|{s}|P0']],
    'P3': lambda s: [b4[f'score|{s}|S4|P3|{k}|last'] for k in range(3)],
    '최종': lambda s: [b5[f'score|{s}|M1|tv|P3best+U3s|a0.50b0.75|{k}'] for k in range(3)],
}
out = {}


def auc_fast(y, s):
    """순위 기반 AUC(동점 무시 — 연속 점수라 무시해도 됨). 행렬곱 없음."""
    r = np.empty(len(s))
    r[np.argsort(s, kind='mergesort')] = np.arange(1, len(s) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# ── ⑤ 경계 적중 무작위 기준선 ─────────────────────────────────────────
z = np.load(os.path.join(EXP, 'cache', 'masks', 'test_loc.npz'), allow_pickle=True)
g = z['gt11'].astype(np.float32)
bnd = ((g >= 0.1) & (g <= 0.9)).reshape(len(g), -1).mean(1) * 100       # 무작위 칸 하나가 경계일 확률(프레임별)
out['random_boundary'] = {'all': float(bnd.mean())}
for m in np.unique(z['meth']):
    out['random_boundary'][str(m)] = float(bnd[z['meth'] == m].mean())

# ── ⑨ 운영점 ─────────────────────────────────────────────────────────
def tpr_at(y, s, f):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= f].max() * 100)


def ece(y, s, nb=10):
    s = np.clip(np.asarray(s, np.float64), 0, 1)
    e = 0.0
    for lo in np.linspace(0, 1, nb, endpoint=False):
        m = (s >= lo) & (s < lo + 1 / nb) if lo < 1 - 1 / nb else (s >= lo)
        if m.any():
            e += m.mean() * abs(s[m].mean() - y[m].mean())
    return float(e * 100)


op = {}
for mname, get in MODELS.items():
    rows = {}
    sel = [np.concatenate([get(s)[k][LAB[s] == 0] for s in SELECT]) for k in range(len(get('ffval')))]
    tau5 = [float(np.quantile(r, 0.95)) for r in sel]                        # 선택 세트 진짜 영상 오탐률 5%가 되는 기준값
    for s in REPORT:
        y = LAB[s]
        per = []
        for k, sc in enumerate(get(s)):
            sc = np.asarray(sc, np.float64)
            p5 = sc >= tau5[k]
            p05 = sc >= 0.5
            per.append([auc_fast(y, sc) * 100, tpr_at(y, sc, 0.01), tpr_at(y, sc, 0.05), tpr_at(y, sc, 0.10),
                        p5[y == 0].mean() * 100, p5[y == 1].mean() * 100,
                        p05[y == 0].mean() * 100, p05[y == 1].mean() * 100, ece(y, sc)])
        per = np.array(per).mean(0)
        rows[s] = dict(zip(['auc', 'tpr@fpr1', 'tpr@fpr5', 'tpr@fpr10', 'dep_fpr', 'dep_tpr', 'fpr05', 'tpr05', 'ece'],
                           [float(v) for v in per]))
    rows['tau5'] = tau5
    op[mname] = rows
out['operating_point'] = op

# ── ⑪ Holm 다중 비교 보정 (영상 짝 bootstrap, seed 평균 AUC 차이) ─────────────
B = 2000
rng = np.random.default_rng(0)
tests = []
for base in ('P0', 'P3'):
    for s in REPORT:
        y = LAB[s]
        new = [np.asarray(v, np.float64) for v in MODELS['최종'](s)]
        bas = [np.asarray(v, np.float64) for v in MODELS[base](s)]
        stat = lambda i: np.mean([auc_fast(y[i], v[i]) for v in new]) - np.mean([auc_fast(y[i], v[i]) for v in bas])
        pt = stat(np.arange(len(y))) * 100
        ds = []
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                ds.append(stat(i) * 100)
        ds = np.array(ds)
        p = max(2 * min((ds <= 0).mean(), (ds >= 0).mean()), 1 / len(ds))
        tests.append({'base': base, 'set': s, 'delta': float(pt), 'lo': float(np.quantile(ds, 0.025)),
                      'hi': float(np.quantile(ds, 0.975)), 'p': float(p)})
order = np.argsort([t['p'] for t in tests])
m = len(tests)
run = 0.0
for rank, j in enumerate(order):
    run = max(run, min(1.0, (m - rank) * tests[j]['p']))
    tests[j]['p_holm'] = float(run)
out['holm'] = tests

# ── ⑮ 병합 비율 민감도 그림 ───────────────────────────────────────────
sel = json.load(open(os.path.join(REP, 't5_merge_sel.json'), encoding='utf-8'))
A = [0.5, 0.75, 1.0, 1.25]
Bv = [0.25, 0.5, 0.75, 1.0]
grid = np.zeros((4, 4))
for i, a in enumerate(A):
    for j, b in enumerate(Bv):
        k = f'tv|P3best+U3s|a{a:.2f}b{b:.2f}'
        grid[i, j] = np.mean([(sel[sd][k]['ffval'] + sel[sd][k]['dfdval']) / 2 for sd in ('0', '1', '2')])
md = open(os.path.join(REP, 't5_merge.md'), encoding='utf-8').read()
sec = md.split('## 2.')[1].split('## 3.')[0]
tcurve = []
for line in sec.splitlines():
    mt = re.match(r'\|\s*([0-9.]+)[^|]*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|', line)
    if mt:
        tcurve.append([float(x) for x in mt.groups()])
tcurve = np.array(tcurve)
out['merge_grid_selection'] = {'a': A, 'b': Bv, 'score': grid.round(3).tolist()}

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={'width_ratios': [1, 1.35]})
im = ax[0].imshow(grid, cmap='Blues', vmin=grid.min() - 0.3, vmax=grid.max())
for i in range(4):
    for j in range(4):
        ax[0].text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center', fontsize=12,
                   color='white' if grid[i, j] > grid.max() - 0.35 else '#172133',
                   fontweight='bold' if (A[i], Bv[j]) == (0.5, 0.75) else 'normal')
ax[0].add_patch(plt.Rectangle((2 - 0.5, 0 - 0.5), 1, 1, fill=False, ec='#C9661E', lw=3))
ax[0].set_xticks(range(4), [f'{b}' for b in Bv])
ax[0].set_yticks(range(4), [f'{a}' for a in A])
ax[0].set_xlabel('b (SBI 학습분 비율)')
ax[0].set_ylabel('a (FF++ 학습분 비율)')
ax[0].set_title('task vector 병합: 선택 점수 (FF++ val·DFD-val 평균 AUC)\n주황 = 최종 모델(a 0.5, b 0.75) · 논문 모델 83.94', fontsize=11)
cols = {'fftest': '#5F6675', 'cdf': '#2A5DB8', 'fsh': '#C9661E', 'dfdtest': '#7A4FB5'}
p0 = {s: op['P0'][s]['auc'] for s in REPORT}
for c, s in zip([3, 4, 5, 6], REPORT):
    ax[1].plot(tcurve[:, 0], tcurve[:, c], '-o', color=cols[s], label=f'{NAME[s]}', ms=4)
    ax[1].axhline(p0[s], color=cols[s], ls=':', lw=1)
ax[1].axvline(0.8, color='#172133', ls='--', lw=1)
ax[1].text(0.805, ax[1].get_ylim()[0] + 1, '선택 t=0.8', fontsize=10)
ax[1].set_xlabel('t (0 = FF++ 전문가만, 1 = SBI 전문가만)')
ax[1].set_ylabel('AUC (seed 3개 평균)')
ax[1].set_title('선형 병합 비율 t에 따른 보고 세트 AUC (점선 = 논문 모델)', fontsize=11)
ax[1].legend(loc='lower left', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(REP, 'qa_merge_sensitivity.png'), dpi=150)

json.dump(out, open(os.path.join(REP, 'qa_followup.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])
