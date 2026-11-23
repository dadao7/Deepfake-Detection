"""TASKS5: 병합 비율 t에 따른 보고 세트 AUC 곡선 (seed 평균 ± 표준편차) + P3·모델 2개 결합 기준선.
  python code/t5_fig.py → reports/t5_fig_merge.png"""
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import common as C  # noqa: E402
import t34_common as T  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_models as MD  # noqa: E402

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
TS = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
NAMES = {'fftest': 'FF++ test (학습 분포)', 'cdf': 'Celeb-DF (처음 봄)', 'fsh': 'FaceShifter (처음 보는 방식)', 'dfdtest': 'DFD-test (처음 봄)'}


def main():
    lab, b5 = T5.labels(), T5.load_bank5()
    models = MD.collect()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, s in zip(axes, T.REPORT_SETS):
        a = np.array([[T.auc(lab[s], b5[f'score|{s}|M1|lin|P3best+U3s|t{t:.1f}|{sd}']) for sd in (0, 1, 2)] for t in TS])
        m, sd = a.mean(1), a.std(1)
        ax.fill_between(TS, m - sd, m + sd, color='#2a6fdb', alpha=0.15, lw=0)
        ax.plot(TS, m, color='#2a6fdb', lw=2, marker='o', ms=4, label='병합 (단일 모델)')
        for name, ls, txt in (('P3', '--', 'P3 (현재 모델)'), ('결합2', ':', '모델 2개 결합')):
            v = np.mean([T.auc(lab[s], d[s]) for d in models[name]['scores']])
            ax.axhline(v, color='#666666', ls=ls, lw=1.2, label=txt)
        tv = np.mean([T.auc(lab[s], d[s]) for d in models['병합TV']['scores']])
        ax.plot([0.8], [tv], marker='*', ms=13, color='#d9480f', ls='none', label='task vector 병합 a=0.5·b=0.75 (x 위치는 표시용)')
        ax.axvline(0.8, color='#bbbbbb', lw=1)
        ax.set_title(NAMES[s], fontsize=11)
        ax.set_xlabel('t (0 = FF++ 전문가 P3, 1 = SBI 전문가 U3s)')
        ax.grid(alpha=0.3)
        ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes[0].set_ylabel('영상 AUC (%)')
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('전문가 가중치 병합: 비율 t는 선택 세트(FF++ val·DFD-val)로만 골라 t=0.8 (세로선). 음영 = seed 3개 표준편차', fontsize=11)
    fig.tight_layout()
    out = os.path.join(C.REPORTS, 't5_fig_merge.png')
    fig.savefig(out, dpi=150)
    print('저장', out)


if __name__ == '__main__':
    main()
