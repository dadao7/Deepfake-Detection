"""TASKS5 B0: t5_common.eval_models가 TASKS4 점수 은행과 같은 영상 점수를 내는지 확인 (P3_s0 last, U3s_s0 best)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import eval_newdata as EV  # noqa: E402
import t34_common as T  # noqa: E402
import t5_common as T5  # noqa: E402

sps = [('S4', 'P3', 0, 'last', os.path.join(T5.C.RUNS, 'P3_d100_s0', 'ckpt_last.pt')),
       ('S9', 'U3s', 0, 'best', os.path.join(T5.C.RUNS, 'U3s_s0', 'ckpt_best.pt'))]
models = [EV.build(sp) for sp in sps]
t0 = time.time()
res = T5.eval_models(models)
bank = T.load_bank()
for j, sp in enumerate(sps):
    k = f'{sp[0]}|{sp[1]}|{sp[2]}|{sp[3]}'
    for s in T5.ALL:
        ref = bank[f'score|{s}|{k}']
        new = res[s][j]
        print(k, s, len(ref), len(new), f'max|Δ| {np.nanmax(np.abs(ref - new)):.2e}',
              f'AUC ref {T.auc(bank["label|" + s], ref):.3f} new {T.auc(bank["label|" + s], new):.3f}')
print(f'{(time.time() - t0) / 60:.1f}분')
