"""TASKS5 M5: 좌우 반전 TTA. 프레임마다 원본·반전 확률 평균 → 같은 영상 점수 규칙. P0에도 똑같이 적용해 공정 비교.
대상: P0, P3(last, seed 3), M1 선택 병합(seed 3), M3 증류(있으면, 선택 세트로 고른 ckpt). 결과: scorebank_t5 키 score|{set}|TTA|{이름}|{seed}
  python code/t5_tta.py"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import common as C  # noqa: E402
import eval_newdata as EV  # noqa: E402
import run_train_unfreeze as U  # noqa: E402
import t34_common as T  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_merge as M  # noqa: E402

SEEDS = (0, 1, 2)


class P0Model(nn.Module):
    """캐시된 저자 baseline logit을 그대로 내는 껍데기 (ExpModel과 같은 호출 방식)"""
    k = 1

    def forward(self, x, b3, lg):
        return {'pred': lg}


def targets():
    out = [('P0', 0, lambda: P0Model().cuda())]
    for sd in SEEDS:
        out.append(('P3', sd, lambda sd=sd: EV.build(('S4', 'P3', sd, 'last', os.path.join(C.RUNS, f'P3_d100_s{sd}', 'ckpt_last.pt')))))
    ch = os.path.join(C.REPORTS, 't5_merge_choice.json')
    if os.path.exists(ch):
        chj = json.load(open(ch))
        for tag, key in (('병합', 'best_lin'), ('병합TV', 'best_tv')):
            name = chj[key][1]                        # lin|{쌍}|t{t} 또는 tv|{쌍}|a{a}b{b}
            pname, par = name.split('|')[1], name.split('|')[2]
            if name.startswith('lin|'):
                wa, wb = 1 - float(par[1:]), float(par[1:])
            else:
                wa, wb = map(float, par[1:].split('b'))
            for sd in SEEDS:
                def mk(sd=sd, pname=pname, wa=wa, wb=wb):
                    s0 = M.init_state(sd)
                    A, B = M.PAIRS[pname]
                    return M.model_from(M.combine(s0, M.expert_state(*A, sd, s0), M.expert_state(*B, sd, s0), wa, wb), sd)
                out.append((tag, sd, mk))
    dch = os.path.join(C.REPORTS, 't5_distill_choice.json')
    if os.path.exists(dch):
        d = json.load(open(dch))                      # {"setting": "Ds_b1", "ckpt": "best"}
        for sd in SEEDS:
            def mk(sd=sd):
                m = U.UModel('U3', sd)
                m.load_state_dict(torch.load(os.path.join(C.RUNS, f'{d["setting"]}_s{sd}', f'ckpt_{d["ckpt"]}.pt'),
                                             map_location='cpu', weights_only=False)['model'])
                return m.cuda().eval()
            out.append(('증류', sd, mk))
    return out


def main():
    b5 = T5.load_bank5()
    todo = [t for t in targets() if f'score|fftest|TTA|{t[0]}|{t[1]}' not in b5]
    print('TTA 평가', [(n, s) for n, s, _ in todo], flush=True)
    if todo:
        t0 = time.time()
        models = [mk() for _, _, mk in todo]
        sc = T5.eval_models(models, tta=True)
        add = {f'score|{s}|TTA|{n}|{sd}': sc[s][j] for j, (n, sd, _) in enumerate(todo) for s in T5.ALL}
        print('은행 항목', T5.save_bank5(add), f'{(time.time() - t0) / 60:.1f}분', flush=True)


if __name__ == '__main__':
    main()
