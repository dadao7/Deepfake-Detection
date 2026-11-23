"""TASKS5 M3: 증류 학생 모델(runs/D*_s*)의 best·last를 모든 세트에서 평가해 scorebank_t5에 추가.
키: score|{set}|M3|{설정}-{ckpt}|{seed}.  python code/t5_eval_distill.py"""
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common as C  # noqa: E402
import run_train_unfreeze as U  # noqa: E402
import t34_common as T  # noqa: E402
import t5_common as T5  # noqa: E402


def main():
    b5 = T5.load_bank5()
    todo = []
    for rd in sorted(glob.glob(os.path.join(C.RUNS, 'D*_s*'))):
        if not os.path.exists(os.path.join(rd, 'DONE')):
            continue
        st, seed = os.path.basename(rd).rsplit('_s', 1)
        for ck in ('best', 'last'):
            key = f'M3|{st}-{ck}|{seed}'
            if f'score|fftest|{key}' not in b5:
                todo.append((key, os.path.join(rd, f'ckpt_{ck}.pt'), int(seed)))
    print('평가할 체크포인트', len(todo), flush=True)
    if not todo:
        return
    models = []
    for _, path, seed in todo:
        m = U.UModel('U3', seed)
        m.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
        models.append(m.cuda().eval())
    sc = T5.eval_models(models)
    add = {f'score|{s}|{key}': sc[s][j] for j, (key, _, _) in enumerate(todo) for s in T5.ALL}
    print('은행 항목', T5.save_bank5(add), flush=True)
    lab = T5.labels()
    for j, (key, _, _) in enumerate(todo):
        print(key, ' | '.join(f'{s} {T.auc(lab[s], sc[s][j]):.2f}' for s in T5.ALL), flush=True)


if __name__ == '__main__':
    main()
