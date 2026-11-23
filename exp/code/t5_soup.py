"""TASKS5 M1′: seed 수프. seed 3개의 병합 모델 가중치를 평균한 단일 모델(추론 비용 1).
근거: seed가 달라도 baseline 상단(저자 가중치)·heatmap head·detail branch(SBI 사전학습)는 같은 값에서 출발한다.
seed마다 다른 것은 gating network(16차원 MLP) 초기값뿐이다. 평균이 깨지는지(같은 basin인지)는 선택 세트로 판단한다.
대상: M1 선택 모델 2개(best_lin, best_tv)의 seed 수프. 결과: scorebank_t5 키 score|{set}|M1|soup-{lin|tv}|0
  python code/t5_soup.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common as C  # noqa: E402
import t34_common as T  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_merge as M  # noqa: E402

SEEDS = (0, 1, 2)


def merged_state(name, sd):
    pname, par = name.split('|')[1], name.split('|')[2]
    wa, wb = (1 - float(par[1:]), float(par[1:])) if name.startswith('lin|') else tuple(map(float, par[1:].split('b')))
    s0 = M.init_state(sd)
    A, B = M.PAIRS[pname]
    return M.combine(s0, M.expert_state(*A, sd, s0), M.expert_state(*B, sd, s0), wa, wb)


def soup(name):
    sds = [merged_state(name, sd) for sd in SEEDS]
    out = {}
    for k, v in sds[0].items():
        out[k] = v if not torch.is_floating_point(v) else (sum(d[k].float() for d in sds) / len(sds)).to(v.dtype)
    return out


def main():
    ch = json.load(open(os.path.join(C.REPORTS, 't5_merge_choice.json')))
    b5 = T5.load_bank5()
    todo = [(f'soup-{tag}', ch[key][1]) for tag, key in (('lin', 'best_lin'), ('tv', 'best_tv'))
            if f'score|fftest|M1|soup-{tag}|0' not in b5]
    if todo:
        models = [M.model_from(soup(n), 0) for _, n in todo]
        sc = T5.eval_models(models, verbose=False)
        T5.save_bank5({f'score|{s}|M1|{k}|0': sc[s][j] for j, (k, _) in enumerate(todo) for s in T5.ALL})
        b5 = T5.load_bank5()
    lab = T5.labels()
    print('모델 | 선택 | ' + ' | '.join(T5.ALL))
    for tag, key in (('lin', 'best_lin'), ('tv', 'best_tv')):
        name = ch[key][1]
        per = [[T.auc(lab[s], b5[f'score|{s}|M1|{name}|{sd}']) for s in T5.ALL] for sd in SEEDS]
        m = np.mean(per, 0)
        print(f'{name} (seed 평균) | {np.mean(m[:2]):.2f} | ' + ' | '.join(f'{x:.2f}' for x in m))
        a = [T.auc(lab[s], b5[f'score|{s}|M1|soup-{tag}|0']) for s in T5.ALL]
        print(f'soup-{tag} | {np.mean(a[:2]):.2f} | ' + ' | '.join(f'{x:.2f}' for x in a))


if __name__ == '__main__':
    main()
