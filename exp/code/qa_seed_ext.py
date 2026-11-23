"""질문 방어 ⑩: seed 확대 (2026-10-21). 기존 코드는 고치지 않고 함수만 불러 쓴다.
새 seed(3, 4)의 P3(FF++ 전문가)·U3s(SBI 전문가)는 run_train*.py --no_test로 학습해 registry를 건드리지 않는다.
병합 비율은 seed 0~2에서 미리 고른 task vector a=0.50, b=0.75를 그대로 쓴다(새 seed = 사전 고정 규칙의 재현 검증).
  --sanity : seed 0 병합 모델을 다시 만들어 FF++ test·Celeb-DF 점수가 은행(scorebank_t5)과 같은지 확인
  --seeds 3 4 : 새 seed의 P3 last · U3s best · 병합 모델을 전체 세트로 평가 → cache/qa_seedext_scores.npz
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common as C  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_merge as M  # noqa: E402

OUT = os.path.join(C.CACHE, 'qa_seedext_scores.npz')


def build(seed):
    s0 = M.init_state(seed)
    p3b = M.expert_state('P3', 'best', seed, s0)
    p3l = M.expert_state('P3', 'last', seed, s0)
    u3s = M.expert_state('U3s', 'best', seed, s0)
    return [('merge', M.combine(s0, p3b, u3s, 0.5, 0.75)), ('P3last', p3l), ('U3sbest', u3s)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sanity', action='store_true')
    ap.add_argument('--seeds', type=int, nargs='*', default=[])
    a = ap.parse_args()
    if a.sanity:
        name, sd = build(0)[0]
        sc = T5.eval_models([M.model_from(sd, 0)], sets=['fftest', 'cdf'])
        b5 = T5.load_bank5()
        for s in ('fftest', 'cdf'):
            ref = b5[f'score|{s}|M1|tv|P3best+U3s|a0.50b0.75|0']
            d = float(np.abs(sc[s][0] - ref).max())
            print(f'sanity {s}: 최대 차이 {d:.2e}', flush=True)
            assert d < 1e-4, f'재현 실패 {s} {d}'
        print('SANITY_OK', flush=True)
    if a.seeds:
        old = dict(np.load(OUT, allow_pickle=True)) if os.path.exists(OUT) else {}
        for seed in a.seeds:
            vs = build(seed)
            sc = T5.eval_models([M.model_from(sd, seed) for _, sd in vs], sets=T5.ALL)
            for j, (n, _) in enumerate(vs):
                for s in T5.ALL:
                    old[f'score|{s}|{n}|{seed}'] = sc[s][j]
            np.savez(OUT, **old)
            print(f'seed {seed} 평가 저장 ({len(old)}키)', flush=True)
        print('EVAL_OK', flush=True)


if __name__ == '__main__':
    main()
