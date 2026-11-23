"""질문 방어 ⑦: 사전 고정 선택 규칙 재검증 (2026-10-21, 사용자 승인: 규칙 = "FF++ val 비저하 + 선택 점수 최고").
새 seed 3·4에서 TASKS6의 두 후보 전문가를 같은 설정으로 다시 학습하고(run_train_sbi2.py --debug, registry 미기록),
seed 0~2에서 이미 고른 병합 변형을 그대로 만든다(변형·비율 재탐색 없음):
  기존 병합   : P3 best + U3s best, task vector a0.50 b0.75 (최종 모델)
  Xp3 병합    : P3 last + Xp3 best, task vector a0.75 b0.75 (TASKS6 3위)
  Xlr1 병합   : P3 best + Xlr1 best, 선형 t0.8 (TASKS6 선택 점수 1위)
규칙 A(옛 사전 규칙) = 선택 점수 최고, 규칙 B(이번 사전 규칙) = FF++ val이 기존 병합 이상인 후보 중 선택 점수 최고.
출력: cache/qa_rule_scores.npz (분석은 ref/qa_followup/qa_rule_stats.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import common as C  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_merge as M  # noqa: E402

OUT = os.path.join(C.CACHE, 'qa_rule_scores.npz')


def build(seed):
    s0 = M.init_state(seed)
    p3b, p3l = M.expert_state('P3', 'best', seed, s0), M.expert_state('P3', 'last', seed, s0)
    u3s = M.expert_state('U3s', 'best', seed, s0)
    xp3, xlr = M.expert_state('Xp3', 'best', seed, s0), M.expert_state('Xlr1', 'best', seed, s0)
    return [('base', M.combine(s0, p3b, u3s, 0.5, 0.75)),
            ('Xp3', M.combine(s0, p3l, xp3, 0.75, 0.75)),
            ('Xlr1', M.combine(s0, p3b, xlr, 0.2, 0.8))]           # 선형 t = 0.8 → (1−t, t)


def main():
    out = {}
    for seed in (3, 4):
        vs = build(seed)
        sc = T5.eval_models([M.model_from(sd, seed) for _, sd in vs], sets=T5.ALL)
        for j, (n, _) in enumerate(vs):
            for s in T5.ALL:
                out[f'score|{s}|{n}|{seed}'] = sc[s][j]
        np.savez(OUT, **out)
        print(f'seed {seed} 저장', flush=True)
    print('RULE_EVAL_OK', flush=True)


if __name__ == '__main__':
    main()
