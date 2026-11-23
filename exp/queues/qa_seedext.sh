#!/usr/bin/env bash
# 질문 방어 ⑩ seed 확대 대기열 (2026-10-21). 학습은 --no_test(registry 미기록), 평가는 qa_seed_ext.py.
# 중단되면 다시 실행: 각 학습은 runs/{id}/resume.pt에서 이어받는다.
set -e
cd /c/m2f2/exp
PY=/c/Users/COM/miniconda3/envs/m2f2/python.exe
for s in 3 4; do
  $PY -X utf8 -u code/run_train.py --setting P3 --data 100 --seed $s --no_test > logs/qa_seedext_P3_s$s.log 2>&1
  echo "P3 seed $s done $(date)"
done
for s in 3 4; do
  $PY -X utf8 -u code/run_train_unfreeze.py --cfg U3 --data sbi --seed $s --no_test > logs/qa_seedext_U3s_s$s.log 2>&1
  echo "U3s seed $s done $(date)"
done
$PY -X utf8 -u code/qa_seed_ext.py --seeds 3 4 > logs/qa_seedext_eval.log 2>&1
echo "EVAL done $(date)"
