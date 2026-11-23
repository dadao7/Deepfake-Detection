#!/usr/bin/env bash
# 질문 방어 ② 두 head 구조 학습·평가 대기열 (2026-10-21). 학습 설정은 P3와 같음, --no_test(registry 미기록).
set -e
cd /c/m2f2/exp
PY=/c/Users/COM/miniconda3/envs/m2f2/python.exe
for s in 0 1 2; do
  $PY -X utf8 -u code/qa_twohead.py train --seed $s > logs/qa_twohead_s$s.log 2>&1
  echo "TH seed $s done $(date)"
done
$PY -X utf8 -u code/qa_twohead.py eval > logs/qa_twohead_eval.log 2>&1
echo "EVAL done $(date)"
