#!/usr/bin/env bash
# c23 캐시·평가 후 두 head 구조 학습·평가 (2026-10-21)
set -e
cd /c/m2f2/exp
PY=/c/Users/COM/miniconda3/envs/m2f2/python.exe
$PY -X utf8 -u code/qa_c23.py base > logs/qa_c23_base.log 2>&1
$PY -X utf8 -u code/qa_c23.py eval > logs/qa_c23_eval.log 2>&1
echo "c23 done $(date)"
bash queues/qa_twohead.sh
