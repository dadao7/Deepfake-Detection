#!/usr/bin/env bash
# 질문 방어 ⑦ 사전 고정 규칙 재검증 대기열 (2026-10-21). --debug: 테스트 평가·registry 생략(체크포인트는 저장).
set -e
cd /c/m2f2/exp
PY=/c/Users/COM/miniconda3/envs/m2f2/python.exe
for s in 3 4; do
  $PY -X utf8 -u code/run_train_sbi2.py --tag p3 --init p3last --seed $s --debug > logs/qa_rule_Xp3_s$s.log 2>&1
  echo "Xp3 seed $s done $(date)"
  $PY -X utf8 -u code/run_train_sbi2.py --tag lr1 --init theta0 --lr 1e-5 --seed $s --debug > logs/qa_rule_Xlr1_s$s.log 2>&1
  echo "Xlr1 seed $s done $(date)"
done
$PY -X utf8 -u code/qa_rule_check.py > logs/qa_rule_eval.log 2>&1
echo "EVAL done $(date)"
