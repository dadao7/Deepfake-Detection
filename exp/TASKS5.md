# P3를 넘는 개선 시도 + AUC 외 가치 설명 (TASKS5)

> 2026-10-17 사용자 요청: "P3보다 좋은 성능이 나오도록 다른 방법을 더 추가 시도 및 테스트. 실제로 ACC와 AUC가 향상했으면 좋겠음. ACC·AUC 외에도 가치 있는 성능 향상(위치 적중 등)을 자세하게 설명."
> **이어하기: 이 파일의 "현재 위치"부터 진행한다.** 단계가 끝날 때마다 이 파일을 먼저 갱신한다.
> 규칙은 TASKS4와 같다: 원본 코드 수정 금지, 새 코드·결과는 `C:\m2f2\exp\` 아래, **선택은 FF++ val + DFD-val로만**, 보고 세트(FF++ test, Celeb-DF, FaceShifter, DFD-test)는 고르는 데 쓰지 않는다.

## 목표와 판정
- 비교 기준: **P3(last, 현재 모델)** 과 P0. 지표 = 영상 AUC, ACC(기준값 0.5), 균형 ACC(bACC). 각 보고 세트에서 P3 대비 짝 bootstrap 95% CI.
- "단일 모델"(추론 비용 P3와 같음)과 "여러 모델 결합"(비용 증가)은 따로 표시한다.

## 시도할 방법 (근거 → 방법)

| # | 근거(이전 실험) | 방법 | 학습 | 산출 |
|---|---|---|---|---|
| M1 | P3·U3f·U3s는 seed가 같으면 **초기값이 같다**(저자 baseline 상단 + SBI 분기 초기화). FF++ 전문가와 SBI 전문가가 상보적(T4 ④) | **전문가 가중치 병합**: θ(t) = (1−t)θ_FF + tθ_SBI, 그리고 task vector 합 θ0 + a·τ_FF + b·τ_SBI. 단일 모델로 결합 효과를 얻는지 | 없음 | `code/t5_merge.py`, `reports/t5_merge.md` |
| M2 | CDF ACC 53%(P0)는 기준값 0.5가 도메인 밖에서 어긋난 탓(미탐 68%). AUC는 높은데 ACC가 낮음 | **선택 세트 기준값 보정**: FF++ val·DFD-val 평균 bACC가 최대인 기준값을 고정해 보고 세트에 적용 | 없음 | `code/t5_calib.py`, `reports/t5_calib.md` |
| M3 | 혼합 학습 단일 모델은 상보성을 잃음(T4 ⑥). 결합은 이득 | **상보 전문가 증류**: 학생(U3 구조)을 FF++ + SBI로 학습, 교사 = P3·U3s 결합 logit에 KD. (M1이 좋으면 병합 모델에서 시작) | seed 3 | `code/run_train_distill.py` |
| M5 | 일반 기법(P0에도 적용됨) | **좌우 반전 TTA**: 평가 세트 반전 캐시(`code/build_flip_eval_cache.py`) → 프레임 확률 평균(`t5_common.eval_models(tta=True)`). P0+TTA·P3+TTA·최고+TTA를 함께 보고 | 없음 | 캐시 약 21만 프레임 ≈ 35분 |
| M4 | M1~M5 중 좋은 것 | 최종 조합(단일 모델 최고 / 결합 최고) | — | `reports/t5_final.md` |
| D | 사용자 요청 | **AUC·ACC 외 가치**: 위치 적중·셀 AUROC·crop 적중·과적합 방지·seed 안정성·오탐/미탐 균형·위조 방식별 성능 설명 | — | `reports/t5_beyond_auc.md` |

## 단계와 상태
| 단계 | 내용 | 상태 | 결과 |
|---|---|---|---|
| B0 | 평가 공통 코드(`code/t5_common.py`) + 기존 점수 재현 확인 | **완료** | `code/t5_check.py`: P3_s0 last·U3s_s0 best × 6세트 영상 점수가 TASKS4 은행과 비트 단위 일치(최대 차 0). 모델 2개 전체 평가 1.7분. 비교 모델 모음 `code/t5_models.py` |
| M1 | 전문가 가중치 병합 | **완료(향상)** | `reports/t5_merge.md`. 선택 1위(전체 변형) = task vector `tv|P3best+U3s|a0.50b0.75`(85.581), 선형 1위 `lin|P3best+U3s|t0.8`(85.576). TV 병합 P3 대비 AUC: FF++ −0.16, **CDF +2.60 ✔, FSH +4.12 ✔, DFD-test +5.25 ✔**(단일 모델, 추론 비용 1). ACC@0.5는 FF++·DFD에서 하락(진짜 쪽 판정 증가) → 선택 세트 기준값 보정 시 4세트 bACC 모두 상승. **seed 수프**(`code/t5_soup.py`): soup-lin 선택 85.74(전체 1위, 단일 모델), soup-tv 85.60. **대조 완료**: 병합TV − WiSE대조(θ0+0.7τ_U3s) = FF++ +1.17 ✔·CDF +4.69 ✔·FSH +2.49 ✔·DFD +0.06 → FF++ 전문가 task vector가 실제로 기여. 위치 지표 유지(경계 적중 35.1~35.3%, `reports/t5_loc.json`) |
| M2 | 기준값 보정 | **완료(효과 없음)** | `code/t5_calib.py`, `reports/t5_calib.md`. FF++ val·DFD-val bACC 최대 기준값이 P0·P3에서 0.63~0.82로 올라가 CDF ACC가 크게 떨어짐(P3 51.7→40.5). DFD는 오탐이 많아 기준값을 올려야 하고 CDF는 미탐이 많아 내려야 함 → 도메인마다 방향이 반대라 하나의 기준값으로는 못 고침(음성 결과). 참고: DFD-test·FF++ test는 가짜 85%·80%라 ACC보다 bACC가 공정 |
| M3 | 전문가 증류 | **완료(병합보다 약함)** | 3seed, 선택 세트로 last 선택(84.51). P3 대비 AUC: FF++ −0.42, CDF +1.34, FSH +1.75, DFD-test +3.54 (병합TV는 −0.16/+2.60/+4.12/+5.25). `reports/t5_distill_choice.json` |
| M4 | 최종 조합 | **완료** | `code/t5_final.py` → `reports/t5_final_tables.md`. 선택 1위 병합+TTA(85.89, 비용 2), 단일 모델 1위 수프(lin)(85.74). 보정 τ 적용 시 CDF ACC +3.1~5.6 ✔, DFD bACC +5.6~7.2 ✔, FF++ ACC −0.7~−0.9(불분명), DFD ACC 하락(가짜 85% 세트, P3도 전부-가짜 기준선 85.1%보다 낮음) |
| M5 | 반전 TTA | **완료(작은 향상, 불분명)** | 모든 모델에서 ΔAUC +0.0~+0.5, CI 0 포함. 병합TV+TTA: P3 대비 CDF +2.89 ✔·FSH +4.42 ✔·DFD +5.38 ✔. 캐시 74.6분(게이트 비트 일치) | `code/t5_tta.py`(P0·P3·병합·증류에 같은 TTA). 점검: P0 껍데기 = 은행 점수, P0+TTA ffval 94.85→95.11, fsh 68.01→68.00 |
| D | AUC 외 가치 문서 | **완료** | `reports/t5_beyond_auc.md`(서술, §7 병합 모델) + `t5_beyond_tables.md`(자동) |
| Z | 정리·보고, 메모리 갱신 | **완료** | `reports/t5_final.md`, `paper/draft_v2.md`, 발표·가치 문서 갱신, PROTOCOL §18, PROGRESS, 메모리, 원본 해시 |

## 현재 위치
- **TASKS5 완료(2026-10-17).** 진행 중인 작업 없음.
- 핵심 결과: 학습 없는 전문가 가중치 병합(task vector, P3best + U3s, a=0.5·b=0.75) 단일 모델이 P3 대비 Celeb-DF +2.60·FaceShifter +4.12·DFD +5.25 AUC(CI 0 밖), FF++ −0.16. bACC DFD +6.0 ✔, 보정 기준값 시 Celeb-DF ACC +3~5.6 ✔. 위치 설명 유지.
- 결과물: `reports/t5_final.md`(서술) · `t5_final_tables.md`(자동) · `t5_merge.md` · `t5_fig_merge.png` · `t5_calib.md` · `t5_beyond_auc.md` · `t5_beyond_tables.md`, 논문 `paper/draft_v2.md`
- 마지막 갱신: 2026-10-17

## 진행 로그 (최신이 위)
| 시각 | 한 일 | 결과 |
|---|---|---|
| 2026-10-17 | Z: 최종 보고서·논문 v2·발표/가치/D 문서 갱신, PROGRESS·메모리·원본 해시 | 완료 |
| 2026-10-17 | M3 증류 3seed 완료·평가, WiSE 대조·TTA·수프·위치 평가 완료, PROTOCOL §18, 최종 표·그림(`t5_fig_merge.png`), `reports/t5_final.md` 초안 | 병합이 최고(단일 모델) |
| 2026-10-17 | M1 완료(병합 단일 모델이 P3 대비 CDF·FSH·DFD AUC 확실한 향상), 수프 시험, 증류 s0 평가, 보정 재실행(병합 모델은 보정 시 ACC·bACC 4세트 상승) | 향상 |
| 2026-10-17 | M1 1차 실행 중단: task vector 외삽에서 BN running_var 음수 → NaN. BN 통계는 볼록 결합으로, NaN 변형은 순위 제외로 고쳐 재실행. 점검: t=0 = P3, t=1 = U3s(AUC 동일). 반전 TTA 캐시 생성 시작(`logs/build_flip_eval_cache.log`, 재실행 시 이어감) | — |
| 2026-10-17 | B0·M2 완료, M1·M3 시작 | M2 음성(도메인마다 기준값 이동 방향 반대) |
| 2026-10-17 | 계획 수립 | — |
