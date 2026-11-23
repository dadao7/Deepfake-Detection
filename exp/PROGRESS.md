# 진행 기록 (PROGRESS)

> 새 세션/이어하기: 이 파일의 **현재 위치**부터 이어간다. 단계가 끝날 때, 중요한 일이 있을 때마다 이 파일을 먼저 갱신한다.
> 설계·고정 변수·설정 정의는 `HANDOFF.md`(§5)가 기준이다.
> **처음 보는 사람용 흐름 정리(단계별 구조 그림·결과·읽는 순서 + 문서 복사본): `PROJECT_FLOW/00_프로젝트_흐름_안내.md`**. 전체 파일 지도: `GUIDE.md`. 최종 발표 개요: `paper/final_presentation_outline.md`.

## 현재 위치
- **단계:** S0~S10 완료 + TASKS4(요청 4가지) 완료 + TASKS5(P3를 넘는 개선) 완료 + **TASKS6(추가 향상 탐색 8가지) 완료**. 진행 중인 작업 없음. 단계·결과는 `TASKS4.md`·`TASKS5.md`·`TASKS6.md`.
- **최종 산출물:** `reports/S7_report.md`(S7~S10 통합: §10 S8, §11 S9, §12 S10), 자동 생성 표 `s7_tables.md`·`s8_tables.md`·`s9_tables.md`·`s10_tables.md`·`s10_bootstrap.md`, 그림 `s7_fig_*`(4)·`s8_fig_*`(2)·`s9_fig_compare.png`·`s10_fig_compare.png`, `registry.csv`(P0 + 시범 2 + 본실험 51 + S8 9 + S9 12 = 75줄), HANDOFF §7.1~7.3
- **핵심 결론:** ① detach는 heatmap 위치와 교차 데이터셋 일반화를 함께 지킨다(제거 시 위치 붕괴 + Celeb-DF·DFD −5~7, CI 0 밖) ② 분기·baseline 미세조정 어느 쪽도 P0를 일관되게 못 넘고, 일반화 효과는 데이터셋마다 반대 ③ 현재 모델 P3는 어디서도 P0보다 나쁘지 않음
- **TASKS4 결과:** ① 논문 초안 `paper/draft_v1.md`·가치 평가 `paper/value_assessment.md` ② 발표 분석 `paper/presentation_highlights.md`(실제로 오른 수치만) ③ 단일 최고 = P3(last), 결합 최고 = P3×0.3 + U3s×0.7 → DFD-test +4.82·FaceShifter +5.80(CI 0 밖), FF++·CDF 손해 없음 ④ `reports/t4_improvements.md`: 상보 전문가 결합 향상, anchor 학습은 P4a 손해 완전 회복(위치 붕괴가 원인), AdaBN·경계 증거·GMM 기준값·혼합 학습은 효과 없음
- **TASKS5 결과:** 학습 없는 **전문가 가중치 병합** 단일 모델(θ0 + 0.5·τ_P3best + 0.75·τ_U3s)이 P3 대비 Celeb-DF +2.60·FaceShifter +4.12·DFD-test +5.25 AUC(CI 0 밖), FF++ −0.16(유지), 추론 비용 동일, 위치 설명 유지. 대조: WiSE형 대비 Celeb-DF +4.69 ✔(FF++ 전문가 기여), 혼합 학습 대비 FaceShifter +8.60 ✔, 증류 대비 FaceShifter +2.37 ✔, 출력 결합(2개) 대비 Celeb-DF +1.55 ✔. 기준값 보정(P3에는 역효과), TTA(+0~0.5)는 작음. `reports/t5_final.md`, 논문 `paper/draft_v2.md`
- **TASKS6 결과:** 추가 향상 1건 = Xp3 병합(P3에서 출발해 SBI 추가 학습한 전문가 + P3): 기존 병합 대비 FaceShifter +1.92 ✔, 나머지 동등(사후 규칙 "FF++ val 비저하"로 선택). 미리 정한 규칙의 1위(Xlr1)는 선택 세트 과적합(기존 병합 대비 Celeb-DF −5.4 ✔). 병합 이득은 주로 baseline 상단. `reports/t6_final.md`
- **최종 모델(사용자 결정 2026-10-17, A안):** TASKS5 task vector 병합 단일 모델. Xp3 병합은 보조 결과.
- **다음 할 일:** 사용자 결정 대기
- **환경 주의:** ① numpy BLAS(MKL 멀티스레드 계층) 깨짐 → numpy를 불러오기 전에 `MKL_THREADING_LAYER=SEQUENTIAL`(분석 스크립트에 내장). ② PowerShell `Get-Content`/`Set-Content`로 한글 .py를 고치면 인코딩이 깨진다 → Python(utf-8)이나 Edit 도구로.
- 마지막 갱신: 2026-10-17 (TASKS5)

## 단계표
| ID | 내용 | 상태 | 완료일 | 산출물 | 결과 한 줄 |
|---|---|---|---|---|---|
| 준비 | 논문 모델 평가(P0), 데이터 점검, FF++ 마스크·위조 c40(test+val) 다운로드 | **완료** | 2026-10-14 | `ref/p0/`, `data/FFPP_dl/` | P0 영상 AUC 95.22, 마스크·영상 2,240개 검증 통과 |
| S0 | 뼈대 + PROTOCOL.md + registry.csv + 원본 해시 대조 | **완료** | 2026-10-15 | `PROTOCOL.md`, `registry.csv`(51열), `logs/original_hash_check.txt` | 원본 23/23 일치, GPU 비어 있음, 고정 변수·설정 정의 전부 기록 |
| S1 | 프레임 인덱스·이미지 캐시, baseline 출력 캐시 | **완료** | 2026-10-15 | `cache/`(29GB, cache_sha 6fd7da8a23ec), `reports/p0_cache_metrics.json`, registry `P0_ref` | 캐시로 P0 영상 AUC 95.219(기준 95.217, +0.003), 프레임 확률 평균 차 0.00066. Celeb-DF P0 AUC 76.15 |
| S2 | ExpModel + 학습·평가·기록 루프 + 동일성 검증 + 시범(P3 캐시 vs 원본) | **완료** | 2026-10-15 | `code/expmodel.py`, `run_train.py`, `verify_s2.py`, `reports/s2_pilot_compare.json` | 게이트 ①~④ 비트 단위 통과, 시범 차이(캐시−원본) FF++ AUC +0.17/+0.09, CDF −0.78/−0.19, epoch 20 확정 |
| S3 | SBI 생성 + 사전학습 + seg head 복사본 | **완료** | 2026-10-15 | `cache/sbi/`, `runs/sbi_pretrain/{sbi_best,seg_head}.pt`, `reports/s3_*` | SBI val: detail 정확도 82.7%, heatmap 진짜/가짜 AUC 84.0, peak 적중 20.9%(초기 4.0, 중앙 0), 1학기 SBI는 같은 지표에서 50.2/50.2/10.6 |
| S4 | 100% 분류 실험 27회 (P1 스모크 먼저) | **완료** | 2026-10-15 | `runs/*_d100_s*`, `reports/s7_tables.md` | registry 27줄·NaN 없음. best FF++ AUC 전부 P0±0.35(94.32~95.13), P4a CDF 70.44±2.81로 저하, P0→P3 차이 불분명 |
| S5 | 데이터 양별 24회 | **완료** | 2026-10-15 | `runs/*_d25_s*`, `runs/*_d50_s*` | registry 24줄, 곡선 NaN 없음. P4a가 25·50%에서 뚜렷이 낮음(FF++ AUC best 93.0/93.8, CDF 71.0/73.6) |
| S6 | 마스크 정렬 + 위치 평가 | **완료** | 2026-10-15 | `code/s6_*.py`, `loc_eval.py`, `cache/masks/`, `reports/s6_*` | 정렬 test 97.7%(NCC 1.000), 임계값 val 결정, seg head·P3 경계 적중 35%, **P4a 4.8%(중앙 붕괴)**, 중앙 기준선이 안쪽 지표 포화 |
| S9 | 후속: baseline 일부 풀기(U0·U3 × FF++/SBI × 3seed) | **완료** | 2026-10-17 | `code/basetop.py`, `run_train_unfreeze.py`, `reports/s9_*` | 풀어도 P0를 못 넘음: U0f FF++ ±0·CDF −2.1(last), U0s FF++ −4.4·CDF −10.6, P3→U3f CDF −2.4. U3s에서 분기가 baseline을 보호(baseline만 CDF 76.5 vs U0s 65.6) |
| S10 | 후속: DFD·FaceShifter 교차 평가 | **완료** | 2026-10-17 | `code/face_align.py`, `build_newdata_cache.py`, `eval_newdata.py`, `s10_bootstrap.py`, `reports/s10_*` | 재현 게이트 통과(Δ −0.07%p). detach 제거 CDF·DFD −5~7(CI 0 밖), 일반화 효과는 데이터셋마다 반대, P3는 어디서도 P0 이상. DFDC 제외(Kaggle 인증 없음) |
| S8 | 후속: SBI 학습 프로토콜(P2·P3·P4a × 3seed) | **완료** | 2026-10-16 | `code/run_train_sbi.py`, `analyze_s8.py`, `reports/s8_*`, `S7_report.md` §10 | 가설 반증: SBI 학습은 P0보다 전부 저하(FF++ −3~4, CDF −7~14.5). 동적결합+M4 안전장치 효과는 재확인 |
| S7 | 분석·보고서·논문 가치 평가 | **완료** | 2026-10-16 | `reports/S7_report.md`, `s7_tables.md`, `s7_fig_*.png`, HANDOFF §7.1 | 덧붙인 기능 중 AUC 향상 없음(P3≈P0), detach 제거는 CDF −6~7·heatmap 중앙 붕괴, seg head가 위치 설명 유지, 평가 함정 4종 정량화 |

상태 표기: 대기 / 진행 / 완료 / 막힘. 막힘은 사유를 "결과 한 줄"에 적는다.

## 실행 현황 (S4·S5 시작 후 채움)
| 설정 | seed | 데이터 % | 상태 | registry 줄 | 비고 |
|---|---|---|---|---|---|
| (아직 없음) | | | | | |

## 실행 로그 (최신이 위)
| 날짜 | 한 일 | 결과 | 다음 |
|---|---|---|---|
| 2026-10-21 | 질문 방어용 추가 분석·실험 15개 항목(최종발표 리뷰에서 도출) 완료. 종합 `reports/qa_followup.md`, 답변 `reports/qa_answers.md`. 새 학습: P3·U3s seed 3·4, Xp3·Xlr1 seed 3·4, 두 head TH seed 0~2 (모두 --no_test/--debug, registry 미기록) | 5 seed 최종 vs P0 CDF +2.81·FSH +4.98·DFD +4.86(Holm 8개 후 ≤0.044, 5/5 seed). 사전 고정 규칙 B가 새 seed에서 Xp3 재선택(FSH +1.39 ✔). c23 test에서 최종 − P0 +1.37 ✔. Celeb-DF 재압축으로 SBI 손해 −10.6 → −2.4(변화 +8.2 ✔). SNS 모사 열화에서 향상은 DFD만 유지. 두 head 구조는 P3보다 나쁨 → detach 유지 지지. 추론 +3.4%(배치 32) | 발표 |
| 2026-10-17 | TASKS6 완료: 학습 없는 병합 확장(N1 모듈별·N2 TIES/DARE·N3 다중 전문가·N1b 하이브리드·N4 집계), SBI 전문가 개선 학습 7회(Xp3×3·Xema·Xlr1×3, kind=sbi2), PROTOCOL §19 | Xp3 병합 = 손해 없는 추가 향상(FSH +1.9 ✔). 선택 세트 과적합 발견 | 사용자 결정 |
| 2026-10-17 | TASKS5 완료: 평가 공통 코드(은행 비트 재현), 전문가 가중치 병합(243 변형 선택 세트 격자 + WiSE 대조 + seed 수프), 기준값 보정, 증류 학습 3회(registry kind=distill), 평가 세트 반전 캐시(약 21만 프레임) + TTA, 위치 평가, PROTOCOL §18, 원본 해시 | 병합 단일 모델이 P3 대비 교차 데이터셋 3곳 AUC 확실한 향상 | 사용자 결정 대기 |
| 2026-10-17 | 사용자 요청 4가지(TASKS4) 완료: DFD 신원 분할·점수 은행, T4 개선 7종(학습 11회: 혼합 6 + anchor 5), 논문 초안·가치 평가·발표 분석·최고 조합 문서, PROTOCOL §17, 원본 해시 23/23 | 확실한 향상은 상보 전문가 결합(DFD-test +4.8, FSH +5.8)뿐. anchor는 원인 규명(위치 보존 ↔ 일반화). 나머지 음성 | 사용자 결정 대기 |
| 2026-10-17 | S10 완료: 전체 평가(97개 체크포인트 × ffv·fsh·dfd, `reports/s10_eval.csv` 291줄) + 영상 단위 짝 bootstrap 95% CI(`s10_bootstrap.md`) + `analyze_s9_s10.py` 표·그림, 보고서 §11·§12·평가 함정 ⑤ 추가, HANDOFF §7.3, 원본 해시 23/23 | DFD(3,362영상) P0 70.96, FaceShifter(280) P0 68.01. CI 0 밖: P4a CDF −5.7~−7.1·DFD −4.6~−6.8, U0s DFD +4.2·CDF −10.6, P2 last FSH +6.8·CDF −7.0·DFD −3.7, U3s best DFD +5.2·FSH +6.1, P3 FSH +0.8, U0f CDF −2.1·DFD −2.3 | 사용자 결정 대기 |
| 2026-10-17 | S9 완료(12회, 게이트 통과: registry 12줄·NaN 없음) | **baseline 상단을 풀어도 P0를 넘지 못함.** 짝 비교: P0→U0f FF++ +0.01·CDF −2.09(last, 저하), P0→U0s FF++ −4.3·CDF −10.3(저하), P3→U3f CDF −2.4(저하), U0f→U3f 차이 불분명~저하. 예외: U0s→U3s CDF +7.6(best)·+5.6(last) 향상 — SBI 학습에서 P3 분기와 함께 학습하면 baseline logit만 따로 봐도 CDF 76.5 유지(U0s는 65.6으로 붕괴) → 분기·동적 결합이 SBI 특화 학습을 흡수해 baseline을 보호. U3s는 불안정(s1 last FF++ 73.0) | S10 평가 진행 |
| 2026-10-17 | S10 base 완료(32분) + **재현 검증 게이트 통과**: FF++ test 700영상 22,308프레임, 우리 정렬 crop vs h5 crop NCC 중앙값 0.985(평균 0.968), P0 영상 AUC 94.90 vs 94.97(**Δ −0.07%p**, 기준 ±1%p), 프레임 확률 평균 차 0.032 | 우리 파이프라인이 원본 전처리를 충실히 재현 → DFD·FaceShifter 결과를 FF++와 같은 조건으로 비교 가능 | S9 끝나면 전체 평가 |
| 2026-10-17 | S10 crop 완료: ffv 22,308/22,400프레임(99.6%), fsh 8,929/8,960, **dfd 102,080/109,792(93.0%), 8장 미만 영상 69개 제외**, 열기 실패 0. S9 큐 12회 중 6회 완료(U0f·U3f·U0s 일부) | baseline 출력 캐시(base 단계) 00:24 시작(PID 1408) | 게이트·평가 |
| 2026-10-16 | S9·S10 준비: ① S9 재구성 모듈 `basetop.py` — 캐시(CLS + b_3)로 재구성한 logit이 val 캐시 logit과 **비트 동일**(최대 차 0.0, 게이트 통과). `run_train_unfreeze.py`·큐 `s9_{a,b}.json`(6+6) 작성. ② CLS 캐시 val 완료(재계산 logit 차 0), 나머지 2개 프로세스 진행. ③ S10: h5 정렬 = 고정 템플릿 닮음변환 확인(정렬 랜드마크 눈 위치 SD 약 2px). dlib 68점 모델 다운로드(dlib.net, sha256 fbdc2cb8…). **정렬 점 집합: 눈 12 + 바깥 입술 12(36~59번)** — val 위조 16영상 144프레임에서 h5와 NCC 중앙값 0.982(68점 전체 0.966, 코·눈·입 0.968), test에서도 0.994로 같은 결론. 다운로드 `dl_s10.py`(DFD 전체 3,431 + FF++ 원본·FaceShifter test 280) | DFDC 제외(Kaggle 인증 없음). 다운로드가 0.9MB/s로 느려 동시 8개로 재시작 | 캐시·다운로드 끝나면 S9 큐, S10 crop·base·게이트·평가 |
| 2026-10-16 | S8 완료: 게이트 통과(registry kind=sbi 9줄, NaN 없음), 원본 해시 23/23, 위치 평가(S8 heatmap 포함) 재실행, `analyze_s8.py` 작성·실행, `S7_report.md`에 §10(S8) 추가, HANDOFF §7.2 추가 | **가설 반증**: P0 대비 SBI 학습 짝 비교(last) — P2 FF++ −3.97·CDF −14.5(저하), P3 FF++ −3.00(불분명)·CDF −7.4(저하), P4a FF++ −3.47·CDF −4.7(저하). 원인: detail 단독 AUC가 FF++ 학습보다 낮음(결합-baseline 상관 0.97→0.72~0.89로 더 하락), P4a는 학습이 불안정(손실 스파이크, 균형ACC SD 16.6%p). SBI 학습 안에서도 P2→P3 향상 재현(FF++ +2.1·CDF +7.3, 2SD 충족), detach 위치 보호도 재현(경계적중 35%→8%) | 사용자 결정 대기 |
| 2026-10-16 | S8 준비: `run_train_sbi.py` 작성(run_train 함수 재사용, 기존 학습 코드 무변경), `run_queue.py`에 script 항목 추가, 스모크(P4a 2 epoch×30스텝) 통과, 어댑터 검증(baseline SBI val AUC 72.27 일치, 이미지·라벨 대응, epoch당 진짜 7,200·가짜 7,200·seed 무관 쌍 집합) | 30스텝 시점 SBI val 손실 2.99·FF++ val 균형정확도 51%(학습 초기 실제 동작, 버그 아님) | S8 큐 2개 20:49 병행 실행 |
| 2026-10-16 | S7 완료: `analyze.py`(표·짝 비교·데이터 양·진단·누수·그림 4종) + `reports/S7_report.md` 작성, 수치를 원자료와 대조해 정정(상관 범위, CDF 미탐률 범위, heatmap head 설정 간 비트 동일 직접 확인). 논문 수치 확인: Table 3 CDF 95.10은 SBI 학습 변형, c40 학습 모델은 Table 4 74.82(우리 P0 76.15와 일치). HANDOFF §7.1 갱신. 원본 해시 최종 대조 23/23 | 게이트 통과(`reports/`에 표·그림·결론) | 사용자 결정 대기(후속 실험) |
| 2026-10-15 | S6 완료: 위치 평가 전체(test 정렬 프레임 5,472장, 임계값·τ는 val) | 경계 AUROC / 경계 적중 / 셀 AUROC / 최댓값 적중 — seg head(P4b·P4): 70.8 / 35.3% / 80.6 / 82.1%, P3 분류 heatmap: 70.5±0.1 / 35.4±0.1% / 81.2 / 79.4%(P2a·P2·P3a·P3b 같음 — BN 통계 차이 영향 없음), **P4a: 56.1±2.3 / 4.8±2.0%(last 50.8 / 4.0%)** / 84.6±5.3 / 97.1%, 얼굴 중앙 기준선: 47.7 / 0% / 93.6 / 99.3%. 방식별 seg head 경계 적중 FS 49.9·F2F 32.1·NT 31.2·DF 27.6% | S7 |
| 2026-10-15 | S4 완료(04:37~11:30, 27회, 후반 1회 약 14분). 원본 해시 재대조 23/23 일치 | 게이트 통과: registry 27줄(9설정×3seed), 곡선 NaN 없음, best epoch 대부분 1(17/27). best 평균 FF++ AUC: P0 95.22 / P1 94.95 / P1′ 94.87 / P2a 95.03 / P2b 94.88 / P2 95.13 / P3a 95.10 / P3b 95.10 / P3 95.11 / **P4a 94.32**. CDF: 76.15 / 75.57 / 76.00 / 76.73 / 76.03 / 76.28 / 76.11 / 76.56 / 76.86 / **70.44**. 짝 비교(3/3 + 2SD): P0→P1 저하(FF++ −0.27, CDF −0.58), P1′→P2a 균형ACC +2.56, P2→P3 last에서 향상(FF++ +0.99, CDF +7.09, 후반 붕괴 방지), **P3→P4a CDF −6.41(best)·−7.17(last) 저하**, P0→P3 차이 불분명 | S6 전체 위치 평가 → S7 |
| 2026-10-15 | S6 위치 평가 **미리보기**(S4 진행 중, seed 0·1 일부) | seg head: 셀 AUROC 80.6·경계 AUROC 70.8·IoU 0.60·최댓값 적중 82.1%·경계 적중 35.3%. detach 있는 분류 heatmap(P2a·P2·P3a·P3b·P3)은 seg head와 거의 같음(경계 적중 35.2~35.7%) — BN 통계만 다르고 같은 seed·epoch면 설정 간 비트 동일(공통 난수 확인). **P4a(detach 제거)는 heatmap이 얼굴 중앙으로 붕괴**: 경계 적중 35%→2%, 경계 AUROC 70→54(last 48), 대신 "안쪽" 지표는 중앙 기준선처럼 상승(최댓값 적중 99%) — 안쪽 지표의 중앙 편향 착시 | S4 끝나면 전체 재실행 |
| 2026-10-15 | S5 완료(05:22~08:26, S4와 병행, 1회 5~10분) | 게이트 통과: registry 24줄, 20 epoch 곡선 전부 유한, FAILED 없음. best epoch가 데이터가 적을수록 늦어짐(1~9, epoch당 스텝 225/450). seed 평균 best FF++ AUC — P1 94.83/94.89, P2 95.05/94.94, P3 94.64/94.95, **P4a 93.01/93.82** (25%/50%); CDF best — P1 75.9/76.1, P2 76.2/76.4, P3 75.1/76.9, **P4a 71.0/73.6**. last에서 P1·P2는 떨어지고(고정 α) P3는 유지(동적 α가 baseline 쪽으로) | S4 끝나면 S6·S7 |
| 2026-10-15 | S7 분석 코드 `analyze.py` 작성·부분 결과로 시험(표·Δ·데이터 양·진단·누수·그림 4종) | 1학기 분할 누수 재계산: 공식 test 140영상 중 **115개가 종류마다 1학기 학습 쪽**(HANDOFF와 일치), test 신원 140개 중 **126개**가 1학기 학습 영상(진짜 또는 가짜 쌍)에 등장(이전 요약 "66/200"은 정의 불명 → 이 값으로 대체). 진단(P1·P1′): detail 단독 영상 AUC FF++ 72~78·CDF 54~58, 결합-baseline 상관 0.96~0.98 | S4 완료 후 전체 실행 |
| 2026-10-15 | S2 완료: 원본 방식 시범 `pilot_P3_orig_s0` 104.7분(캐시 방식 18.3분, 약 5.7배) | 캐시−원본: best FF++ AUC +0.17·ACC@0.5 +0.86·균형 −0.54·오탐 +2.86·CDF AUC −0.78 / last +0.09·−0.09·+0.71·CDF −0.19. 프레임 확률 상관 0.991~0.997(평균 차 0.015~0.029), top-1 peak 위치 96.7% 동일, heatmap 가중치 동일·BN running_var 상대 차 약 2%. 동적 α 평균 best 0.64/0.62 → last 0.95/0.88(학습할수록 baseline 쪽으로). 두 방식 모두 best epoch 1 | 차이 = 캐시 근사 + 난수 흐름 차이(원본은 bridge_adapter dropout이 CUDA 난수 소비)의 상한. seed 변동(S4)과 비교해 S7에서 해석. epoch 20 유지 |
| 2026-10-15 | S4 스모크 P1_d100_s0 완료(27.9분, 다른 작업과 GPU 공유) | 파일·registry 줄 정상, NaN 없음. best=1 epoch: FF++ AUC 94.88·CDF 75.51 / last: 94.27·74.77 (P0 95.22·76.15보다 약간 낮음). val 손실 1 epoch 최저 후 상승(시범과 같은 양상) | S4 본 큐 04:37 분리 실행(PID 7408) |
| 2026-10-15 | S3 완료: SBI 사전학습 10 epoch(18분, 다른 작업과 GPU 공유) → best = 10 epoch(val focal+CE 0.585, 아직 완만히 감소) | SBI val: detail crop 정확도 82.7%, heatmap 최댓값 진짜/가짜 AUC 84.0, top-1 peak GT 적중 20.9%, peak 경계값 0.30. **1학기 SBI 가중치는 같은 SBI val에서 50.2%·AUC 50.2(찍기)·적중 10.6%** → 입력 규약이 달랐을 가능성, 새 사전학습이 우리 파이프라인에 맞음. seg head 격자(`reports/s3_seghead_grid.png`): 얼굴 둘레 고리 모양, FF++ 실제 가짜에서도 마스크 윤곽을 따라감. SBI val crop 적중(경계 픽셀 하나라도 포함)은 무작위 초기에도 100%라 포화 → 기록만 | S4 |
| 2026-10-15 | S6 준비: 정렬 재실행(화면 전체 SIFT 재시도) → val 98.6%·test 97.7% 통과. 방식별 임계값(val 히스토그램 골짜기) DF 84·FS 7·NT 20·F2F 20, 조작 면적 42~54%. `loc_eval.py` 시험: 얼굴 중앙 기준선이 안쪽 지표에서 포화(최댓값 적중 99.3%, IoU 0.73) → **경계 셀 AUROC 보조 지표 추가**(중앙 47.7, 1학기 SBI head 77.0) | 위치 평가는 S4 끝난 뒤 실행 | S3 사전학습 |
| 2026-10-15 | S6 탐색: h5 ↔ 위조 c40 영상 정렬 방법 확정 | 1/4·원 해상도 축 정렬 템플릿 매칭은 실패(1·2위 점수 차 0.01). **SIFT+RANSAC 닮음변환으로 해결**: h5 프레임 i = 영상 프레임 floor(i·T/N)(균등 추출, N=h5 프레임 수), h5 얼굴은 회전 약 ±8°·배율 약 1.4로 정렬된 crop. 정렬 후 NCC 중앙값 1.000. 1차 전체 실행: val 97.6%·test 95.1% 프레임 통과(영상 단위 99.8%), 격자 `reports/s6_align_grid_test_probe3.png` 육안 확인 | 실패 프레임은 전체 화면 SIFT로 재시도(재실행 중) |
| 2026-10-15 | S2 시범 캐시 방식(`pilot_P3_cache_s0`, 1학기 SBI 초기화) 완료 18.3분 | best=1 epoch: FF++ AUC 95.18·오탐 25.7%·CDF 76.35 / last=20: 95.22·25.7%·76.22 (P0 95.22·27.1%·76.15와 거의 같음). val detail CE 0.70~0.95(찍기 ln2=0.69 수준), val 손실 1 epoch 최저 후 상승, val AUC 평평(프레임 92.1·영상 94.9). 학습 정확도 97.6%는 고정 baseline이 이미 학습 데이터를 맞추기 때문 | epoch 20 유지(늘릴 이유 없음, best+last 기록으로 조기 종료 효과 포함) |
| 2026-10-15 | S3 SBI 쌍 생성 | train 56,308쌍(523영상 14,077프레임 × 4) + val 880쌍, 1.1분(6프로세스). 블렌드 비율 분포·68점 25%·반전 50%·JPEG 50% 설계대로, 재생성 == 캐시(결정적) | baseline 캐시 진행 |
| 2026-10-15 | S2 게이트 ④ 재현성 | P4a 2 epoch×40스텝을 같은 seed로 2회: 파라미터 742개·옵티마이저 상태·곡선(시간 제외) 비트 동일 | 시범 실행 |
| 2026-10-15 | S1 완료: 캐시 생성 26분(test 70,000 / val 5,600 / cdf 16,420 / train 36,000 + 반전 36,000), P0 재평가 | 게이트 통과: 영상 AUC 95.219 vs 95.217, ACC@0.5 91.0·균형 84.2·오탐 27.1%·미탐 4.5% 동일, 방식별 AUC 동일(±0.005). Celeb-DF(전 프레임, 양쪽 10% 제외): AUC 76.15, ACC@0.5 53.1, 균형 62.5, 오탐 7.3%, **미탐 67.6%**, 최적기준 ACC 74.3. test_only h5 == 전체 h5(표본 12/12), val 캐시 이미지 == h5(40/40) | S2 게이트 ④·시범 |
| 2026-10-15 | S2 단위 검증(게이트 ①②③): crop 벡터화 121위치·무작위 b3 비트 동일, Method 4 20 seed 동일, 색 변형 torchvision 대비 최대 6e-8(평균 계산 순서), 캐시 == 실시간 baseline(b3 비트 동일), P3 결합 출력이 원본 M2F2DetImproved와 eval·train 모두 비트 동일, P3 heatmap grad 없음 / P4a grad norm 0.034 | 통과. 처음엔 결합 출력이 7e-4 달랐음 → 원인: autocast에서 detail은 fp32(sum), 원본 baseline logit은 fp16 → 캐시 logit을 autocast일 때 fp16으로 맞춰 해결 | 게이트 ④는 train 캐시 후 |
| 2026-10-15 | S1 baseline 점검(`check_baseline.py`) | bridge_adapter 출력 배치 표준편차 0(상수 확인), 같은 모델 2회·다른 seed 재구성 모두 logit 완전 동일, fp32 vs P0 최대 0.00025, fp16 vs fp32 최대 0.012, 프레임당 9.2ms(배치 32) | 캐시 생성 |
| 2026-10-15 | S0: 폴더 뼈대(`code/ cache/{index,frames,base} runs/ reports/ logs/`), PROTOCOL.md(고정 변수·설정·세부 규칙), registry.csv 헤더, 원본 해시 대조 | 23/23 일치, 가중치 목록 동일, GPU 922MiB(데스크톱만), RAM 31GB, 디스크 여유 449GB. 원본 코드에서 bridge_adapter가 list라 미로드·dropout 활성이지만 LayerNorm(1) 때문에 출력 상수임을 발견(S1에서 수치 확인) | S1 |
| 2026-10-15 | 인수인계 문서(HANDOFF.md)·PROGRESS.md·시작 프롬프트 작성, 참고 파일을 `ref/`로 복사, 원본 코드 해시 기록 | 완료 | S0 |
| 2026-10-14 | FF++ 위조 마스크·c40 영상(test+val, 4종 양방향) 2,240개 다운로드 및 검증 | 실패 0건, 10.46GB, FaceSwap 마스크 값이 낮음(임계값 방식별) | 실험 코드 작성 대기 |
| 2026-10-14 | P0 평가, 데이터·속도 측정, 설계 승인 | HANDOFF.md §4~5 참조 | |

## 결정·변경 기록 (설계에서 달라진 것은 사유와 영향을 적는다)
| 날짜 | 내용 | 사유 | 영향 |
|---|---|---|---|
| 2026-10-21 | 질문 방어 실험 추가(사용자 위임): 새 코드 `code/qa_*.py`·`ref/qa_followup/`·`queues/qa_*.sh`, 기본 miniconda에 imageio-ffmpeg 설치(승인), FF++ c23 test 영상 700개 다운로드(`data/ffpp_c23_test_download.py`, 팀 약관 동의 범위). 기존 코드·결과·registry 불변(새 학습은 registry 미기록). 사전 고정 규칙 B("FF++ val 비저하 + 선택 점수 최고")는 사용자 승인 후 실험 전에 고정. 최종 모델은 팀 결정대로 TASKS5 병합 유지 | 최종발표 질의응답 대비 | 기존 수치 불변. 새 결과는 qa_followup.md에만 |
| 2026-10-17 | **최종 모델 = TASKS5 task vector 병합(A안)**, Xp3 병합(TASKS6)은 보조 결과 | 사용자가 권장안 선택. 모든 선택이 사전 규칙인 모델을 본 결과로 둠(Xp3는 사후 규칙으로 선택) | 논문 v2·발표 개요·GUIDE의 본 결과 수치 = TASKS5 병합 |
| 2026-10-14 | 평가는 논문과 같은 100프레임, fp16 | P0와 같은 방식, 시간 절약 | 모든 설정 동일 |
| 2026-10-14 | P4a = detach 제거만 | 기여도를 섞지 않기 위해 | |
| 2026-10-14 | SBI 사전학습 새로 수행, P1′ 추가 | 새 코드로 방법을 정확히 기술, SBI 효과 분리 | 학습 +3회 |
| 2026-10-14 | baseline 출력 캐시, seg head = SBI heatmap head 고정 복사본 | GPU 70h→15h, 분류 불변 구조 | 색 변형을 crop에만 적용(S2에서 차이 확인) |
| 2026-10-14 | 마스크는 test+val만 다운로드(원본 c40 영상 불필요) | 위치 평가에 train 마스크 불필요 | |
| 2026-10-15 | code_sha 정의 변경: `code/*.py` 전체 → `common.py, expmodel.py, run_train.py`만 | 분석 스크립트를 추가할 때마다 해시가 바뀌어 같은 학습 코드인데 다른 값이 됨 | `P0_ref`·시범·`P1_d100_s0`(스모크, 04:07 시작)은 옛 정의 값(a95733826b7d 등). 이후 본실험은 새 정의(a3339a0e9a6f, common.py의 code_sha 함수만 바뀜 — 학습 동작 동일). S4·S5 중에는 이 3개 파일을 고치지 않는다 |
| 2026-10-15 | 위치 평가에 경계 셀 AUROC 보조 지표 추가 | FF++ 마스크가 얼굴 crop 절반을 덮어 "안쪽" 지표가 중앙 기준선에 포화(변별 불가) | 설계 지표(IoU·최댓값·crop 적중)는 그대로 보고하고 경계 지표를 함께 보고 |
| 2026-10-16 | **설계 변경: S9·S10 추가**(PROTOCOL §15·§16). S9 = baseline 상단(EffNet features[7..8] + 투영층 + 출력층, 약 10.3M) 풀기, 캐시(b_3 + CLIP CLS)로 계산, 풀린 모듈은 eval 모드(BN 고정). S10 = DFD 전체·FaceShifter test로 교차 평가, 얼굴은 h5와 같은 닮음변환 정렬로 재현하고 FF++ test에서 h5와 직접 대조해 검증 | 사용자 승인("2,3 진행"). DFDC는 Kaggle 인증이 없어 제외. 캐시 방식으로 baseline 상단 학습이 저렴해짐 | 기존 결과 불변. 새 학습 12회 + 평가 |
| 2026-10-16 | **설계 변경: 후속 실험 S8 추가**(PROTOCOL §14) — P2·P3·P4a를 SBI 쌍(공식 train 원본 + SBI 가짜)으로 학습, 선택은 SBI val, 나머지는 S4와 동일 | 사용자 승인("설계변경이 필요하면 추천방식으로 진행"). S7 원인 가설(같은 데이터라 분기가 배울 것 없음)을 직접 검증하는 가장 싼 실험(기존 캐시 재사용). baseline 풀기·DFD/DFDC·SBI 사전학습 연장은 비용·비교성 문제로 제외 | 기존 결과 불변. 새 실행 9회, 약 1.5시간 |
| 2026-10-15 | S5 큐를 S4와 **병행** 실행(05:22) | 단일 실행의 GPU 사용률 34%·메모리 1.9GB(CPU·입출력 병목) → 두 큐를 동시에 돌리면 전체 약 6시간 단축. S5는 S4 결과에 의존하지 않고 실행 이름도 겹치지 않음 | 결과 불변(프로세스마다 결정적), train_min(소요 시간)만 경합으로 늘어남 — 시간 비교에는 쓰지 않음 |
| 2026-10-15 | numpy BLAS 우회 확정: `MKL_THREADING_LAYER=SEQUENTIAL` | matplotlib도 좌표 변환에 행렬곱을 써서 그림 단계에서 조용히 죽음 → 원인은 MKL 스레드 계층 DLL | 분석·그림 스크립트에만 적용, 결과 영향 없음 |
| 2026-10-15 | numpy 행렬곱 사용 금지(환경 우회) | conda generic BLAS 구현 DLL이 깨져 np.corrcoef 등에서 0xC06D007F 즉사. 환경은 앱 서버와 공유라 건드리지 않음 | 결과 영향 없음. eval_p0_cache의 상관계수를 원소별 계산으로 교체 |
| 2026-10-15 | SBI 경계 GT는 **블렌드 비율을 곱하기 전 마스크**로 B=4M(1−M) 계산 (S3 위임 범위) | 비율을 곱한 마스크로 하면 비율 ≤0.5에서 얼굴 내부 전체가 최대가 되어 GT가 경계가 아닌 큰 덩어리가 됨(격자 그림으로 확인) | 모든 SBI 표본에서 경계 고리 4사분면 peak. PROTOCOL §4.8에 S3 완료 시 기록 |
| 2026-10-15 | 위임된 세부 확정(PROTOCOL §3~5): 프레임 i=floor((j+0.5)n/k), 1:1 = 매 epoch 진짜 전부 + 가짜 4묶음 순환(영상 단위 균형, seed 무관 고정), nested = 쌍 순열(seed 2026) 앞 90/180/360, val_loss_bal(클래스 균형 CE)로 best 선택, 데이터 난수는 seed만으로 결정(설정 간 공통), 특징 평균 맵 = σ(b_3 채널 평균), P2b 집계 = 단순 평균, Celeb-DF = 전 프레임·256→224 INTER_LINEAR·양쪽 floor(0.1n) 제외 | HANDOFF §5가 "PROTOCOL.md에 기록"으로 위임한 항목 | 모든 설정 동일 적용. S2 시범은 1학기 SBI 가중치(읽기 전용)로 캐시/원본 두 방식 비교 |

## 열린 문제
- ~~마스크 정렬 방법 미확정~~ → SIFT+RANSAC 닮음변환으로 해결(실행 로그 참조). 남은 것: 방식별 이진화 임계값(val) — S6
- P2a·P2b 정의, heatmap BN 처리 — S2에서 PROTOCOL.md에 확정 기록
- ~~신원 겹침 수치(66/200) 재확인~~ → 재계산 126/140(test 신원 중 1학기 학습 영상에 등장), 영상 115/140(`reports/s7_leakage.json`)
- ~~M2F2-Det 논문의 Celeb-DF 교차 데이터셋 AUC~~ → 확인 완료: Table 3의 95.10은 SBI 학습 변형("w/SBI"), FF++(c40) 학습 모델은 Table 4에서 74.82 → 우리 P0 76.15와 일치
