# 4가지 후속 작업 계획·진행 기록 (TASKS4)

> 2026-10-17 사용자 요청. **이어하기: 이 파일의 "현재 위치"부터 진행한다.** 각 단계가 끝날 때마다 이 파일을 먼저 갱신한다.
> 기존 실험 규칙(HANDOFF·PROTOCOL)은 그대로 따른다: 원본 코드 수정 금지, 새 코드·결과는 `C:\m2f2\exp\` 아래, 테스트셋으로 고르지 않기.

## 요청 4가지와 산출물

| # | 요청 | 산출물 |
|---|---|---|
| 1 | 결과를 솔직하게 쓰되 가치 있는 논문 초안 (데이터·환경·방법 정리) + "성능 향상 없음"을 논문·최종 발표로 내도 가치 있는지 평가 문서 | `paper/draft_v1.md`, `paper/value_assessment.md` |
| 2 | 최종 발표에서 성과가 최대한 돋보이는 분석 + 분석 방법 문서 (**실제로 오른 수치만, 선택 규칙 공개, 테스트로 고르기·불리한 결과 숨기기 금지**) | `paper/presentation_highlights.md` |
| 3 | 지금까지 기능 중 최고 조합과 결과 | `reports/t3_best_combo.md` |
| 4 | 결과에서 찾은 약점 기반 개선 제안 + 실제 구현·테스트 (논문 가치 우선) | `reports/t4_improvements.md`, PROTOCOL §17 |

## 공통 원칙 (3·4번 선택 규칙)
- **선택(하이퍼파라미터·조합·가중치 고르기)에는 선택 전용 데이터만 쓴다**: FF++ val(도메인 내) + **DFD-val**(DFD를 배우 신원 기준으로 반으로 나눈 앞쪽, 도메인 밖).
- **보고 전용(끝까지 안 봄)**: FF++ test, Celeb-DF, FaceShifter, **DFD-test**(DFD 나머지 반).
- 판정: 보고 데이터에서 P0 대비 영상 단위 짝 bootstrap 95% CI(2,000회)가 0을 벗어나야 "향상". seed가 있으면 seed 평균.

## 단계와 상태

| 단계 | 내용 | 상태 | 결과·파일 |
|---|---|---|---|
| A1 | DFD 신원 분할(val/test) 정의 | **완료** | `cache/index/dfd_split.json` — 배우 28명 14/14(seed 2026), val 879·test 1,195영상, 신원 섞인 가짜 1,357 제외 |
| A2 | 점수 은행: 모든 모델의 영상 점수(FF++ val·test, CDF, DFD-val·test, FSH) | **완료** | `cache/scorebank.npz`(591항목, 97모델), `code/t34_common.py`·`t34_scorebank.py`. 유효 영상: dfdval 864(181/683), dfdtest 1,165(174/991), fsh 280 |
| T4-1 | 개선① 대상 도메인 BN 적응(AdaBN, 먼저 상단 f7·f8만 캐시로 → 효과 있으면 전체) | **완료(효과 없음)** | 선택 세트가 λ=0(적응 안 함)을 고름. λ↑일수록 CDF 소폭↑(최대 +0.3)·DFD/FSH↓ → 전체 AdaBN은 진행 안 함. `reports/t4_results.md` |
| T4-2 | 개선② 고정 seg head 경계 증거 점수 결합 | **완료(효과 없음)** | 경계 증거 단독 AUC: FF++ 84~87(도메인 안에선 유효) / 교차 55~58 → 결합 β=0 선택 |
| T4-3 | 개선③ 라벨 없는 판정 기준 적응(오탐/미탐 균형) | **완료(효과 없음)** | 2성분 GMM 기준: CDF 균형정확도 62.5→68.5지만 FF++ val 84.6→70.8, DFD-val 66.2→64.3 → 선택 세트에서 탈락 |
| T4-4 | 개선④ FF++ 전문가 + SBI 전문가 상보 앙상블(가중치는 DFD-val로) | **완료(향상)** | P3 last + U3s best(w=0.6): DFD-test **+3.91**·FSH **+4.75**(CI 0 밖), FF++ test +0.01, CDF +1.73(불분명). P0+U3s(w=0.7): DFD-test +4.70·FSH +5.09 ✔, FF++ −0.44 |
| T4-5 | 효과 있는 것 조합 + (필요 시) 학습 기반 개선 | **완료** | ① 앙상블 확장(`code/t34_select.py`, `reports/t4_ensemble.md`) **완료**: 선택 1위권 전부 P3+U3s 계열. P3(last)×0.3 + U3s(best)×0.7(seed 평균): DFD-test **+4.82**·FSH **+5.80**(CI 0 밖), CDF +1.38·FF++ −0.30(불분명). ② 단일 모델 대안 혼합 학습 U3m·U0m×3 **완료(앙상블보다 못함)**: FSH 63~65(P0 68.0보다 낮음), DFD-test 70.7~73.1(≈P0), CDF best 75~78.5, FF++ ≈P0 → 두 전문가를 따로 학습해 결합하는 쪽이 낫다는 근거. `code/t4_eval_new.py`로 은행에 추가(675항목) |
| T4-6 | 개선⑤ heatmap 고정점(anchor) 학습: detach 제거 + 고정 seg head와의 MSE 정규화(λ) → P4a의 중앙 붕괴를 막으면서 end-to-end 학습 | **완료(원인 규명, P3 초과 없음)** | λ∈{1,10,100} seed 0 → 선택 세트로 λ=100 → seed 1·2 추가. 용량-반응: 경계 적중 5.0/14.9/**37.4±0.4%**, CDF 69.3/74.8/**76.47±0.49**, DFD-test 66.6/68.3/**70.25±0.61** (P4a 4.8%·70.44·66.43, P3 35.4%·76.86·70.75). P0 대비(best): FF++ −0.20·CDF +0.32·FSH +0.58·DFD-test −0.53 ✔. → 손해 원인은 gradient가 아니라 위치 붕괴. `code/run_train_anchor.py`, `reports/t4_improvements.md` §5, `reports/t4_fig_anchor.png`, 위치 지표 `reports/s6_loc.csv`(P4c_*) |
| T4-7 | 결합 방식 비교(logit·순위·z점수·신뢰도 게이트) | **완료** | `code/t4_fusion.py`, `reports/t4_fusion.md`. P3(last)+U3s(best): 선택 1위 **순위 평균 w=0.5**(선택 85.47) → FF++ −0.15·CDF +1.78·**FSH +5.98 ✔·DFD-test +4.52 ✔**. z 평균 w=0.5(85.44): FF++ +0.05·CDF +2.23(CI 하한 −0.17)·FSH +5.53 ✔·DFD +4.65 ✔. logit w=0.7(85.39, 영상 1개에도 적용 가능): −0.30·+1.38·+5.80 ✔·+4.82 ✔. 순위·z는 평가 세트 분포가 필요(transductive) → 실사용은 logit |
| T3 | 기존 기능 최고 조합 선택·보고 | **완료** | `reports/t3_best_combo.md`. 선택 1위 = P3(last, 현재 모델): FSH +0.82 ✔, FF++ +0.07·CDF +0.09·DFD-test −0.39(불분명). 2위 P3a(best) FSH +1.96 ✔, P2(best) FSH +3.00 ✔. 앙상블 포함 최고 = P3×0.3+U3s×0.7 |
| T1a | 논문 초안 | **완료**(혼합 학습·결합 방식·anchor 3seed·위치 지표 반영, [채움] 없음) | `paper/draft_v1.md` |
| T1b | 가치 평가 문서 | **완료** | `paper/value_assessment.md` |
| T2 | 발표용 분석 문서 | **완료** | `paper/presentation_highlights.md` (실제 향상만, 정당한 표현법, 금지 목록·이유, 예상 질문) |
| Z | 최종 정리·보고 | **완료** | PROGRESS·PROTOCOL §17 갱신, 발표·가치 문서에 anchor·혼합 학습 반영, 원본 해시 23/23 일치(`logs/original_hash_check.txt`) |

## 현재 위치
- **4가지 작업 모두 완료(2026-10-17 05:10).** 진행 중인 백그라운드 작업 없음.
- 결과물: `paper/draft_v1.md`, `paper/value_assessment.md`, `paper/presentation_highlights.md`, `reports/t3_best_combo.md`, `reports/t4_improvements.md`(+ `t4_results.md`, `t4_ensemble.md`, `t4_fusion.md`, `t4_fig_anchor.png`)
- 추가 요청이 오면: 이 표의 결과를 기준으로 시작한다. 새 학습은 `code/run_queue.py`(script 분기) + registry 한 줄 규칙을 그대로 따른다.
- 마지막 갱신: 2026-10-17 05:10

## 진행 로그 (최신이 위)
| 시각 | 한 일 | 결과 |
|---|---|---|
| 2026-10-17 | Z: TASKS4·PROGRESS·PROTOCOL §17 갱신, 발표·가치 문서 보강, 원본 해시 23/23 | 4가지 작업 완료 |
| 2026-10-17 | anchor λ=100 seed 1·2 완료·평가, 위치 평가(loc_eval), 그림, t4_improvements·논문 §5.6 완성 | 위치 37.4%·CDF 76.5·DFD 70.3 = P3 수준, P3 초과 없음 |
| 2026-10-17 | 논문 초안 §5.5(결합 방식·혼합 학습)·§5.6(anchor) 갱신, loc_eval에 anchor 추가 | — |
| 2026-10-17 | anchor λ 1·10·100 seed 0 완료·평가 | λ=100 선택, P3 수준까지 회복 |
| 2026-10-17 | T4-7 결합 방식 비교 | 순위/z 평균(같은 가중치)이 logit 평균과 동급 이상 |
| 2026-10-17 | 혼합 학습 6회 완료·평가 | 단일 혼합 모델은 FSH↓, 앙상블 이득 재현 못함 |
| 2026-10-17 | T1b·T2·T1a(초안) 작성 | `paper/` 3개 문서 |
| 2026-10-17 | T3·T4-4b 완료(`code/t34_select.py`), 혼합 학습 시작 | 단일 최고 = P3(last), 앙상블 최고 = P3+U3s |
| 2026-10-17 | T4-1~4 완료 (`code/t4_methods.py`, `reports/t4_results.md`) | ①②③ 효과 없음, ④ 상보 앙상블이 DFD-test·FSH에서 확실한 향상 |
| 2026-10-17 | A1·A2 완료 | 점수 은행 생성 |
| 2026-10-17 | 계획 수립 | — |
