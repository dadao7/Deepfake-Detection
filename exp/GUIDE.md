# DeepSight 통제 비교 실험 — 전체 안내 (파일 지도 · 읽는 순서 · 논문 버전)

작성 2026-10-17. 이 실험 폴더(`C:\m2f2\exp\`)에서 지금까지 한 모든 일과 그 결과 파일이 어디 있는지 정리한 안내서다.
경로는 모두 `C:\m2f2\exp\` 기준이다. 원본 코드(`C:\m2f2\code`, `M2F2_Det`, `weights`)는 한 번도 수정하지 않았다(해시 23/23 일치, `logs/original_hash_check.txt`).

---

## 1. 한눈에 보는 진행 흐름

| 구간 | 단계 | 질문 | 한 줄 결과 |
|---|---|---|---|
| **A. 논문 모델 + 우리 기능 검증** | S0~S7 | 논문 모델(P0) 위에 우리가 붙인 기능(P1~P4)이 실제로 성능을 올리나? | 단일 기능으로는 P0를 확실히 못 넘음. detach가 위치·일반화를 지키고, 동적 결합이 과적합을 막는다는 것을 발견 |
| **B. 후속 원인 검증** | S8~S10 | 학습 데이터를 바꾸면? baseline을 풀면? 다른 데이터셋에서는? | 어느 쪽도 P0를 일관되게 못 넘음. 효과가 데이터셋마다 반대(평가 함정) |
| **C. 자체 개선 1차** | TASKS4 | 결과에서 찾은 약점으로 개선할 수 있나? | 상보 전문가 결합(모델 2개)으로 DFD +4.8·FaceShifter +5.8(P0 대비). 나머지 4종은 효과 없음 |
| **D. 자체 개선 2차** | TASKS5 | P3(현재 모델)를 단일 모델로 넘을 수 있나? | **학습 없는 전문가 가중치 병합**으로 P3 대비 Celeb-DF +2.6·FaceShifter +4.1·DFD +5.3(모두 CI 0 밖) |
| **E. 추가 향상 탐색** | TASKS6 | 더 올릴 수 있나?(8가지 방법) | Xp3 병합이 기존 병합보다 FaceShifter +1.9(손해 없음, 사후 규칙). 선택 세트 과적합 발견, 이득은 baseline 상단에서 옴 |

> **최종 모델(팀 결정 2026-10-17)**: TASKS5의 task vector 병합 단일 모델(P3best·U3s, a = 0.5·b = 0.75). P3 대비 Celeb-DF +2.6·FaceShifter +4.1·DFD +5.3(CI 0 밖), FF++ 유지. TASKS6의 Xp3 병합은 사후 규칙으로 고른 보조 결과다.

> **처음부터 흐름을 이해하려면 `PROJECT_FLOW/00_프로젝트_흐름_안내.md`부터 읽는다.** 단계별 모델 구조 그림(바뀐 부분 표시), 설정·결과·해석, 읽는 순서가 정리되어 있다. 같은 폴더에 필요한 문서를 읽는 순서대로 복사해 두었다(`python code/make_project_flow.py`로 다시 만듦).

## 2. 공통 문서 (어느 단계든 기준이 되는 파일)

| 파일 | 내용 |
|---|---|
| `GUIDE.md` | 이 안내서 |
| `HANDOFF.md` | 실험 설계 원본: 규칙, 환경, 데이터, 이전 세션에서 밝힌 사실(§4), **설정 정의 P0~P4(§5.2)**, 단계·게이트(§6), 단계별 결과 요약(§7.1~7.3) |
| `PROTOCOL.md` | 실험 규칙 상세: 고정 변수·학습법·평가·판정(§1~13). 후속 설계는 S8 §14, S9 §15, S10 §16, TASKS4 §17, TASKS5 §18 |
| `PROGRESS.md` | 현재 위치, **결정·변경 기록**(설계를 바꾼 이유와 영향), 실행 로그(날짜별), 환경 주의 |
| `TASKS4.md` / `TASKS5.md` | 자체 개선 1차·2차의 계획·단계표·진행 로그 |
| `START_PROMPT.md` | 새 세션에서 이어갈 때 쓰는 시작 문구 |
| `registry.csv` | 학습 실행마다 한 줄. P0 기준 1줄 + **학습 95회**(시범 2, S4·S5 51, S8 9, S9 12, 혼합 6, anchor 5, 증류 3, SBI 전문가 개선 7). 설정, seed, best/last 지표, 학습 시간, 코드 해시가 들어 있다. SBI 사전학습 1회는 `runs/sbi_pretrain/`에 있다 |
| `runs/{실행 이름}/` | 실행별 config.json, curves.csv(epoch별 손실·AUC·ACC), ckpt_best/last.pt, pred_best/last.npz(테스트 예측값), train.log |
| `cache/scorebank.npz`, `cache/scorebank_t5.npz` | 모든 모델의 영상 점수 모음(TASKS4·5 분석의 원자료) |
| `logs/` | 캐시 생성·학습 큐·분석 로그, 원본 해시 대조 기록 |
| `code/` | 실험 코드 59개(아래 단계별 표에 주요 파일 표시) |

## 3. 단계별 결과 파일 지도

### A. 논문 모델부터 기능을 하나씩 추가한 실험 (S0~S7)

| 단계 | 한 일 | 결과 정리 파일 (`reports/`) | 주요 코드 (`code/`) |
|---|---|---|---|
| S0 준비 | 폴더 뼈대, 규칙 문서, 원본 해시, 데이터 점검 | `PROTOCOL.md`, `ref/original_code_sha256.txt`, `ref/data_check/`, `ref/p0/` | – |
| S1 캐시·P0 재현 | 공식 분할 인덱스, 이미지·baseline 출력 캐시. P0 재평가 AUC 95.22(논문 96.58) | `p0_cache_metrics.json` | `build_index.py`, `build_cache.py`, `check_baseline.py`, `eval_p0_cache.py` |
| S2 실험 모델·시범 | 옵션형 모델(P1~P4를 설정으로 바꿈), 원본과 같은 출력인지 검증, 캐시 방식 5.7배 가속 | `s2_pilot_compare.json`, `s2_pilot_curves.csv`, `S7_report.md` §7 | `expmodel.py`, `run_train.py`, `verify_s2.py` |
| S3 SBI 사전학습 | SBI(진짜 영상으로 가짜 합성) 생성 + heatmap head·분기 사전학습 | `s3_sbi_compare.json`, `s3_sbi_grid.png`, `s3_seghead_grid.png` | `sbi.py`, `build_sbi_cache.py`, `pretrain_sbi.py`, `s3_check.py` |
| **S4 본 실험** | **P1→P1′→P2a/P2b→P2→P3a/P3b→P3→P4a, 9설정 × seed 3 (100% 데이터)** | **`S7_report.md` §2·§3**, `s7_tables.md`, `s7_main_table.csv`, `s7_deltas.csv`(짝 비교), `s7_fig_ablation.png`, `s7_fig_curves_100.png` | `run_queue.py`, `run_train.py` |
| S5 데이터 양별 | 25·50% 데이터로 P1·P2·P3·P4a | `S7_report.md` §4.1, `s7_datasize.csv`, `s7_fig_datasize.png`, `s7_fig_datasize_curves.png` | 〃 |
| S6 위치 평가 | FF++ 조작 마스크 정렬 → heatmap 위치 정확도(경계 적중 등) | `S7_report.md` §5, `s6_loc.csv`, `s6_loc_summary.json`, `s6_thresholds.json`, `s6_mask_hist.png` | `s6_align.py`, `s6_prepare.py`, `loc_eval.py` |
| **S7 종합 분석** | 단계별 표, 과적합, 효과 없는 기능의 원인, 평가 함정, 논문 가치 | **`S7_report.md` §0(결론)·§6(원인)·§8(평가 함정)·§9(논문 가치)**, `s7_diag.csv`, `s7_leakage.json` | `analyze.py` |

### B. 후속 원인 검증 (S8~S10) — 모두 `S7_report.md`에 이어서 정리

| 단계 | 한 일 | 결과 정리 파일 | 주요 코드 |
|---|---|---|---|
| S8 | 분기를 SBI 데이터로 학습(가설: baseline이 안 본 데이터면 나아질 것) → 반증 | `S7_report.md` §10, `s8_tables.md`, `s8_deltas.csv`, `s8_fig_compare.png`, `s8_fig_curves.png` | `run_train_sbi.py`, `analyze_s8.py` |
| S9 | baseline 상단(10.3M)까지 풀어 학습 → P0 못 넘음, 분기가 baseline을 보호함 발견 | `S7_report.md` §11, `s9_tables.md`, `s9_fig_compare.png` | `basetop.py`, `run_train_unfreeze.py` |
| S10 | 새 교차 데이터셋 DFD·FaceShifter 평가 → 효과가 데이터셋마다 반대 | `S7_report.md` §12, `s10_tables.md`, `s10_bootstrap.md`, `s10_eval.csv`, `s10_fig_compare.png`, `s10_gate.json` | `face_align.py`, `build_newdata_cache.py`, `eval_newdata.py`, `s10_bootstrap.py`, `analyze_s9_s10.py` |

### C. 자체 개선 1차 (TASKS4) — 계획·로그 `TASKS4.md`, 규칙 `PROTOCOL.md` §17

| 내용 | 결과 정리 파일 | 주요 코드 |
|---|---|---|
| 선택 세트(FF++ val + DFD-val, 배우 신원 분리)·점수 은행 | `TASKS4.md` A1·A2 | `t34_common.py`, `t34_scorebank.py` |
| 지금까지 기능 중 최고 조합 | `reports/t3_best_combo.md` | `t34_select.py` |
| **약점 기반 개선 7종 종합**(AdaBN, 경계 증거, GMM 기준값, 상보 결합, anchor 학습, 혼합 학습, 결합 방식) | **`reports/t4_improvements.md`** | – |
| 개선 ①~④ 상세 표 | `t4_results.md`, `t4_ensemble.md`, `t4_fusion.md` | `t4_methods.py`, `t4_fusion.py` |
| anchor 학습(위치 보존 ↔ 일반화) | `t4_improvements.md` §5, `t4_fig_anchor.png` | `run_train_anchor.py` |
| 혼합 학습 | `t4_improvements.md` ⑥ | `run_train_mix.py`, `t4_eval_new.py` |

### D. 자체 개선 2차 (TASKS5) — 계획·로그 `TASKS5.md`, 규칙 `PROTOCOL.md` §18

| 내용 | 결과 정리 파일 | 주요 코드 |
|---|---|---|
| **종합 보고서**(방법·결과·대조·ACC·논문 가치·한계) | **`reports/t5_final.md`**, 자동 표 `t5_final_tables.md` | `t5_final.py`, `t5_models.py`, `t5_common.py` |
| M1 전문가 가중치 병합(최종 개선) | `t5_merge.md`, 그림 `t5_fig_merge.png` | `t5_merge.py`, `t5_soup.py`, `t5_fig.py` |
| M2 기준값 보정(ACC) | `t5_calib.md` | `t5_calib.py` |
| M3 전문가 증류 | `t5_final.md` §3.4 | `run_train_distill.py`, `t5_eval_distill.py` |
| M5 좌우 반전 TTA | `t5_final_tables.md` §5 | `build_flip_eval_cache.py`, `t5_tta.py` |
| **AUC·ACC 외 가치**(위치·과적합·재현성·오탐·위조 방식별) | **`reports/t5_beyond_auc.md`**, 자동 표 `t5_beyond_tables.md` | `t5_beyond.py`, `t5_loc.py` |

### E′. 추가 향상 탐색 (TASKS6) — 계획·로그 `TASKS6.md`, 규칙 `PROTOCOL.md` §19

| 내용 | 결과 정리 파일 | 주요 코드 |
|---|---|---|
| **종합 보고서**(8가지 방법, 선택 과적합, 분해, 권장 A/B) | **`reports/t6_final.md`**, 자동 표 `t6_final_tables.md` | `t6_final.py` |
| 학습 없는 병합 확장(모듈별·TIES·DARE·다중 전문가·하이브리드) | `t6_sel.json`, `t6_merge_choice.json`, `t6_hyb_choice.json` | `t6_merge.py` |
| SBI 전문가 개선 학습(Xp3·Xema·Xlr1) | `t6_texp_sel.json`, `t6_texp_choice_*.json` | `run_train_sbi2.py`, `t6_texp.py` |
| 영상 점수 집계 방식 | `t6_agg.md` | `t6_agg.py` |

### E. 논문·발표 문서 (`paper/`)

| 파일 | 내용 |
|---|---|
| `paper/draft_v1.md` | 논문 초안 v1 — 분석 중심(아래 §5) |
| `paper/draft_v2.md` | 논문 초안 v2 — 개선 중심(v1 + 전문가 병합, **추천**) |
| `paper/value_assessment.md` | "성능 향상이 크지 않은 결과"를 논문·발표로 내도 가치 있는가(결론: 있음) |
| `paper/presentation_highlights.md` | 발표용 성과 분석: 실제로 오른 수치만, 크게 보이게 하는 정당한 표현, **쓰면 안 되는 방식**, 예상 질문과 답 |
| `paper/final_presentation_outline.md` | 최종 발표 슬라이드 개요 |

### (실험 폴더 밖) 이전 자료
- `C:\m2f2\ppt\`: 1학기 중간·기말, 2학기 중간 발표 PDF, 2학기 중간발표 질문 답변 준비(`질문답변_준비.md`)
- `C:\m2f2\docs\`: 구조도·기말 아이디어(html/pdf), 발표 대본(7~10쪽)
- `C:\m2f2\app\`: 앱 설명(README)과 변경 기록(CHANGES_*.md)

## 4. 처음부터 이해하는 읽는 순서

### 빠른 길 (약 30분, 전체 그림만)
1. 이 파일 §1
2. `HANDOFF.md` §0(한 문단 요약)·§5.2(설정 P0~P4가 무엇인지)
3. `reports/S7_report.md` §0 — 기능 추가 실험의 결론
4. `reports/t5_final.md` §0 — 최종 개선의 결론
5. `paper/presentation_highlights.md` §1~2 — 발표에서 말할 수치

### 전체 이해 (순서대로)
1. **배경과 설계**
   - `HANDOFF.md` §0~5: 왜 이 실험을 했는지, 1학기에 무엇이 문제였는지(분할 누수 등), 무엇을 고정했는지
   - `PROTOCOL.md` §1~12: 데이터·학습·평가·판정 규칙
2. **기능 추가 실험(S1~S7)**: `reports/S7_report.md` §1~9
   - 숫자는 `s7_tables.md`, 그림은 `s7_fig_ablation.png`·`s7_fig_curves_100.png`
   - 핵심은 §2(단계별 표), §3(효과 판정), §5(위치), §6(원인), §8(평가 함정)
3. **후속 검증(S8~S10)**: `S7_report.md` §10~12 → `s10_bootstrap.md`
4. **개선 1차(TASKS4)**: `t3_best_combo.md` → `t4_improvements.md`
   - 상세가 필요하면 `t4_ensemble.md`·`t4_fusion.md`
5. **개선 2차(TASKS5)**: `t5_final.md` → `t5_merge.md`(+ 그림) → `t5_calib.md` → `t5_beyond_auc.md`
6. **추가 탐색(TASKS6)**: `t6_final.md` → `t6_final_tables.md`
7. **정리 문서**: `value_assessment.md` → `draft_v1.md` → `draft_v2.md` → `presentation_highlights.md` → `final_presentation_outline.md`
8. **재현·검증이 필요할 때**: `PROGRESS.md`(결정·변경 기록, 실행 로그), `PROTOCOL.md` §14~18, `registry.csv`, `runs/`, `code/`

### 용어 미리 알기
- **P0**: 논문 모델(M2F2-Det 공개 가중치, 고정)
- **P1~P4**: P0 위에 우리 기능을 단계적으로 더한 설정
  - P2: heatmap Top-3 crop
  - P3: P2 + 패치 정규화 + 동적 결합 = **현재 앱 모델**
  - P4a: P3에서 detach만 제거
- **SBI**: 진짜 영상만으로 합성한 가짜(Self-Blended Images)
- **U3s / U3f**: baseline 상단까지 풀고 SBI / FF++로 학습한 모델
- **선택 세트**: FF++ val + DFD-val. 모델·가중치 고르기에만 쓴다.
- **보고 세트**: FF++ test, Celeb-DF, FaceShifter, DFD-test. 결과 보고에만 쓴다.
- **✔**: 영상 단위 짝 bootstrap 95% 신뢰구간이 0 밖 = 확실한 차이
- **bACC**: 균형 정확도 = (진짜 정답률 + 가짜 정답률)/2

## 5. 논문 초안의 방향

**두 버전 모두 실제로 측정한 결과만 쓴다.** "성능이 오른 것처럼 꾸민 버전"은 만들지 않았다. 테스트로 고르기, 좋은 seed만 보고, 불리한 결과 숨기기는 질문 한 번에 무너지고 연구 윤리 문제가 되기 때문이다(`presentation_highlights.md` §5). 두 버전의 차이는 **무엇을 중심 성과로 내세우는가**다.

| | v1 `draft_v1.md` — 분석 중심 | v2 `draft_v2.md` — 개선 중심 (**추천**) |
|---|---|---|
| 제목 방향 | 고정 탐지기 위 국소 분기의 기여 **재평가** | 국소 분기의 재평가 **+ 학습 데이터 전문가 가중치 병합을 통한 일반화** |
| 중심 메시지 | "우리 기능은 단독으로 논문 모델을 넘지 못했다(인정). 대신 원인과 평가 함정을 밝혔다." | "분석에서 찾은 원인으로 **학습 없이 단일 모델을 개선**했다(교차 데이터셋 3곳 확실한 향상)." |
| 성능 향상 | 모델 2개 출력 결합: DFD +4.8, FaceShifter +5.8(P0 대비) | 병합 단일 모델: Celeb-DF +2.6, FaceShifter +4.1, DFD +5.3(P3 대비), 출력 결합·혼합 학습·증류·WiSE-FT형과 비교 |
| 음성 결과 | 본문 중심 | 그대로 포함(§5.1~5.6, §5.8) |
| 어울리는 곳 | 재현성·평가 트랙, "분석 논문" | 학부생 학술대회(KCC·KSC) 일반 트랙 |
| 주의 | 개선이 앙상블뿐이라 "새로운 것"이 약함 | 병합 기법 자체는 기존 것(model soup·WiSE-FT·task arithmetic·딥페이크용 R²M) → 기여를 "데이터 종류가 다른 전문가 병합의 효과·원인 규명"으로 한정 |

- 솔직하게 써도 가치가 있는지에 대한 판단은 `paper/value_assessment.md`에 따로 정리했다.
- 발표에서 수치를 돋보이게 하는 정당한 방법(상대 오류 감소율, 비용 대비, 불균형 세트는 bACC 등)은 `paper/presentation_highlights.md` §3에 있다.
