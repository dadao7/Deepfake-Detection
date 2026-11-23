# 실험 프로토콜 (PROTOCOL)

DeepSight 통제 비교 실험의 **실행 규칙 전체**를 적는다. 근거·배경은 `HANDOFF.md` §4~5(사용자 승인 완료), 진행 상황은 `PROGRESS.md`에 있다.
이 문서의 모든 항목은 **모든 설정에서 똑같이** 적용한다. 바꾸면 이 문서의 §12(변경 이력)와 `PROGRESS.md` "결정·변경 기록"에 사유와 영향을 함께 적는다.

- 작성: 2026-10-15 (S0). 상태 표기: **[확정]** = 이대로 실행, **[S2 확정]/[S3 확정]/[S6 확정]** = 해당 단계에서 원본 동작을 확인한 뒤 이 문서에 확정 기록.

---

## 1. 목적

저자 공개 가중치 M2F2-Det를 **고정**하고, 우리가 얹은 기능(세부 crop 분기 → heatmap/Top-K → 국소 패치 정규화·동적 결합 → detach 제거·seg head)을 한 단계씩 더하면서
**같은 데이터·학습법·epoch·seed·평가**로 성능을 재어 각 기능의 실제 기여도를 밝힌다. 효과가 없으면 이유를 분석한다. 최종 산출물은 논문용 단계별 ablation 표.

## 2. 고정 변수 (HANDOFF §5.1 전체) [확정]

| 변수 | 값 |
|---|---|
| 기본 모델 | 저자 `C:\m2f2\weights\current_model_180.pth` + `vision_tower.pth`, **고정(eval 모드, 학습 안 함)**. 로드 방식은 `ref/p0_eval.py`와 같음(`torch.load(weights_only=False)` 패치 + 키 remap) |
| 데이터 | FF++ c40 **공식 분할**(`M2F2_Det/utils/FFPP_split/{train,val,test}.json`: 360/70/70 쌍) |
| 프레임 | 학습 영상당 **10장**, 검증 영상당 **8장**(균등 간격, 고정 인덱스를 파일로 저장), 테스트 영상당 **첫 100프레임**(저자 방식) |
| 클래스 | 진짜:가짜 = **1:1 균형 샘플링** (규칙 §3.4) |
| 학습 | AdamW lr 5e-5, wd 1e-3, cosine(T_max=epoch 수, eta_min 1e-6, epoch마다 step), 배치 16, AMP(autocast fp16)+GradScaler, grad clip 1.0, 손실 CE(결합)+0.15·CE(detail), K=3, crop 64, **epoch 20**(S2 시범에서 확정) |
| seed | 0, 1, 2 (**초기값·순서·증강만** 변함, §6.3) |
| 모델 선택 | **검증 손실 최저 epoch(best) + 마지막 epoch(last) 둘 다** 테스트·기록 |
| 데이터 양 | 학습 쌍 25% / 50% / 100%, **작은 집합이 큰 집합에 포함**(nested, §3.3) |
| 평가 | 영상 점수 = 프레임 확률에서 높은·낮은 10개씩 뺀 평균(저자 방식), **fp16(autocast)** |
| 지표 | 주: 영상 AUC. 실사용: ACC@0.5, 균형 정확도, 오탐률(진짜→가짜)·미탐률(가짜→진짜). 논문 비교용: 최적기준 ACC(따로 표시). 보조: 방식별 AUC, 프레임 AUC. **라벨 귀속 버그 수정판**(§8.1) |
| 교차 데이터셋 | Celeb-DF v2 테스트 518영상, 영상당 전 프레임(3~32장, 평균 31.7), 양쪽 10% 뺀 평균(§8.2) |
| 기록 | epoch별 train/val loss·AUC·ACC CSV, 설정 파일, 코드·분할·캐시 해시, 라이브러리 버전, 테스트 예측값 전체, `registry.csv`에 실행 1회 1줄(§9) |
| crop 크기 | **모든 설정 64** |

## 3. 데이터 규칙

### 3.1 영상 목록 [확정]
- 진짜 영상: 각 쌍 `[a, b]`의 `a`, `b` (train 720 / val 140 / test 140).
- 가짜 영상: 방식 m ∈ {Deepfakes, Face2Face, FaceSwap, NeuralTextures}마다 `a_b`, `b_a` (train 2,880 / val 560 / test 560).
- 원천: train·val은 `C:\m2f2\data\FF++\FF++_{종류}_c40.h5`, test는 `C:\m2f2\data\test_only\FF++_{종류}_c40_test_only.h5`(P0과 같음. S1에서 전체 h5와 프레임이 같은지 표본 대조).
- 라벨: 진짜 0, 가짜 1.

### 3.2 프레임 인덱스 [확정]
- h5 영상 프레임 수 n(100 또는 270). 학습 k=10, 검증 k=8: **i_j = floor((j + 0.5)·n / k), j = 0..k−1** (k등분 구간의 가운데).
- 테스트: 0..99 (첫 100프레임).
- Celeb-DF: `{Celeb-real,Celeb-synthesis,YouTube-real}/frames/{영상}/*.png` 전부, 파일 이름의 숫자 순서.
- 모든 인덱스는 `cache/index/`에 파일로 저장하고 해시(index_sha)를 기록한다.

### 3.3 데이터 양(nested) [확정]
- 학습 360쌍을 고정 순열(`numpy.random.default_rng(2026)`)로 섞은 뒤 **앞 90쌍 = 25%, 앞 180쌍 = 50%, 360쌍 = 100%**. 쌍 단위라 진짜(a, b)와 그 가짜(a_b, b_a ×4방식)가 함께 들어간다.
- 검증·테스트는 데이터 양과 무관하게 항상 전체.

### 3.4 1:1 균형 스케줄 [확정]
학습 풀(데이터 양 d)의 진짜 프레임 R장(=쌍 수×2×10), 가짜 프레임 4R장(방식마다 R장).
- **매 epoch: 진짜 프레임 전부(R장) + 가짜 프레임 R장(방식마다 R/4장)** → 1 epoch = 2R장 = 2R/16 스텝 (100%: 14,400장·900스텝, 50%: 450스텝, 25%: 225스텝. 모두 나누어떨어짐).
- 가짜 프레임은 4개 묶음(chunk 0~3)으로 나누고 **epoch e(1부터)는 chunk (e−1) mod 4**를 쓴다 → 4 epoch마다 모든 가짜 프레임을 정확히 한 번, 20 epoch에 5번 본다(진짜는 20번).
- 묶음 배정(영상 단위 균형): 방식마다 가짜 영상을 (쌍 순열 순위, a_b→b_a) 순으로 놓고 순위 r을 매긴다. 영상 r의 10프레임을 영상별 고정 순열(시드 = 영상 키의 해시)로 섞은 위치 p에 대해 **chunk = (p + r) mod 4** → 영상마다 epoch당 2~3장, 방식·묶음마다 정확히 R/4장. 순위가 쌍 순열에서 오므로 **25% 스케줄은 50%·100% 스케줄의 부분집합**(nested 유지).
- 이 스케줄은 **seed와 무관하게 고정**하고 `cache/index/schedule_d{25,50,100}.npz`로 저장한다(seed는 epoch 안의 순서만 바꾼다).

### 3.5 증강 [확정, 1학기와 다름 → §10]
- 학습만 적용. 샘플마다: 좌우반전 p=0.5 → 밝기 U(0.8,1.2) → 대비 U(0.8,1.2) → 채도 U(0.9,1.1) (1학기와 같은 분포·순서, torchvision 공식과 같은 수식).
- **좌우반전**: baseline 출력을 원본·반전 **양쪽 모두 캐시**해 두고 반전된 이미지와 반전 캐시를 함께 쓴다 → 원본 방식과 정확히 같음.
- **색 변형**: baseline 출력은 캐시(색 변형 전 이미지 기준)를 쓰고, 색 변형은 **224 이미지에 적용한 뒤 crop을 잘라 detail 분기에만** 들어간다. (1학기는 색 변형된 이미지가 baseline에도 들어갔음. 차이는 S2 시범에서 측정 → §10.)
- 검증·테스트는 증강 없음.

## 4. 공통 모델 구성 요소

### 4.1 baseline 출력 캐시 [확정]
- 저자 M2F2Det(eval, autocast fp16)로 프레임마다 **logit(2, fp32 저장)**과 **b_3 = EfficientNet-B4 `features[6]` 출력(272×11×11, fp16 저장)**을 한 번만 계산해 `cache/base/`에 저장. 학습 프레임은 좌우반전본도 저장.
- 입력 처리는 저자 코드 그대로: [0,1] 텐서 → `uint8(x·255)`(버림) PIL → CLIPImageProcessor(336 리사이즈·center crop·CLIP 정규화) → CLIP·EfficientNet 공통 입력. 그래서 b_3가 11×11.
- 관찰(원본 코드 읽음, S1에서 수치 확인): 저자 `bridge_adapter`는 `nn.ModuleList`가 아닌 list라 가중치가 저장·로드되지 않고(무작위 초기화) `eval()`도 전파되지 않지만, 뒤의 `BridgeAdapter_Proj`가 `LayerNorm(1)`이라 출력이 항상 bias 상수 → **입력과 무관, 예측에 영향 없음**.

### 4.2 위치 맵 [확정]
- **heatmap**: `HeatmapHead`(원본 구조: Conv272→64(3×3)→BN→ReLU→Conv64→1(1×1), 마지막 bias −2.19) → hm_logit(1×11×11), hm_prob = σ(hm_logit). detach 있음(P2~P3): `σ(hm_logit.detach())`, 없음(P4a): `σ(hm_logit)`.
- **특징 평균 맵**(P1, P1′, P2b): σ(b_3의 채널 평균, fp32) (11×11). σ는 단조라 최댓값 위치는 채널 평균과 같고, NMS에서 억제된 0보다 항상 커지도록 쓰는 것.

### 4.3 peak와 crop [확정, 원본 `topk_crop.py`와 동일]
- NMS: 커널 max(3, 2·min_distance+1)=5 (min_distance 2)의 max_pool로 국소 최대만 남김 → 상위 k개(`torch.topk`), 부족하면 마지막 값 반복.
- 좌표: peak (y, x)(11×11) → 이미지 좌표 (int(y·224/11), int(x·224/11)) → 64×64 상자를 그 점에서 위·왼쪽으로 32만큼 뺀 위치에 두고 경계에서 안쪽으로 밀어 넣음 → bilinear 리사이즈(64→64, 사실상 복사).
- 구현은 벡터화하되 **원본 `crop_by_heatmap_topk`와 출력이 같은지 S2에서 검증**(게이트 ②).

### 4.4 detail 분기·집계·결합 [확정]
- detail 분기: 원본 `DetailCropBranch`(torchvision DenseNet121 ImageNet 가중치 features → avgpool → 1024→512→ReLU→Dropout0.2→2). 입력은 [0,1] crop(정규화 없음, 원본과 같음).
- 집계: k=1이면 그 crop의 logit. heatmap Top-3는 원본 'weighted'(peak 확률 가중 평균). 특징 평균 Top-3(P2b)는 **단순 평균**(특징 평균 맵 값은 확률이 아니므로).
- 결합: fused = α·baseline_logit + (1−α)·detail_logit. 고정 α = 0.5, 동적 α = MLP(입력 [baseline_logit(2), peak_vals(K)] → 16 → ReLU → 1 → Sigmoid), 원본 `gating_network`와 같은 구조.

### 4.5 국소 패치 정규화(Method 4) [확정, 원본과 같은 논리]
학습 중에만, 배치 전체에 한 번씩 결정(Python `random`): p=0.5 가우시안 블러(커널 3 또는 5, σ~U(0.1,2.0)) → p=0.5 가우시안 노이즈(std~U(0.01,0.04), [0,1] clamp) → p=0.3 `F.dropout(p=0.1)`(살아남은 값 ×1/0.9, 원본 그대로).

### 4.6 heatmap head의 BatchNorm [S2 확정]
- 원본 동작(검증함, `logs/verify_s2_unit.json`): `model.train()`이라 heatmap head의 BN이 학습 모드 → 학습 중에는 배치 통계로 정규화하고 running 통계가 갱신된다(1학기 SBI 가중치에서 한 번의 forward로 running_var 최대 3,176 변화). detach가 있으면 heatmap head 가중치에는 gradient가 없어(P3: grad None) 가중치·γ·β는 그대로다. 평가 때는 갱신된 running 통계를 쓴다.
- **확정: 원본 동작을 그대로 재현**(P2a·P2·P3a·P3b·P3·P4a 모두 같은 처리). ExpModel의 train 모드 출력·BN 통계가 원본 `M2F2DetImproved`와 비트 단위로 같음을 확인(게이트 ①). 따라서 P3의 분류용 heatmap은 seg head와 **BN 통계만** 다르고, P4a는 가중치까지 분류 손실로 바뀐다.

### 4.7 seg head [S3 확정]
SBI 사전학습 직후의 heatmap head를 복사해 **가중치·BN 통계 모두 고정(항상 eval)**. 분류에는 쓰지 않고 위치 평가(§11)에만 쓴다. baseline이 고정이라 b_3가 모든 설정에서 같으므로, SBI 마스크로 따로 학습한 seg head와 같다.

### 4.8 SBI 사전학습 [S3 확정 — 규칙]
- **생성기**(`code/sbi.py`, 공식 SBI 구현의 변환을 numpy/cv2로 재작성): 81점 랜드마크 볼록껍질 마스크(25% 확률로 앞 68점만) → source·target 중 한쪽(각 50%)에 원천 변환[RGBShift ±20(p=0.3), HSV ±0.3(공식 설정, uint8에선 거의 무변화), 밝기·대비 ±0.1, 다운스케일(×1/2 또는 ×1/4, NEAREST→LINEAR) 또는 샤픈(α 0.2~0.5, 밝기 0.5~1.0) 중 하나] → source·마스크 affine(이동 x ±3%, y ±1.5%, 배율 0.95~1/0.95) + 마스크만 elastic(α 50, σ 7) → 블렌드 마스크(192~256 리사이즈, 가우시안 블러 k1 ∈ {5,7,…,25}, 1 미만 0, 다시 블러 k2·σ 5~45, 정규화) → 비율 r ∈ {0.25, 0.5, 0.75, 1, 1, 1}로 SBI = r·M·source + (1−r·M)·target, **진짜 = (변환됐을 수 있는) target**(공식과 같음) → 쌍 공통 증강(같은 계수: RGBShift·HSV·밝기대비 각 p=0.3, 좌우반전 p=0.5, JPEG 품질 40~100 p=0.5).
- **데이터**: 공식 train 원본 중 랜드마크가 있는 523영상의 랜드마크 프레임 14,077장(h5 프레임 10k) × 변형 4 = 56,308쌍, 공식 val 원본 88영상 880쌍(변형 1). 표본 난수 = `default_rng([20261015, 영상번호, k, 변형])`(결정적). 모든 SBI·진짜 이미지의 baseline 출력(b_3, logit)을 캐시(`cache/sbi/`).
- **heatmap GT**: 경계 B = 4·M·(1−M)를 **블렌드 비율을 곱하기 전 마스크 M(최대 1)**으로 계산 → 11×11(INTER_AREA) → 원본 `heatmap_loss.generate_heatmap_gt`(4사분면 최대점, 적응 σ). 진짜는 전부 0. (비율을 곱한 마스크로 하면 r ≤ 0.5에서 얼굴 내부 전체가 최대가 되어 경계가 아니게 됨 → 격자 그림으로 확인 후 결정)
- **학습**(`code/pretrain_sbi.py`, 한 번만, seed 0): heatmap head(모듈 seed 규칙 §6.3) + detail 분기(ImageNet DenseNet121 + 무작위 head). 손실 = 원본 focal loss(alpha 0.25, fp32) + 1.0·CE(detail). detail 입력 = GT peak 위치(GT에 NMS top-3, 값 ≥ 0.5)의 64 crop: SBI(가짜) vs **같은 위치의 진짜**. AdamW lr 1e-4, wd 1e-3, cosine 10 epoch(eta_min 1e-6), 배치 16쌍(32장), AMP+GradScaler, clip 1.0. epoch e는 프레임마다 변형 (e−1) mod 4를 쓴다. heatmap head BN은 학습 모드.
- **선택**: SBI val의 (focal + CE) 최저 epoch → `runs/sbi_pretrain/sbi_best.pt`(설정 표의 init='sbi'). 1학기 `pretrained_heatmap_detail.pth`는 S2 시범("원본 방식" 재현)에서만 읽기 전용으로 사용.
- **SBI val 위치 지표**(S3 게이트 기록용): 실제 heatmap 최댓값의 진짜/가짜 AUC, top-1 peak 셀의 GT ≥ 0.5 비율(peak 적중), peak 셀의 경계값 B 평균, 예측 top-3 crop 중 경계(B > 0.5) 픽셀을 포함하는 비율(crop 적중), 얼굴 중앙 셀 기준선.

## 5. 설정 정의 (HANDOFF §5.2 전체) [확정, 세부 해석은 §4]

| 설정 | 구성 | 위치 맵 | k | 집계 | α | Method 4 | 초기화 | heatmap detach | 학습 횟수 |
|---|---|---|---|---|---|---|---|---|---|
| **P0** | 논문 모델(baseline만) | – | – | – | – | – | 저자 가중치 | – | 평가만(완료, S1에서 캐시로 재확인) |
| **P1** | + 세부 crop 분기 | 특징 평균 | 1 | 단일 | 0.5 | 없음 | detail: ImageNet DenseNet + 무작위 head (**SBI 없음**) | – | 100%×3 + 25/50%×3 |
| **P1′** | P1 + SBI 초기화 | 특징 평균 | 1 | 단일 | 0.5 | 없음 | detail: 새 SBI | – | 100%×3 |
| **P2a** | + heatmap만 | heatmap | 1 | 단일 | 0.5 | 없음 | heatmap·detail: 새 SBI | 있음 | 100%×3 |
| **P2b** | + Top-3만 | 특징 평균 | 3 | 평균 | 0.5 | 없음 | detail: 새 SBI | – | 100%×3 |
| **P2** | + heatmap + Top-3 | heatmap | 3 | peak 가중 | 0.5 | 없음 | heatmap·detail: 새 SBI | 있음 | 100%×3 + 25/50%×3 |
| **P3a** | P2 + 패치 정규화만 | heatmap | 3 | peak 가중 | 0.5 | 있음 | 새 SBI | 있음 | 100%×3 |
| **P3b** | P2 + 동적 결합만 | heatmap | 3 | peak 가중 | 동적 | 없음 | 새 SBI + MLP 무작위 | 있음 | 100%×3 |
| **P3** | P2 + 둘 다 (= 현재 모델) | heatmap | 3 | peak 가중 | 동적 | 있음 | 새 SBI + MLP 무작위 | 있음 | 100%×3 + 25/50%×3 |
| **P4a** | P3 + **detach 제거만** | heatmap | 3 | peak 가중 | 동적 | 있음 | 새 SBI + MLP 무작위 | **없음** | 100%×3 + 25/50%×3 |
| **P4b** | P3 + 고정 seg head | – | – | – | – | – | – | – | 추가 학습 없음. 분류 = P3, 위치 평가만 다름 |
| **P4** | P4a + 고정 seg head | – | – | – | – | – | – | – | 추가 학습 없음. 분류 = P4a |

- P4a에서 바꾸는 것은 detach 하나뿐(희소성 손실·학습형 α 등 다른 변경 금지). gradient는 peak 값(집계 가중치·동적 결합 입력)을 통해서만 heatmap head로 간다(crop 위치는 이산이라 미분 불가).
- P2b 해석: "heatmap 없이 Top-3 다중 crop만" = P1의 특징 평균 맵에서 NMS 상위 3개 + 단순 평균. 따라서 P2b→P2 차이 = **학습된 위치(heatmap) + 신뢰도 가중**, P2a→P2 차이 = **crop 수(1→3) + 가중 집계**. 원본 설계에서 가중치는 heatmap 확률에 묶여 있으므로 이렇게 나눈다.
- 학습 수: 100% 9설정×3seed = 27회, 25·50% 4설정(P1·P2·P3·P4a)×2×3seed = 24회, 합계 **51회**. 실행 이름 `{설정}_d{25|50|100}_s{seed}` (P1′ → `P1p`).

## 6. 학습 절차 [확정]

### 6.1 반복
- 학습 대상 파라미터: 설정에 쓰이는 모듈(detail 분기, heatmap head, 동적 MLP)만 AdamW에 넣는다. gradient가 없는 파라미터(detach된 heatmap head 등)는 AdamW가 건너뛰므로 weight decay도 받지 않음(원본과 같음).
- 스텝: autocast fp16 forward → loss = CE(fused.float(), y) + 0.15·CE(detail.float(), y) → scaler.scale(loss).backward() → unscale → clip_grad_norm_(학습 파라미터, 1.0) → scaler.step/update. epoch 끝에 scheduler.step().
- 모드: 학습 중 baseline은 캐시(항상 eval 결과), 학습 모듈은 train 모드(heatmap BN은 §4.6), 검증·테스트는 eval 모드.

### 6.2 검증 (매 epoch)
- 검증 전체(진짜 140×8 = 1,120장, 가짜 560×8 = 4,480장), eval 모드, autocast fp16.
- **val_loss_bal** = ½·(진짜 프레임 평균 CE + 가짜 프레임 평균 CE) — 결합 logit 기준. 학습의 1:1 사전확률과 맞춘 것. **모델 선택(best)은 이 값 최저 epoch**(같으면 앞 epoch).
- 함께 기록: val_loss_raw(가중 없음), val_loss_detail, 프레임 AUC, 영상 AUC(8장 평균), ACC@0.5, 균형 ACC@0.5.

### 6.3 seed와 난수 분리
- **데이터 난수**(epoch 안 순서, 반전 여부, 색 변형 계수)는 `numpy.random.default_rng([seed, epoch, 용도])`로 만들어 **설정과 무관하게 seed가 같으면 모든 설정이 같은 배치·같은 증강**을 본다(공통 난수 → 설정 간 비교 분산 감소).
- **모델 난수**(초기화, dropout, Method 4 결정·노이즈)는 `torch.manual_seed(seed)`, `random.seed(seed)`.
- cuDNN deterministic, benchmark 끔. **같은 seed 재현성 확인(게이트 ④)**: P4a를 2 epoch × 40스텝으로 두 번 학습 → 파라미터 742개·옵티마이저 상태·곡선이 비트 단위로 같음(`use_deterministic_algorithms`는 필요 없었음).
- 모듈 초기화는 모듈마다 `torch.manual_seed(seed·1000 + 모듈번호)`(heatmap 11, detail 22, gating 33) → 같은 seed면 설정이 달라도 같은 모듈은 같은 초기값(예: P3와 P4a는 완전히 같은 가중치에서 출발).

### 6.4 이어받기
- 매 epoch 끝에 `runs/{id}/resume.pt`(학습 모듈·optimizer·scaler·scheduler·난수 상태·지금까지의 곡선)를 덮어쓴다. 중단되면 마지막 완료 epoch 다음부터 재개. 데이터 순서는 (seed, epoch)로 결정되므로 재개해도 같다.
- `runs/{id}/DONE`이 있고 registry에 줄이 있으면 큐 러너가 건너뛴다.

## 7. 모델 선택 [확정]
best(val_loss_bal 최저 epoch)와 last(마지막 epoch) 체크포인트를 **둘 다** FF++ 테스트·Celeb-DF로 평가하고 둘 다 기록한다. 본문 표는 best를 주로 쓰고 last를 함께 보인다(테스트로 고르지 않는다).

## 8. 테스트 평가와 지표 [확정]

### 8.1 FF++ c40 공식 test
- 700영상(진짜 140 + 가짜 4×140) × 첫 100프레임, 프레임 확률 p = softmax(fused)[가짜], autocast fp16, 캐시 사용.
- 영상 점수 = 정렬한 100개에서 높은·낮은 10개를 뺀 80개 평균. **라벨은 영상 키로 귀속**(저자 코드는 영상이 바뀔 때 다음 영상 라벨을 붙이고 마지막 영상을 빠뜨리는 버그 → 수정판 사용, 저자식 값은 P0에서만 참고로 병기).
- 지표: 영상 AUC(주), ACC@0.5, 균형 ACC@0.5, 오탐률 = 진짜 중 p≥0.5 비율, 미탐률 = 가짜 중 p<0.5 비율, 최적기준 ACC(0~1, 0.0001 간격으로 **테스트에서** 고른 기준값 → 논문 비교용으로만 따로 표시), 방식별 AUC(진짜 140 + 그 방식 140), 프레임 AUC(70,000장).

### 8.2 Celeb-DF v2 (교차 데이터셋)
- `List_of_testing_videos.txt` 518영상. **목록 라벨 1 = 진짜, 0 = 가짜 → 우리 라벨 가짜 = (목록 라벨 == 0)** (진짜 178, 가짜 340).
- 프레임 전부(3~32장) RGB, 256→224 `cv2.resize(INTER_LINEAR)`(1학기 평가와 같은 리사이즈), autocast fp16, 캐시 사용.
- 영상 점수 = 정렬 후 양쪽에서 floor(0.1·n)개씩 뺀 평균(n=32면 3개씩, n<10이면 빼지 않음. FF++의 100→10과 같은 비율).
- 지표: AUC(주), ACC@0.5, 균형 ACC@0.5, 오탐률, 미탐률, 최적기준 ACC(따로 표시).

## 9. 기록

### 9.1 실행 폴더 `runs/{id}/`
| 파일 | 내용 |
|---|---|
| `config.json` | 설정 옵션 전체, seed, 데이터 양, epoch, 하이퍼파라미터, code_sha·cfg_sha·index_sha·cache_sha, 라이브러리 버전, GPU, 시작·끝 시각 |
| `curves.csv` | epoch별: lr, train_loss(합계)·train_loss_fused·train_loss_detail·train_acc·train_auc(프레임), val_loss_bal·val_loss_raw·val_loss_detail·val_auc_frame·val_auc_video·val_acc05·val_bacc05, 소요 초 |
| `ckpt_best.pt`, `ckpt_last.pt` | 학습 모듈 가중치(+BN 통계)와 epoch |
| `pred_{best,last}.npz` | FF++ 테스트 70,000프레임·Celeb-DF 16,420프레임의 결합 확률, detail 확률, α, peak 좌표·값 (순서는 `cache/index`와 같음) |
| `metrics.json` | §8 지표 전체(best, last) |
| `train.log`, `DONE` | 로그(큰 로그는 tail만 본다), 완료 표시 |

### 9.2 `registry.csv` (실행 1회 1줄)
열: `run_id, kind(ref/pilot/main), setting, data_pct, seed, epochs, status, best_epoch, best_val_loss, last_val_loss`, 이어서 체크포인트 c ∈ {best, last}마다 `c_ffpp_auc, c_ffpp_acc05, c_ffpp_bacc05, c_ffpp_fpr, c_ffpp_fnr, c_ffpp_accbest, c_ffpp_frame_auc, c_auc_DF, c_auc_F2F, c_auc_FS, c_auc_NT, c_cdf_auc, c_cdf_acc05, c_cdf_bacc05, c_cdf_fpr, c_cdf_fnr`, 끝에 `train_min, started, finished, code_sha, cfg_sha, index_sha, cache_sha, torch, notes`. 값은 % 단위(소수 4자리).
- P0은 `kind=ref` 한 줄(best = last). P4b·P4는 학습이 없으므로 registry 줄 없음(분류 = P3·P4a 줄, 위치 지표는 §11 결과 파일).

### 9.3 해시
- code_sha: 학습·평가 결과에 영향을 주는 `code/common.py, expmodel.py, run_train.py` 내용을 이어 붙인 SHA-256 앞 12자리(2026-10-15 S4 시작 전 정의 변경: 처음엔 `code/*.py` 전체였으나 분석 스크립트 추가만으로 바뀌어 비교에 혼란 → 변경. `P1_d100_s0`·시범·P0 줄은 옛 정의 값). cfg_sha: seed·run_id를 뺀 설정 JSON의 SHA-256(같은 설정이면 seed가 달라도 같음). index_sha: `cache/index/` 파일들. cache_sha: `cache/MANIFEST.json`.
- 원본 무변경 확인: `ref/original_code_sha256.txt`와 대조(S0 시작 시 23/23 일치, 단계마다 재대조해 `logs/`에 남김).

## 10. 1학기 방식과의 차이 (논문 "실험 설정"·"평가 함정" 절 근거)

| 항목 | 1학기 | 이번 | 영향/근거 |
|---|---|---|---|
| 분할 | 영상 이름순 80/20 → 공식 test 140개 중 115개가 학습 쪽 | 공식 분할 | 누수 제거 |
| 클래스 | 가짜:진짜 4:1 그대로 | 1:1 균형 스케줄(§3.4) | 오탐률 편향 완화 |
| 검증 | 영상당 3장 무작위, fp32 | 8장 균등, fp16 | 고정 인덱스, 모든 설정 동일 |
| 모델 선택 | val AUC 최고 | val_loss_bal 최저 + last 둘 다 | 선택 편향 표시 |
| 색 증강 | 전체 이미지(baseline 입력 포함) | crop 입력에만(baseline은 캐시) | **S2 시범에서 차이 측정** |
| 평가 | 저자 코드(라벨 귀속 버그, 테스트에서 고른 기준값 ACC) | 수정판 + ACC@0.5·균형 ACC·오탐/미탐 | 오탐 27% 가림 해소 |
| Celeb-DF | 영상당 3장 평균 | 전 프레임(≤32) 양쪽 10% 뺀 평균 | 저자 방식과 같은 집계 |
| SBI 가중치 | `pretrained_heatmap_detail.pth`(코드 없음) | 새 코드로 재사전학습 | 방법 기술 가능 |

## 11. 위치 평가 (S6) [S6 확정 — 규칙]
- **정렬**(`code/s6_align.py`): 탐색 결과 h5 프레임 i는 위조 c40 영상의 **t = floor(i·T/N)**(균등 추출, N = h5 프레임 수, 보통 100) 프레임이고, h5 얼굴은 **닮음변환(회전 약 ±8°, 배율 약 1.4)으로 정렬된 crop**이다(축 정렬 템플릿 매칭은 실패). 프레임 t에서 1/4 해상도 다중 스케일 매칭으로 ROI → SIFT + RANSAC `estimateAffinePartial2D`(재투영 2px) → 같은 변환으로 마스크 프레임을 224에 옮김(INTER_LINEAR). 실패 시 화면 전체 SIFT로 재시도.
- **품질 기준**: 정렬한 영상 프레임과 h5 프레임의 NCC(가운데 192×192) ≥ 0.95 그리고 인라이어 ≥ 6. 통과 프레임만 평가에 쓰고 통과율을 보고한다(1차 실행 val 97.6%, test 95.1%, NCC 중앙값 1.000).
- **대상 프레임**: val·test의 가짜 영상(방식 4 × 140), 영상당 i ∈ {0, 10, …, 90}(test는 분류 평가와 같은 첫 100프레임 안). 진짜 영상 마스크는 전부 0이라 정렬하지 않는다.
- **이진화 임계값**: 방식별로 **val 정렬 마스크만으로** — 값 1~254 히스토그램(log1p, 5칸 이동평균)의 [4, 100] 구간 최소점(배경과 얼굴 렌더링 값 사이 골짜기). test는 보지 않는다(`code/s6_prepare.py`, 결과 `reports/s6_thresholds.json`).
- **비교 대상**: 분류용 heatmap(heatmap을 쓰는 설정 P2a·P2·P3a·P3b·P3·P4a, seed별 best·last 체크포인트, eval 모드 = 학습으로 갱신된 BN 통계) vs 고정 seg head(P4b·P4 공통, SBI best의 heatmap head) vs 얼굴 중앙 기준선(중앙 셀 / 중앙 crop).
- **지표**(11×11 셀 기준, GT11 = 셀 안 조작 픽셀 비율): ① 셀 AUROC(heatmap 확률 vs GT11 ≥ 0.5, 프레임별 계산 후 평균) ② IoU(σ(heatmap) ≥ τ vs GT11 ≥ 0.5, τ는 모델마다 **val에서 평균 IoU 최대**로 고름) ③ 최댓값 적중률(top-1 peak 셀의 GT11 ≥ 0.5) ④ 경계 적중률(top-1 peak 셀의 GT11 ∈ [0.1, 0.9]) ⑤ crop 적중률(분류와 같은 규칙의 top-3 64 crop 중 하나라도 조작 픽셀 ≥ 50%)·crop 평균 조작 비율. 방식별·전체를 보고, seed 평균 ± 표준편차.
- **보조 지표(S6 준비 중 추가)**: ⑥ 경계 셀 AUROC(heatmap 확률 vs GT11 ∈ [0.1, 0.9]인 셀). 이유: 임계값에서 FF++ 조작 영역이 얼굴 crop의 42~54%를 가운데에 덮어, 얼굴 중앙 기준선이 "안쪽" 지표에서 포화됨(최댓값 적중 99.3%, IoU 0.73, 셀 AUROC 93.6). 경계 AUROC에서는 중앙 기준선 47.7(무작위 수준), 1학기 SBI head 77.0으로 변별된다. SBI head는 블렌딩 경계를 찾도록 학습되므로 경계 지표가 그 목적에 맞는다.
- 결정된 임계값(val): DF 84, FS 7, NT 20, F2F 20 → 조작 면적 DF 42%, FS 51%, NT 54%, F2F 54%. 평가 프레임: val 5,520 / test 5,472(정렬 통과, 재시도 포함 98.6% / 97.7%).

## 12. 판정 기준 (HANDOFF §7)
- "향상"으로 인정: seed 3개 모두 같은 방향 + 향상폭 ≥ seed 표준편차의 2배 + 보통 AUC 1~2%p 이상. FF++는 포화에 가까우므로 Celeb-DF 향상을 더 중시.
- 효과 없는 기능도 그대로 표에 싣고 원인을 분석한다. seg head는 분류 불변이 구조적 사실이므로 위치 지표로만 평가한다.

## 14. 후속 실험 S8 — SBI 학습 프로토콜 (2026-10-16 추가, 사용자 승인: "설계변경이 필요하면 추천방식으로 진행")
- **목적**: S7의 원인 가설("고정 baseline이 FF++ 학습 데이터를 이미 맞추고 있어 같은 데이터로는 분기가 배울 것이 없다")을 직접 검증한다. 분기를 **baseline이 보지 않은 데이터**로 학습했을 때 덧붙인 기능이 P0를 넘는지(특히 Celeb-DF) 본다.
- **바꾸는 것 — 학습 데이터만**: FF++ 진짜/가짜 대신 S3의 SBI 쌍(`cache/sbi/train_*`)을 쓴다. 공식 train 원본 523영상, 14,077프레임 × 변형 4 = 56,308쌍(진짜, SBI 가짜).
  - epoch e는 고정 순열(`default_rng(2027)`, seed 무관)에서 7,200쌍을 차례로 쓴다(`perm[((e−1)·7200 + j) mod N]`). 그래서 진짜 7,200 + 가짜 7,200 = 14,400장 = **900스텝**으로, S4(100%)와 같다. 20 epoch 동안 각 쌍을 약 2.6번 쓴다.
  - 순서는 `default_rng([seed, e, 0])`, 색 변형은 S4와 같은 분포·같은 방식(crop에만)이다. 좌우반전은 SBI 생성 때 쌍에 이미 p=0.5로 적용했으므로 추가하지 않는다.
- **그대로인 것**: 모델·설정 정의(§5), 초기화(새 SBI 사전학습 + 모듈 seed 규칙), AdamW·cosine·배치 16·AMP·clip·손실(CE 결합 + 0.15·CE detail), 20 epoch, seed 0·1·2, 테스트(FF++ test 700영상·Celeb-DF 518영상, §8 지표).
- **모델 선택**: SBI val 880쌍(1,760장)의 val_loss_bal 최저 epoch(best)와 last를 둘 다 보고한다. FF++ val 지표는 epoch마다 기록만 하고 선택에 쓰지 않는다. 따라서 **FF++ 가짜 영상은 학습에도 선택에도 쓰이지 않는다**(고정 baseline만 FF++로 학습돼 있음).
- **설정**: P2(heatmap Top-3, 고정 α), P3(현재 모델), P4a(detach 제거) × seed 3개 = 9회. 실행 이름 `{설정}_sbi_s{seed}`, registry `kind=sbi`.
- **비교**: P0, 같은 설정의 S4(FF++ 학습)와 짝 비교한다(같은 seed). last 대 last는 선택 방식과 무관한 비교다. 판정 기준은 §12와 같다. 위치 평가(§11)는 P3·P4a heatmap에 적용한다.
- **코드**: `code/run_train_sbi.py`(신규, `run_train.py`의 함수를 불러 씀 — 기존 학습 코드는 고치지 않음). S8 code_sha는 CODE_FILES + `run_train_sbi.py`의 해시.

## 15. 후속 실험 S9 — baseline 일부 풀기 (2026-10-16 추가, 사용자 승인: "2,3 진행")
- **목적**: 덧붙인 분기가 아니라 **baseline 자체의 상단**을 학습하면 성능이 달라지는지, 그리고 그때 분기(P3)가 기여하는지 본다.
- **푸는 부분**: EfficientNet-B4 `features[7]`(마지막 MBConv 단, 4.47M) + `features[8]`(1×1 conv 헤드 448→1792, 0.81M) + `deepfake_proj` + `vision_proj` + `clip_vision_alpha` + 출력층(`output`). 합계 약 10.3M.
  - 고정: CLIP 비전 인코더(원본 학습에서도 고정), EfficientNet `features[0..6]`(= 캐시된 b_3), bridge adapter(입력과 무관한 상수 → `cache/base/adapter_const.npy`).
- **캐시로 계산**: baseline logit = output(cat[α_v·vision_proj(CLS), adapter 상수, deepfake_proj(avgpool(f8(f7(b_3))))]). CLS(투영 전 1024차원)는 `build_cls_cache.py`로 캐시한다(같은 배치로 다시 계산한 logit이 기존 캐시와 비트 동일함을 검사).
  - **게이트**: 캐시(CLS, b_3)로 재구성한 logit이 캐시 logit과 일치해야 한다(최대 차 작음, P0 AUC 차 ≤ 0.05).
- **풀린 baseline 모듈은 항상 eval 모드**: BN 통계 고정, stochastic depth 끔. 가중치만 학습한다(작은 배치 미세조정의 표준 처리).
- **설정**:
  - U0 = baseline만(분기 없음, 출력 = baseline logit, 손실 CE)
  - U3 = P3 + 풀린 baseline 상단(손실은 P3와 같음: CE(결합) + 0.15·CE(detail), baseline logit도 결합을 통해 gradient를 받음)
- **학습 데이터**: FF++(S4 스케줄) → U0f·U3f, SBI(S8 스케줄) → U0s·U3s. seed 3개씩, 총 12회.
- **나머지**: 옵티마이저·lr·wd·cosine·배치·AMP·clip·20 epoch·테스트는 S4/S8과 같다. 모든 학습 파라미터에 AdamW lr 5e-5.
- **선택**: FF++ 학습은 FF++ val, SBI 학습은 SBI val의 val_loss_bal 최저(best) + last.
- **코드**: `code/basetop.py`(재구성 모듈), `code/run_train_unfreeze.py`. registry `kind=unfreeze`, 실행 이름 `{U0|U3}{f|s}_s{seed}`.

## 16. 후속 실험 S10 — 새 교차 데이터셋 평가 (2026-10-16 추가, 사용자 승인: "2,3 진행")
- **데이터**(FF++ 공식 서버, c40):
  - DFD(DeepFakeDetection) 전체: 원본(actors) 363 + 가짜 3,068
  - FF++ FaceShifter 공식 test 140 + FF++ 원본 공식 test 140 — FaceShifter는 학습에 없던 위조 방식이다.
  - DFDC는 Kaggle 계정 인증이 필요해 제외했다(이 PC에 인증 없음).
- **얼굴 crop 재현**(`code/face_align.py`): h5 얼굴은 고정 템플릿으로의 닮음변환 정렬이다(정렬 랜드마크의 눈 위치 표준편차 약 2px, 눈 사이 72.5±3.6px로 확인).
  - 처리 순서: dlib HOG 얼굴 검출(긴 변 640으로 줄여 검출, 가장 큰 얼굴) → dlib 68점 랜드마크(원 해상도) → 68점 → 평균 템플릿(train 정렬 랜드마크 평균) 닮음변환(최소제곱) → 224×224.
  - **정렬 점 집합(확정)**: 눈 12점 + 바깥 입술 12점(36~59번). val 위조 16영상 144프레임에서 h5와의 NCC 중앙값 0.982로 가장 높아 선택했다(68점 전체 0.966, 코·눈·입 0.968). test에서도 같은 순위였다(0.994).
  - 얼굴을 못 찾은 프레임은 뺀다. 영상당 유효 프레임이 8장 미만이면 그 영상을 제외하고 수를 보고한다.
- **프레임**: FF++ 계열은 h5 규칙 t = floor(i·T/100), i ∈ {0, 3, …, 93}(32장). DFD는 t = floor((j+0.5)·T/32)(32장).
- **영상 점수**: 양쪽에서 floor(0.1·n)개씩 뺀 평균(Celeb-DF 규칙).
- **재현 검증 게이트**(FF++ test, 우리 crop vs h5 crop, 같은 32프레임):
  - ① 같은 프레임에서 두 crop의 NCC
  - ② P0의 영상 AUC 차이. 우리 crop으로 잰 AUC가 h5 crop으로 잰 AUC와 크게 다르면(> 1%p) 멈추고 보고한다.
- **평가 대상**: P0, S4(9설정×3seed×best/last), S8, S9 체크포인트 전부. 이미지·b_3·logit·CLS를 한 번 캐시한 뒤 평가한다(`code/build_newdata_cache.py`, `code/eval_newdata.py`).

## 17. 후속 작업 TASKS4 — 선택 규칙과 개선 방법 정의 (2026-10-17 추가, 사용자 요청 4가지)
- **세트 분리**: 선택(하이퍼파라미터·가중치·조합 고르기)에는 FF++ val + DFD-val만 쓴다. 보고에는 FF++ test, Celeb-DF, FaceShifter, DFD-test만 쓴다.
  - DFD-val/test는 배우 28명을 신원 기준 14/14로 나눴다(seed 2026, `cache/index/dfd_split.json`). 두 신원이 섞인 가짜 1,357개는 제외했다.
  - 선택 점수 = FF++ val AUC와 DFD-val AUC의 평균.
- **판정**: 보고 세트에서 P0 대비 영상 단위 짝 bootstrap 95% CI(2,000회, 결합 비교는 1,000회)가 0을 벗어나면 "확실한 차이"다. seed가 있는 모델은 seed 3개의 logit 평균으로 비교한다.
- **점수 은행**: 모든 체크포인트의 영상 점수를 `cache/scorebank.npz`에 한 번 저장하고(`code/t34_scorebank.py`, `t4_eval_new.py`), 학습 없는 방법은 이 은행 위에서 계산한다.
- **개선 방법 정의**(`reports/t4_improvements.md`):
  - ① AdaBN: EfficientNet 상단 BN 통계를 대상 세트 통계와 비율 λ로 섞는다.
  - ② 경계 증거: 고정 seg head heatmap 통계를 점수에 가중치 β로 더한다.
  - ③ 기준값: 대상 세트 점수에 2성분 GMM을 맞춰 판정 기준을 정한다.
  - ④ 상보 결합: 전문가 A = P3(FF++ 학습), B = U3s(SBI 학습). logit·순위·z점수 가중 평균과 신뢰도 게이트를 비교한다(`code/t4_fusion.py`). 순위·z점수는 평가 세트 전체가 필요하다(transductive).
  - ⑥ 혼합 학습: 배치를 FF++ 8 + SBI 8로 구성한다. 선택은 (FF++ val + SBI val)/2이고, 나머지는 S9와 같다(`code/run_train_mix.py`, kind=mix).
  - ⑤ anchor 학습: P4a 설정(detach 제거)의 손실에 λ·mean((σ(heatmap) − σ(seg head heatmap))²)를 더한다. seg head = `runs/sbi_pretrain/seg_head.pt`(고정, 학습에만 사용). λ ∈ {1, 10, 100}은 seed 0으로 시험해 선택 세트로 고르고, 고른 λ에 seed 1·2를 추가한다. 나머지는 S4와 같다(`code/run_train_anchor.py`, kind=anchor, 실행 이름 `P4c_l{λ}_s{seed}`).
- 위치 지표는 §11과 같다(`code/loc_eval.py`에 P4c 추가).

## 18. 후속 작업 TASKS5 — P3를 넘는 개선 (2026-10-17 추가, 사용자 요청 "P3보다 좋은 성능, ACC·AUC 향상")
- **세트·판정**: §17과 같다(선택 = FF++ val + DFD-val 평균 AUC, 보고 = FF++ test·Celeb-DF·FaceShifter·DFD-test, 영상 짝 bootstrap 95% CI). 비교 기준은 P3(last, 기준값 0.5)와 P0. ACC는 기준값 0.5와 아래 보정 τ 두 가지, 불균형 세트를 위해 bACC를 함께 보고한다.
- **평가 공통**: `code/t5_common.py`(TASKS4 점수 은행과 비트 단위 같은 영상 점수 재현 확인, `t5_check.py`). 결과 은행 `cache/scorebank_t5.npz`.
- **M1 전문가 가중치 병합**(`code/t5_merge.py`, 학습 없음): 같은 seed의 P3·U3f·U3s는 초기값 θ0(저자 baseline 상단 + SBI 사전학습 분기 + seed 규칙 gating)이 같다. P3는 상단이 θ0인 U3로 본다.
  - 선형 θ = (1−t)θ_A + tθ_B, t ∈ {0, 0.1, …, 1}. task vector θ = θ0 + a(θ_A−θ0) + b(θ_B−θ0), a ∈ {0.5, 0.75, 1, 1.25}, b ∈ {0.25, 0.5, 0.75, 1}.
  - 쌍: (P3 last, U3s best), (P3 best, U3s best), (U3f best, U3s best). BN running 통계는 (a·A + b·B)/(a+b) 볼록 결합(외삽 시 분산 음수 방지).
  - 선택: 선형·task vector 각 1위를 고르고 둘 중 선택 점수가 높은 것을 단일 모델 대표로 한다.
  - 대조: WiSE형 θ0 + b(θ_U3s−θ0), b ∈ {0.5, …, 0.9}(FF++ 전문가 없이 SBI 학습량만 줄임).
  - seed 수프(`code/t5_soup.py`): seed 3개 병합 모델의 가중치 평균(단일 모델).
- **M2 기준값 보정**(`code/t5_calib.py`): 모델·seed마다 선택 세트 평균 bACC가 최대인 τ(0.01~0.99, 0.005 간격)를 고정해 보고 세트에 적용.
- **M3 상보 전문가 증류**(`code/run_train_distill.py`): 학생 = U3 구조(θ0에서 시작). 데이터·스케줄은 혼합 학습과 같다(배치 FF++ 8 + SBI 8, 900스텝/epoch, 20 epoch).
  - 손실 = CE + 0.15·CE(detail) + β·T²·KL(교사‖학생), β = 1, T = 2(미리 고정).
  - 교사 = 같은 seed (P3 last × 0.3 + U3s best × 0.7) logit.
  - 체크포인트 선택은 (FF++ val + SBI val 균형 CE)/2 최저(best) + last. 둘 중 하나는 선택 세트 AUC로 고른다. registry kind=distill, `Ds_b1_s{seed}`.
- **M5 좌우 반전 TTA**(`code/build_flip_eval_cache.py`, `code/t5_tta.py`): 평가 세트 반전 캐시(게이트: val 64장 재계산이 기존 캐시와 비트 일치)를 쓴다. 프레임마다 원본·반전 확률을 평균한다. P0·P3에도 같게 적용한다.
- **위치 지표**: §11 규칙(`code/t5_loc.py`). **최종 표**: `code/t5_final.py`.

## 19. 후속 작업 TASKS6 — 추가 성능 향상 탐색 (2026-10-17 추가, 사용자 요청 "여러 방법을 시도")
- **세트·판정**: §17·§18과 같다. 기준선은 TASKS5 task vector 병합이다.
- **학습 없는 방법**(`code/t6_merge.py`)
  - N1 모듈별 비율: 상단 b_top ∈ {0, 0.5, 0.75, 1}, 분기 (a, b) 8쌍
  - N2 TIES: 텐서별 상위 k% 크기, 부호 선출, 같은 부호 평균 × λ. k ∈ {0.2, 0.5}, λ ∈ {0.75, 1, 1.25, 1.5}
  - N2 DARE: 확률 p로 버리고 1/(1−p) 재척도. p ∈ {0.5, 0.9}
  - N3 다중 전문가 탐욕 추가: c ∈ {0.25, 0.5}, +0.10 이상일 때만 채택
  - N1b 하이브리드: 상단 6가지 × 분기 3가지
  - N4 영상 집계(`code/t6_agg.py`): 모델마다 선택 세트로 고른다. P0·P3도 같은 방식으로 고른다.
- **학습 방법**(`code/run_train_sbi2.py`, registry kind=sbi2): SBI 전문가 개선. 데이터·손실·스케줄은 S9 U3s와 같다.
  - T1 시작점 = P3 last 분기(Xp3)
  - T2 EMA 0.999(Xema)
  - T3 lr 1e-5(Xlr1)
  - seed 0으로 선별(TASKS5 seed 0 기준과 비교)한 뒤, 유망한 것(Xp3·Xlr1)만 seed 1·2를 추가했다.
  - 평가는 `code/t6_texp.py`: P3와의 선형·task vector 병합 격자를 쓴다.
- **최종 비교**(`code/t6_final.py`): 선택 점수와 보고 세트 AUC의 순위 상관(선택 과적합 점검)을 포함한다.
- **사후 규칙(결과를 본 뒤 추가, 별도 표시)**: "FF++ val이 기준 모델보다 떨어지지 않는 후보 중 선택 점수 최고." 미리 정한 규칙(선택 점수 최고)이 선택 세트 과적합으로 실패해서 제안했다. 다음 실험에서는 사전 규칙으로 검증해야 한다.

## 13. 변경 이력
| 날짜 | 변경 | 사유 | 영향 |
|---|---|---|---|
| 2026-10-17 | §19 TASKS6 정의 추가 | 사용자 요청(추가 성능 향상, 여러 방법) | 기존 결과 불변, 새 학습 7회(sbi2) |
| 2026-10-17 | §18 TASKS5 정의 추가 | 사용자 요청(P3를 넘는 개선, ACC·AUC) | 기존 결과 불변, 새 학습 3회(증류), 학습 없는 병합 모델 다수 |
| 2026-10-17 | §17 TASKS4 선택 규칙·개선 방법 정의 추가 | 사용자 요청 4가지(약점 기반 개선 실험) | 기존 결과 불변, 새 실행 11회(mix 6, anchor 5) |
| 2026-10-16 | §15 S9(baseline 일부 풀기), §16 S10(DFD·FaceShifter 평가) 추가 | 사용자 승인("2,3 진행"). DFDC는 Kaggle 인증 없음 → 제외 | 기존 결과 불변 |
| 2026-10-16 | §14 후속 실험 S8(SBI 학습 프로토콜) 추가 | S7 원인 가설 검증, 사용자 승인("추천방식으로 진행") | 기존 S0~S7 결과·코드 불변, 새 실행 9회 |
| 2026-10-15 | §4.8 SBI 규칙 확정(S3), §11 정렬 방법·지표 확정(S6 탐색 결과) | 위임된 세부를 구현대로 기록. SBI 경계 GT는 비율 곱하기 전 마스크 사용 | 모든 설정 공통 |
| 2026-10-15 | 최초 작성(S0) | HANDOFF §5 고정 변수·설정 정의 옮김, 위임된 세부(1:1 스케줄, 프레임 인덱스 공식, val_loss_bal, P2b 집계=평균, 특징 평균 맵=σ(채널 평균)) 확정 | – |
