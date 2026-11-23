# Deepfake-Detection (DeepSight)

딥러닝 기반의 설명 가능한(XAI) 딥페이크 탐지 시스템 저장소입니다. 팀 와그닥(동아대학교 컴퓨터공학과 졸업 프로젝트).

---

## 0. 최종 모델 (2학기)

M2F2-Det(CVPR 2025) 논문 모델 위에 의심 부위를 다시 보는 분기(heatmap → Top-3 crop → 패치 정규화 → DenseNet121 → 동적 결합)를 붙였습니다. 그다음 FF++ 전문가와 SBI 전문가의 가중치를 **추가 학습 없이 병합**해 모델 하나로 만들었습니다.

- **가중치 식**: θ = θ0 + 0.5 × (FF++ 전문가 − θ0) + 0.75 × (SBI 전문가 − θ0)
  - θ0: 논문 모델(고정) + SBI로 사전학습한 분기 `exp/runs/sbi_pretrain/sbi_best.pt`
  - FF++ 전문가: `exp/runs/P3_d100_s{0..4}/ckpt_best.pt` (분기만 FF++로 학습)
  - SBI 전문가: `exp/runs/U3s_s{0..4}/ckpt_best.pt` (분기 + 논문 모델 상단을 SBI로 학습)
  - BatchNorm 통계는 두 전문가 값을 볼록 결합합니다. 비율(0.5, 0.75)은 성능 보고에 쓰지 않는 선택 세트(FF++ val + DFD-val)로만 골랐습니다.
- **결과** (영상 AUC, seed 3개 평균, 논문 모델 대비)

| 데이터셋 | 논문 모델 | 최종 모델 | 차이 |
|---|---|---|---|
| FF++ c40 test (학습 데이터) | 95.22 | 95.13 | 유지 |
| Celeb-DF (처음 봄) | 76.15 | 78.84 | **+2.7** |
| FaceShifter (처음 보는 위조 방식) | 68.01 | 72.94 | **+4.9** |
| DFD (처음 봄) | 70.78 | 75.65 | **+4.9** |

처음 보는 데이터셋 3곳의 향상은 모두 영상 단위 95% 신뢰구간이 0 밖입니다. seed 5개 모두에서 재현됐고, 추론 시간은 논문 모델 대비 +3.4%입니다. 자세한 내용은 `exp/PROJECT_FLOW/00_프로젝트_흐름_안내.md`, `exp/reports/t5_merge.md`, `exp/reports/qa_followup.md`를 보세요.

### 최종 모델 만들기

```python
import sys; sys.path.insert(0, 'exp/code')
import t5_merge as M

seed = 0                                         # 0~4
s0  = M.init_state(seed)                         # θ0 (저자 가중치 + SBI 사전학습 분기)
p3  = M.expert_state('P3',  'best', seed, s0)    # FF++ 전문가
u3s = M.expert_state('U3s', 'best', seed, s0)    # SBI 전문가
final_sd = M.combine(s0, p3, u3s, 0.5, 0.75)     # 최종 모델 가중치
model = M.model_from(final_sd, seed)             # CUDA, eval 모드
```

저자 가중치(`weights/current_model_180.pth`, `weights/vision_tower.pth`)가 먼저 있어야 합니다(3절).

---

## 1. 프로젝트 소개 (1학기)

본 프로젝트는 딥러닝 기반의 딥페이크 탐지 시스템입니다. **M2F2-Det 기반 탐지 모델**에 **Heatmap 기반 Top-K Crop**, **국소패치 정규화**, **동적 결합 네트워크**를 적용하여 미세 위조 흔적 탐지 성능을 향상시켰습니다.

또한, **SIDA 기반 VLM/LLM 설명 생성 모듈**을 활용하여 단순한 REAL/FAKE 분류 결과뿐만 아니라, 탐지 근거를 자연어로 제공하는 설명 가능한 AI 시스템을 구현하였습니다.

### 주요 목적
* 딥페이크 이미지로 인한 허위 정보 및 디지털 범죄 문제에 대응
* 탐지 결과에 대한 신뢰성 있는 근거(자연어 설명) 제공

---

## 2. 디렉토리 구조 (Directory Structure)

```
├── code/                 # 1학기 개선 모델 코드 + 앱 API 서버(server.py, server_features.py)
├── exp/                  # 2학기 통제 비교 실험과 최종 모델
│   ├── code/             #   실험·병합·평가 코드 (경로 설정: common.py)
│   ├── runs/             #   학습 실행 107개의 설정·지표·학습 곡선·로그 + 최종 모델용 체크포인트
│   ├── reports/          #   결과 보고서·표·그림
│   ├── PROJECT_FLOW/     #   처음부터 최종 모델까지 읽는 순서대로 정리한 문서
│   ├── cache/            #   공식 분할 인덱스, 모든 모델의 영상 점수(scorebank*.npz), 얼굴 정렬 템플릿
│   ├── ref/              #   논문 모델 재현 평가, 질문 대비 분석 스크립트
│   ├── logs/, queues/    #   실행 기록, 실행 큐
│   ├── PROTOCOL.md       #   실험 규칙
│   ├── GUIDE.md          #   파일 지도와 캐시 생성 순서
│   ├── HANDOFF.md        #   실험 설계
│   ├── PROGRESS.md       #   진행·결정 기록
│   └── registry.csv      #   학습 실행 기록
├── M2F2_Det/             # 논문 저자 원본 코드 (수정하지 않음, environment.yml 포함)
├── 데이터셋_준비.md       # 필요한 데이터셋과 배치 위치
└── .gitignore
```

처음 읽을 순서: `exp/GUIDE.md` → `exp/PROJECT_FLOW/00_프로젝트_흐름_안내.md` → `exp/reports/`

---

## 3. 데이터셋 및 가중치 안내 (Dataset & Checkpoints)

⚠️ **저장소에 포함하지 않은 것**

| 빠진 것 | 받는 곳 |
|---|---|
| 저자 가중치 `current_model_180.pth`(M2F2-Det detector-only), `vision_tower.pth`(CLIP 인코더) | `M2F2_Det/README.md`의 Inference 절 링크. 받은 뒤 `weights/`에 둡니다 |
| 데이터셋(FF++ c40, Celeb-DF v2, DFD, FaceShifter) | `데이터셋_준비.md` 참고. 각 데이터셋 약관에 동의하고 공식 경로로 받습니다 |
| FF++ 81점 랜드마크(`data/lm_json/`), DFD 파일 목록(`data/dl_meta/`) | FF++에서 만든 데이터라 올리지 않았습니다. 팀 내부 자료로 받습니다 |
| dlib 68점 랜드마크 모델(`exp/cache/models/shape_predictor_68_face_landmarks.dat`) | dlib.net에서 받습니다 |
| 대용량 캐시(약 136GB), 나머지 학습 체크포인트 | 데이터셋에서 다시 만듭니다(`exp/GUIDE.md`의 단계별 스크립트). 결과는 `exp/runs/*/metrics.json`과 `exp/cache/scorebank*.npz`에 남아 있습니다 |

최종 모델에 필요한 체크포인트(`exp/runs/` 아래 11개, 약 525MB)는 저장소에 포함되어 있습니다.

### 데이터셋 다운로드 (1학기)

* **구글 드라이브 링크**: [Dataset Download Link](https://drive.google.com/drive/folders/1N4X3rvx9IhmkEZK-KIk4OxBrQb9BRUcs)
* **배치 경로**: 다운로드한 데이터셋은 최상위 디렉토리의 `data/` 또는 `M2F2_Det/dataset/` 경로에 위치시켜 주십시오.

---

## 4. 환경 설정 (Environment Setup)

1학기 코드는 Anaconda 가상환경 기반입니다.

```bash
conda env create -f M2F2_Det/environment.yml
conda activate [가상환경이름]
```

2학기 실험(`exp/`) 환경: Python 3.11, PyTorch 2.12(CUDA), transformers 5.9 외.

```bash
pip install -r exp/requirements.txt   # PyTorch는 CUDA 버전에 맞는 휠을 먼저 설치
```

`exp/requirements.txt`는 진행 기록에 남은 버전을 기준으로 만든 목록이며, 실제 환경에서 뽑은 전체 목록은 아닙니다. 버전이 없는 패키지는 기록이 없는 것입니다.

- 코드가 `C:\m2f2\...` 절대 경로를 씁니다. 다른 위치에서 실행하려면 `exp/code/common.py`의 `EXP`, `M2F2`, `ORIG_CODE`, `WEIGHTS`, `DATA`, `PY`를 고치세요.
- 앱 API 서버: `cd code && python server.py` (기본 가중치는 1학기 모델 `weights/improved_finetune_full_last_100ep_best.pth`).

---

## 5. 데이터 약관

- FaceForensics++ 계열(FF++, DFD, FaceShifter)과 Celeb-DF는 **비상업 연구용**이며 재배포가 금지되어 있습니다. 이 저장소에는 데이터와 데이터 얼굴 이미지를 넣지 않았습니다.
- FF++를 쓸 때는 Rössler et al., ICCV 2019를 인용합니다.

---

## 6. 출처 및 라이선스 (Acknowledgements & License)

* 본 프로젝트의 베이스라인 및 핵심 탐지 모듈은 아래 오픈소스 저장소를 기반으로 구현 및 수정되었습니다.
* 원저작자 저장소: [CHELSEA234/M2F2_Det (CVPR25 Oral)](https://github.com/CHELSEA234/M2F2_Det)
* 해당 모듈은 **MIT License**를 따르며, 원본 라이선스 명시는 `M2F2_Det/LICENSE` 파일에서 확인할 수 있습니다.
