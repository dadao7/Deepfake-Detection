```markdown
# Deepfake-Detection

딥러닝 기반의 설명 가능한(XAI) 딥페이크 탐지 시스템 저장소입니다.

---

## 1. 프로젝트 소개
본 프로젝트는 딥러닝 기반의 딥페이크 탐지 시스템입니다. **M2F2-Det 기반 탐지 모델**에 **Heatmap 기반 Top-K Crop**, **국소패치 정규화**, **동적 결합 네트워크**를 적용하여 미세 위조 흔적 탐지 성능을 향상시켰습니다.

또한, **SIDA 기반 VLM/LLM 설명 생성 모듈**을 활용하여 단순한 REAL/FAKE 분류 결과뿐만 아니라, 탐지 근거를 자연어로 제공하는 설명 가능한 AI 시스템을 구현하였습니다.

### 주요 목적
* 딥페이크 이미지로 인한 허위 정보 및 디지털 범죄 문제에 대응
* 탐지 결과에 대한 신뢰성 있는 근거(자연어 설명) 제공

---

## 2. 디렉토리 구조 (Directory Structure)
```text
├── code/               # 모델 학습 및 평가 핵심 소스코드
├── M2F2_Det/           # M2F2-Det 서브 모듈 및 실행 스크립트
│   ├── asset/
│   ├── eval/
│   ├── llava/
│   ├── scripts/
│   ├── sequence/
│   └── utils/
├── .gitignore          # 깃허브 업로드 제외 설정 파일
├── environment.yml     # Conda 가상환경 설정 파일
└── pyproject.toml      # 프로젝트 빌드 설정 파일

```

---

## 3. 데이터셋 및 가중치 안내 (Dataset & Checkpoints)

⚠️ **용량 문제로 인한 안내 사항**

> 학습 데이터셋 및 모델 가중치(Checkpoints) 파일은 GitHub 파일 용량 제한(100MB)으로 인해 저장소에 직접 포함되지 않았습니다. 프로젝트를 정상적으로 실행하려면 아래 링크에서 데이터를 다운로드하여 로컬 경로에 배치해야 합니다.

### 데이터셋 다운로드

* **구글 드라이브 링크**: [Dataset Download Link](https://drive.google.com/drive/folders/1N4X3rvx9IhmkEZK-KIk4OxBrQb9BRUcs)
* **배치 경로**: 다운로드한 데이터셋은 최상위 디렉토리의 `data/` 또는 `M2F2_Det/dataset/` 경로에 위치시켜 주십시오.

---

## 4. 환경 설정 (Environment Setup)

본 프로젝트는 Anaconda 가상환경 기반으로 구성되어 있습니다. 아래 명령어를 통해 필요한 패키지를 일괄 설치할 수 있습니다.

```bash
conda env create -f environment.yml
conda activate [가상환경이름]

```

---

## 5. 출처 및 라이선스 (Acknowledgements & License)

* 본 프로젝트의 베이스라인 및 핵심 탐지 모듈은 아래 오픈소스 저장소를 기반으로 구현 및 수정되었습니다.
* 원저작자 저장소: [CHELSEA234/M2F2_Det (CVPR25 Oral)](https://github.com/CHELSEA234/M2F2_Det)


* 해당 모듈은 **MIT License**를 따르며, 원본 라이선스 명시는 `M2F2_Det/LICENSE` 파일에서 확인할 수 있습니다.

```

```
