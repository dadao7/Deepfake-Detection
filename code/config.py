"""
================================================================================
[M2F2-Det Improved] config.py - 중앙 환경 및 하이퍼파라미터 관리 스크립트
================================================================================
"""
import os, sys, torch

# ── [1] 글로벌 경로 설정 ───────────────────────────────────────────────────
BASE_DIR      = r'C:\m2f2'
M2F2_REPO     = r'C:\m2f2\M2F2_Det'
CODE_DIR      = os.path.join(BASE_DIR, 'code')

FULL_DATA_DIR = os.path.join(BASE_DIR, 'data', 'FF++')
MINI_H5       = os.path.join(BASE_DIR, 'data', 'mini_train', 'FF++_mini_train_20pct.h5')
LM_JSON       = os.path.join(BASE_DIR, 'data', 'lm_json', 'train_81_FF++_processed_720.json')
TEST_ONLY_DIR = os.path.join(BASE_DIR, 'data', 'test_only')

WEIGHTS_DIR   = os.path.join(BASE_DIR, 'weights')
STAGE1_CKPT   = os.path.join(WEIGHTS_DIR, 'current_model_180.pth')
VISION_TOWER  = os.path.join(WEIGHTS_DIR, 'vision_tower.pth')
PRETRAIN_CKPT = os.path.join(WEIGHTS_DIR, 'pretrained_heatmap_detail.pth')
RESULTS_DIR   = os.path.join(BASE_DIR, 'results')

# ── [2] 가중치 및 결과 파일 동적 경로 생성 ──────────────────────────────────
def get_ckpt_path(exp_name):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    return os.path.join(WEIGHTS_DIR, f'{exp_name}_best.pth')

def get_result_path(exp_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, f'{exp_name}.csv')

# ── [3] 하이퍼파라미터 프리셋 ────────────────────────────────────────────────
EPOCHS        = 20
TRAIN_BATCH   = 16
TEST_BATCH    = 8
LR            = 5e-5
WEIGHT_DECAY  = 1e-3
LOSS_W_DETAIL = 0.15
K             = 3
CROP_SIZE     = 64
NUM_WORKERS   = 0  # Windows 안정성을 위해 0 고정

# ── [4] 시스템 디바이스 맵핑 ─────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if M2F2_REPO not in sys.path:
    sys.path.insert(0, M2F2_REPO)
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

def check_paths():
    required = {
        'M2F2_REPO': M2F2_REPO,
        'STAGE1_CKPT': STAGE1_CKPT,
        'VISION_TOWER': VISION_TOWER,
    }
    for name, path in required.items():
        if not os.path.exists(path):
            print(f'[경고] {name} 파일 혹은 폴더 유실됨: {path}')
        else:
            print(f'[OK] {name} 경로 확인 완료: {path}')