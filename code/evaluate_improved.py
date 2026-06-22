"""
================================================================================
[M2F2-Det Improved] evaluate_improved.py - 교차 도메인 자동 보정 평가 스크립트
================================================================================
[명령어 사용 가이드]

1. 🏆 [완전체 모드 평가 및 가중치 해석 수행]
   python evaluate_improved.py --exp improved_finetune_full_last --alpha dynamic --dataset celeb-df --celeb_dir "C:\m2f2\data\Celeb-DF-v2"

2. 🪓 [실험 #6-2 쐐기 모델 평가]
   python evaluate_improved.py --exp improved_method4_20ep --alpha 0.5 --dataset celeb-df --celeb_dir "C:\m2f2\data\Celeb-DF-v2"
================================================================================
"""
import os, sys, argparse, torch, torch.nn as nn, cv2, numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

import config as C

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.pop('weights_only', None)
    return _orig_load(*args, **kwargs, weights_only=False)
torch.load = _patched_load

from sequence.models.M2F2_Det.models.model import M2F2Det
from improved_model import M2F2DetImproved


def alpha_type(value):
    if value.lower() == 'dynamic':
        return 'dynamic'
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"알파 값은 'dynamic'이거나 float 숫자 형태여야 합니다: '{value}'")


def load_improved_model(exp_name, alpha_val):
    print(f'==> Baseline 아키텍처 래퍼 껍데기 빌드 중...')
    base_model = M2F2Det(hidden_size=1792, deepfake_encoder_name='efficientnet_b4')
    base_model = nn.DataParallel(base_model).to(C.device)
    
    model = M2F2DetImproved(base_model, k=C.K, crop_size=C.CROP_SIZE, alpha=alpha_val)
    model = model.to(C.device)
    
    ckpt_path = C.get_ckpt_path(exp_name)
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(C.WEIGHTS_DIR, f"{exp_name}_best.pth")
        
    if os.path.exists(ckpt_path):
        print(f'==> 체크포인트 파일 로드 완료: {ckpt_path}')
        ckpt = torch.load(ckpt_path, map_location=C.device)
        state_dict = ckpt.get('model', ckpt)
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f'[경고] 가중치 누락: {ckpt_path}, 난수 상태로 강제 진입합니다.')
        
    model.alpha = alpha_val
    print(f'[평가 환경 확정] 모델 연산 가중치 모드(model.alpha) = {model.alpha}')
    return model


def evaluate_celeb_df(model, celeb_dir):
    model.eval()
    eval_items = []

    txt_file = None
    for f in os.listdir(celeb_dir):
        if f.lower().startswith('list_of_testing_videos'):
            txt_file = os.path.join(celeb_dir, f)
            break

    if txt_file and os.path.exists(txt_file):
        print(f"==> 테스트 비디오 목록 파일 기반 매칭 시작: {txt_file}")
        with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) >= 2:
                    if parts[0].isdigit() and parts[0] in ['0', '1']:
                        raw_label = int(parts[0])
                        rel_path = parts[1]
                    elif parts[-1].isdigit() and parts[-1] in ['0', '1']:
                        raw_label = int(parts[-1])
                        rel_path = parts[0]
                    else:
                        continue
                    
                    label = 1 if raw_label == 0 else 0
                    full_video_path = os.path.join(celeb_dir, rel_path.replace('/', os.sep).replace('\\', os.sep))
                    
                    dir_parts = rel_path.replace('\\', '/').split('/')
                    full_frame_dir = ""
                    if len(dir_parts) >= 2:
                        video_name = os.path.splitext(dir_parts[1])[0]
                        full_frame_dir = os.path.join(celeb_dir, dir_parts[0], 'frames', video_name)
                    
                    full_frame_dir_alt = os.path.join(celeb_dir, os.path.splitext(rel_path)[0].replace('/', os.sep))

                    if os.path.exists(full_video_path) and os.path.isfile(full_video_path):
                        eval_items.append({'type': 'video', 'path': full_video_path, 'label': label})
                    elif full_frame_dir and os.path.exists(full_frame_dir) and os.path.isdir(full_frame_dir):
                        eval_items.append({'type': 'frames', 'path': full_frame_dir, 'label': label})
                    elif os.path.exists(full_frame_dir_alt) and os.path.isdir(full_frame_dir_alt):
                        eval_items.append({'type': 'frames', 'path': full_frame_dir_alt, 'label': label})

    if len(eval_items) == 0:
        print("==> Fallback: 이미지 프레임 전체 강제 스캔 가동")
        for root, dirs, files in os.walk(celeb_dir):
            root_lower = root.lower()
            if 'frames' in root_lower and any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
                if 'synthesis' in root_lower or 'fake' in root_lower: label = 1
                elif 'real' in root_lower: label = 0
                else: continue
                eval_items.append({'type': 'frames', 'path': root, 'label': label})

    real_cnt = sum(1 for item in eval_items if item['label'] == 0)
    fake_cnt = sum(1 for item in eval_items if item['label'] == 1)
    print(f"==> 데이터 로드 성공 | 총 {len(eval_items)}개 비디오 유닛 검출 (Real: {real_cnt} / Fake: {fake_cnt})")
    
    if len(eval_items) == 0: return
        
    all_preds, all_labels = [], []
    real_alphas, fake_alphas = [], []  # 동적 알파 분석용 리스트

    with torch.no_grad():
        for item in tqdm(eval_items, desc="비디오 단위 평가 진행 중"):
            label = item['label']
            frames = []
            
            if item['type'] == 'video':
                cap = cv2.VideoCapture(item['path'])
                frame_count = 0
                while cap.isOpened() and frame_count < 3:
                    ret, frame = cap.read()
                    if not ret: break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (224, 224))
                    frame = frame.astype(np.float32) / 255.0
                    frame = np.transpose(frame, (2, 0, 1))
                    frames.append(frame)
                    frame_count += 1
                cap.release()
                
            elif item['type'] == 'frames':
                all_files = sorted(os.listdir(item['path']))
                img_files = [os.path.join(item['path'], f) for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
                if len(img_files) > 0:
                    indices = np.linspace(0, len(img_files) - 1, min(3, len(img_files)), dtype=int)
                    for idx in indices:
                        frame = cv2.imread(img_files[idx])
                        if frame is None: continue
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (224, 224))
                        frame = frame.astype(np.float32) / 255.0
                        frame = np.transpose(frame, (2, 0, 1))
                        frames.append(frame)
            
            if len(frames) == 0: continue
                
            imgs_tensor = torch.tensor(np.array(frames)).to(C.device)
            outputs = model(imgs_tensor, return_dict=True)
            logits  = outputs['pred']
            
            # [동적 가중치 분석 이식 파트]
            if model.alpha == 'dynamic' and 'dynamic_alpha' in outputs:
                dyn_alpha = outputs['dynamic_alpha'].mean().item()
                if label == 0: real_alphas.append(dyn_alpha)
                else: fake_alphas.append(dyn_alpha)

            probs = torch.softmax(logits, dim=1)[:, 1]
            all_preds.append(probs.mean().item())
            all_labels.append(label)
            
    all_labels, all_preds = np.array(all_labels), np.array(all_preds)
    auc = roc_auc_score(all_labels, all_preds) * 100
    preds_bin = (all_preds >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds_bin) * 100
    
    print(f"\n 최종 결과 → AUC={auc:.2f}%  Acc={acc:.2f}% ")
    
    # 동적 가중치 사후 정량 분석 리포팅 가동
    if model.alpha == 'dynamic' and len(real_alphas) > 0:
        print(f"\n📊 [Method 3 설명 가능성 지표 분석]")
        print(f"  └─ Real Video 적용 평균 Alpha 비중: {np.mean(real_alphas):.4f} (Global 신뢰도 우세)")
        print(f"  └─ Fake Video 적용 평균 Alpha 비중: {np.mean(fake_alphas):.4f} (Local Detail 브랜치 개입 증가)")


def main():
    parser = argparse.ArgumentParser(description="M2F2-Det Improved Evaluation Script")
    parser.add_argument('--exp', type=str, required=True)
    parser.add_argument('--alpha', type=alpha_type, default='dynamic')
    parser.add_argument('--dataset', type=str, default='celeb-df', choices=['ff++', 'celeb-df'])
    parser.add_argument('--celeb_dir', type=str, default=r"C:\m2f2\data\Celeb-DF-v2")
    
    args = parser.parse_args()
    C.check_paths()
    
    model = load_improved_model(args.exp, args.alpha)
    if args.dataset == 'celeb-df':
        evaluate_celeb_df(model, args.celeb_dir)


if __name__ == '__main__':
    main()