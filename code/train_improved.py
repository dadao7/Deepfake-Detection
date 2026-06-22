"""
================================================================================
[M2F2-Det Improved] train_improved.py - 학습 및 파인튜닝 실행 스크립트
================================================================================
[명령어 사용 가이드]

1. 🏆 [완전체 20에폭 재학습 모드] (Method 3 + Method 4 탑재 및 동적 게이팅 수행)
   python train_improved.py --mode full --epochs 20 --alpha dynamic --exp improved_finetune_full_last --save_last

2. 🪓 [실험 #6-2 쐐기 검증 모드] (Method 3 제거, Method 4 단독 20에폭 고정 알파 수행)
   python train_improved.py --mode full --epochs 20 --alpha 0.5 --exp improved_method4_20ep

3. 🧪 [미니 테스트 모드] (빠른 디버깅용 1에폭 세팅)
   python train_improved.py --mode mini --epochs 1 --alpha 0.5 --exp improved_test_run
================================================================================
"""
import os, sys, argparse, torch, torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score

import config as C

# 가중치 언피클링 보안 우회 패치
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.pop('weights_only', None)
    return _orig_load(*args, **kwargs, weights_only=False)
torch.load = _patched_load

from sequence.models.M2F2_Det.models.model import M2F2Det
from improved_model import M2F2DetImproved
from mini_train_loader import get_mini_train_loader
from full_train_loader import get_full_train_loader


def load_model(alpha_val):
    print(f'==> 모델 초기화 가동 (Alpha 가중치 모드: {alpha_val})')
    model = M2F2Det(hidden_size=1792, deepfake_encoder_name='efficientnet_b4')

    if os.path.exists(C.STAGE1_CKPT):
        ckpt = torch.load(C.STAGE1_CKPT, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt.get('model', ckpt)))
            
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith('module.'): new_key = new_key.replace('module.', '', 1)
            if new_key.startswith('baseline.'): new_key = new_key.replace('baseline.', '', 1)
            if '.text_model.' in new_key: new_key = new_key.replace('.text_model.', '.')
            if '.vision_model.' in new_key: new_key = new_key.replace('.vision_model.', '.')
            new_state_dict[new_key] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        print(f'[OK] Stage-1 Baseline 가중치 인젝션 완료: {C.STAGE1_CKPT}')
    else:
        print(f'[경고] Stage-1 가중치가 유실되었습니다: {C.STAGE1_CKPT}')

    model = nn.DataParallel(model)
    model = model.to(C.device)

    # 파라미터로 전달받은 알파 모드를 아키텍처 융합 코어에 바인딩
    improved = M2F2DetImproved(model, k=C.K, crop_size=C.CROP_SIZE, alpha=alpha_val)
    improved = improved.to(C.device)

    if os.path.exists(C.PRETRAIN_CKPT):
        saved = torch.load(C.PRETRAIN_CKPT, map_location=C.device)
        if 'heatmap_head' in saved: improved.heatmap_head.load_state_dict(saved['heatmap_head'], strict=False)
        if 'detail_branch' in saved: improved.detail_branch.load_state_dict(saved['detail_branch'], strict=False)
        print('[OK] SBI 정규화 사전학습 브랜치 결합 완료')
    else:
        print('[경고] SBI 가중치 전이 실패, 가우시안 무작위 난수 초기화 수행')

    return improved


def train(args):
    # 인자값에서 알파 타입 동적 디코딩 처리
    try:
        final_alpha = float(args.alpha)
    except ValueError:
        final_alpha = args.alpha

    print(f'Device: {C.device}')
    print(f'Mode: {args.mode}, Epochs: {args.epochs}, Alpha: {final_alpha}, Save Last: {args.save_last}')

    # ================= [지정 이름 기반 가중치 네이밍 처리] =================
    if args.exp:
        base_exp_name = args.exp
    else:
        base_exp_name = 'improved_finetune_mini' if args.mode == 'mini' else 'improved_finetune_full'
        if args.save_last:
            base_exp_name += '_last'
        
    save_ckpt = C.get_ckpt_path(base_exp_name)
    print(f'==> 저장 위치 타겟 확정: {save_ckpt}')
    # =====================================================================

    if args.mode == 'mini':
        train_loader = get_mini_train_loader(C.MINI_H5, batch_size=C.TRAIN_BATCH, num_workers=C.NUM_WORKERS, frames_per_video=10, seed=42, split='train')
        val_loader = get_mini_train_loader(C.MINI_H5, batch_size=C.TRAIN_BATCH, num_workers=C.NUM_WORKERS, frames_per_video=10, seed=42, split='val')
    else:
        train_loader = get_full_train_loader(C.FULL_DATA_DIR, batch_size=C.TRAIN_BATCH, num_workers=C.NUM_WORKERS, frames_per_video=10, split='train', augment=True, balanced=False)
        val_loader = get_full_train_loader(C.FULL_DATA_DIR, batch_size=C.TRAIN_BATCH, num_workers=C.NUM_WORKERS, frames_per_video=3, split='val', augment=False, balanced=False)

    model = load_model(final_alpha)

    for param in model.baseline.parameters():
        param.requires_grad = False
    print('[OK] Baseline 가중치 동결 완료 (영역 침범 차단)')

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        model.baseline.eval()  # 글로벌 백본 배치정규화 무결성 잠금

        total_loss, correct, total = 0.0, 0, 0

        for imgs, lbls in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}'):
            imgs, lbls = imgs.float().to(C.device), lbls.to(C.device)
            optimizer.zero_grad()

            with autocast('cuda'):
                out  = model(imgs, return_dict=True)
                loss = criterion(out['pred'].float(), lbls) + C.LOSS_W_DETAIL * criterion(out['detail_pred'].float(), lbls)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            correct    += (out['pred'].argmax(1) == lbls).sum().item()
            total      += imgs.size(0)

        train_loss = total_loss / total
        train_acc  = correct / total
        scheduler.step()

        # ================= Validation =================
        model.eval()
        val_loss, val_total, val_correct = 0.0, 0, 0
        all_preds, all_lbls = [], []

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.float().to(C.device), lbls.to(C.device)
                
                out  = model(imgs, return_dict=True)
                loss = criterion(out['pred'], lbls)
                
                val_loss    += loss.item() * imgs.size(0)
                val_correct += (out['pred'].argmax(1) == lbls).sum().item()
                val_total   += imgs.size(0)
                
                probs = torch.softmax(out['pred'], dim=1)[:, 1]
                all_preds.extend(probs.cpu().numpy())
                all_lbls.extend(lbls.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(all_lbls, all_preds)
        cur_lr = scheduler.get_last_lr()[0]
        
        print(f'Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
              f'val_AUC={val_auc*100:.2f}%  val_acc={val_acc*100:.2f}%  lr={cur_lr:.2e}')

        # ================= [체크포인트 스마트 세이브] =================
        if args.save_last:
            torch.save({'model': model.state_dict(), 'alpha': model.alpha}, save_ckpt)
            print(f'  → [강제 업데이트] Epoch {epoch} 가중치 동기화 완료.')
            if val_auc > best_auc: best_auc = val_auc 
        else:
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save({'model': model.state_dict(), 'alpha': model.alpha}, save_ckpt)
                print(f'  → [Best 갱신 고점 저장] (AUC: {best_auc*100:.2f}%, ACC: {val_acc*100:.2f}%)')
            
    print(f'🎉 학습 파이프라인 무사히 완주 완료. 가중치 타겟 파일: {save_ckpt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="M2F2-Det Improved 파인튜닝 스크립트")
    parser.add_argument('--mode',       default='full', choices=['mini', 'full'])
    parser.add_argument('--epochs',     type=int, default=C.EPOCHS)
    parser.add_argument('--save_last',  action='store_true')
    parser.add_argument('--alpha',      type=str, default='dynamic', help="Alpha 값 규격: 'dynamic' 문자열 또는 float 숫자(예: 0.5)")
    parser.add_argument('--exp',        type=str, default=None, help="커스텀 파일 네이밍 지정인자")
    
    args = parser.parse_args()
    C.check_paths()
    train(args)