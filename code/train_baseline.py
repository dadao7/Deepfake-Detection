"""
train_baseline.py  [기준 모델 - 처음부터 학습]
저자 FF++ 가중치 미사용. EfficientNet-B4(ImageNet) + CLIP(OpenAI) 표준 가중치에서 시작.
개선 모델(train_improved.py)과 동일 조건으로 학습 → 공정한 비교 기준.

실행:
    conda activate m2f2
    cd C:\m2f2\code
    python train_baseline.py --mode full --epochs 100
"""
import os, sys, argparse, torch, torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast, GradScaler

import config as C

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.pop('weights_only', None)
    return _orig_load(*args, **kwargs, weights_only=False)
torch.load = _patched_load

from sequence.models.M2F2_Det.models.model import M2F2Det
from mini_train_loader import get_mini_train_loader
from full_train_loader import get_full_train_loader


def load_model():
    print('Baseline 초기화 (처음부터 학습 - 저자 가중치 미사용)')
    model = M2F2Det(hidden_size=1792, deepfake_encoder_name='efficientnet_b4')
    model = nn.DataParallel(model)
    model = model.to(C.device)
    return model


def train(args):
    print(f'Device: {C.device}')
    print(f'Mode: {args.mode}, Epochs: {args.epochs}')

    if args.mode == 'mini':
        train_loader = get_mini_train_loader(
            C.MINI_H5, batch_size=C.TRAIN_BATCH,
            num_workers=C.NUM_WORKERS, frames_per_video=10, seed=42, split='train')
        val_loader = get_mini_train_loader(
            C.MINI_H5, batch_size=C.TRAIN_BATCH,
            num_workers=C.NUM_WORKERS, frames_per_video=10, seed=42, split='val')
        save_ckpt = C.get_ckpt_path('baseline_scratch_mini')
    else:
        train_loader = get_full_train_loader(
            C.FULL_DATA_DIR, batch_size=C.TRAIN_BATCH,
            num_workers=C.NUM_WORKERS, frames_per_video=10,
            split='train', augment=False, balanced=False)
        val_loader = get_full_train_loader(
            C.FULL_DATA_DIR, batch_size=C.TRAIN_BATCH,
            num_workers=C.NUM_WORKERS, frames_per_video=3,
            split='val', augment=False, balanced=False)
        save_ckpt = C.get_ckpt_path('baseline_scratch_full')

    model = load_model()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, lbls in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}'):
            imgs, lbls = imgs.float().to(C.device), lbls.to(C.device)
            optimizer.zero_grad()

            with autocast('cuda'):
                out  = model(imgs, return_dict=True)
                loss = criterion(out['pred'].float(), lbls)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            correct    += (out['pred'].argmax(1) == lbls).sum().item()
            total      += imgs.size(0)

        train_loss = total_loss / total
        scheduler.step()

        model.eval()
        val_loss, val_total = 0.0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.float().to(C.device), lbls.to(C.device)
                with autocast('cuda'):
                    out  = model(imgs, return_dict=True)
                    loss = criterion(out['pred'].float(), lbls)
                val_loss  += loss.item() * imgs.size(0)
                val_total += imgs.size(0)
        val_loss /= val_total

        cur_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch}: train={train_loss:.4f}  val={val_loss:.4f}  '
              f'acc={correct/total*100:.2f}%  lr={cur_lr:.2e}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'model': model.state_dict()}, save_ckpt)
            print('  → Best saved.')

    print(f'학습 완료. 저장: {save_ckpt}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',   default='full', choices=['mini', 'full'])
    parser.add_argument('--epochs', type=int, default=C.EPOCHS)
    args = parser.parse_args()
    C.check_paths()
    train(args)
