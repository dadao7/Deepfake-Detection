"""
evaluate.py
Best 가중치로 FF++ c40 test_only 전체 평가.

실행:
    python evaluate.py --exp improved_mini
    python evaluate.py --exp improved_full
    python evaluate.py --exp baseline_mini
"""
import os, sys, argparse, torch, torch.nn as nn
from torch.amp import autocast
from tqdm import tqdm

import config as C

_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.pop('weights_only', None)
    return _orig_load(*args, **kwargs, weights_only=False)
torch.load = _patched_load

from model.network import Detector as M2F2Det
from dataset import ImageFolderH5Dataset_inference, get_dataloader
from eval_utils import parse_auc_score, save_results


def evaluate(args):
    ckpt_path = C.get_ckpt_path(args.exp)
    result_csv = C.get_result_path(f'eval_{args.exp}')

    if not os.path.exists(ckpt_path):
        print(f'[오류] 가중치 없음: {ckpt_path}')
        return

    # 모델 로드
    if 'improved' in args.exp:
        from improved_model import M2F2DetImproved
        baseline = M2F2Det()
        baseline = nn.DataParallel(baseline).to(C.device)
        model = M2F2DetImproved(baseline, k=C.K, crop_size=C.CROP_SIZE, alpha=C.ALPHA)
        model = model.to(C.device)
    else:
        model = M2F2Det()
        model = nn.DataParallel(model).to(C.device)

    saved = torch.load(ckpt_path, map_location=C.device)
    model.load_state_dict(saved['model'])
    model.eval()
    print(f'[OK] 가중치 로드: {ckpt_path}')

    # 테스트 데이터
    test_cfg  = {'post': {'blur': {'prob': 0.0, 'sig': [0.0, 3.0]}, 'jpeg': {'prob': 0.0, 'method': ['cv2', 'pil'], 'qual': [30, 100]}}}
    split_fn  = os.path.join(C.M2F2_REPO, 'utils', 'FFPP_split', 'test.json')
    ds        = ImageFolderH5Dataset_inference(C.TEST_ONLY_DIR, test_cfg, split_fn, num_frames=32)
    loader    = get_dataloader(ds, mode='test', bs=C.TEST_BATCH, workers=C.NUM_WORKERS)

    torch.cuda.empty_cache()
    pred_lst, gt_lst, id_lst = [], [], []

    with torch.no_grad(), autocast('cuda'):
        for imgs, lbls, ids in tqdm(loader, desc=f'Eval {args.exp}'):
            imgs = imgs.float().to(C.device)
            if 'improved' in args.exp:
                out = model(imgs, return_dict=True)
                preds = out['pred'].float()
            else:
                preds = model(imgs).float()
            pred_lst.append(preds.cpu())
            gt_lst.append(lbls)
            id_lst.extend(ids)

    preds = torch.cat(pred_lst)
    gts   = torch.cat(gt_lst)
    save_results(preds, gts, id_lst, result_csv)

    auc, acc = parse_auc_score(result_csv)
    print(f'\n{"="*60}')
    print(f'  {args.exp}:  AUC={auc*100:.2f}%  Acc={acc*100:.2f}%')
    print(f'  결과 저장: {result_csv}')
    print(f'{"="*60}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='improved_mini',
                        help='실험 이름 (weights/ 파일명 기준)')
    args = parser.parse_args()
    evaluate(args)
