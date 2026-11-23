"""S3 게이트 기록: (1) 새 SBI 사전학습 vs 1학기 SBI 가중치를 같은 SBI val 지표로 비교 (2) seg head 예측 격자
(SBI val 가짜: 이미지 | GT | seg head heatmap, FF++ test 가짜: 이미지 | 정렬 마스크 | seg head heatmap) — 로컬 확인용
출력: reports/s3_sbi_compare.json, reports/s3_seghead_grid.png"""
import json
import os
import sys

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import expmodel as E  # noqa: E402
import pretrain_sbi as P  # noqa: E402


def load(path):
    sd = torch.load(path, map_location='cpu', weights_only=False)
    head = E.HeatmapHead(272, 64).cuda()
    head.load_state_dict(sd['heatmap_head'])
    det = E.DetailCropBranch().cuda()
    det.load_state_dict(sd['detail_branch'])
    return head, det


def colorize(p):
    return cv2.cvtColor(cv2.applyColorMap((np.clip(p, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)


def main():
    torch.backends.cudnn.deterministic = True
    VA = P.SBISet('val')
    rep = {}
    for name, path in (('new_sbi_best', E.SBI_CKPT), ('sbi1_1학기', E.SBI1_CKPT)):
        head, det = load(path)
        rep[name] = P.evaluate(head, det, VA)
        print(name, {k: round(v, 3) for k, v in rep[name].items()}, flush=True)
    json.dump(rep, open(os.path.join(C.REPORTS, 's3_sbi_compare.json'), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False)
    # 격자
    head, _ = load(E.SBI_CKPT)
    head.eval()
    rng = np.random.default_rng(3)
    tiles = []
    idx = np.sort(rng.choice(VA.N, 6, replace=False))
    with torch.no_grad():
        b3 = torch.from_numpy(VA.b3[idx][:, 1]).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            hm = torch.sigmoid(head(b3).float())[:, 0].cpu().numpy()
    for j, ii in enumerate(idx):
        img = VA.img[ii, 1]
        gt = cv2.resize(VA.gt[ii].astype(np.float32), (224, 224), interpolation=cv2.INTER_NEAREST)
        hp = cv2.resize(hm[j] / max(hm[j].max(), 1e-6), (224, 224), interpolation=cv2.INTER_NEAREST)
        tiles.append(np.concatenate([img, colorize(gt), (img * 0.5 + colorize(hp) * 0.5).astype(np.uint8)], 1))
    loc = np.load(os.path.join(C.CACHE, 'masks', 'test_loc.npz'))
    tb3 = np.load(os.path.join(C.BASE_DIR, 'test_b3.npy'), mmap_mode='r')
    timg = np.load(os.path.join(C.FRAME_DIR, 'test_img.npy'), mmap_mode='r')
    sel = []
    for meth in C.METHODS:
        w = np.where(loc['meth'] == meth)[0]
        sel += list(rng.choice(w, 2, replace=False))
    rows = loc['cache_row'][sel]
    with torch.no_grad():
        with torch.autocast('cuda', dtype=torch.float16):
            hm = torch.sigmoid(head(torch.from_numpy(tb3[rows]).cuda()).float())[:, 0].cpu().numpy()
    g224 = np.unpackbits(loc['gt224'][sel], axis=-1)[..., :224]
    for j in range(len(sel)):
        img = timg[rows[j]]
        hp = cv2.resize(hm[j] / max(hm[j].max(), 1e-6), (224, 224), interpolation=cv2.INTER_NEAREST)
        mk = cv2.cvtColor((g224[j] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        t = np.concatenate([img, mk, (img * 0.5 + colorize(hp) * 0.5).astype(np.uint8)], 1)
        cv2.putText(t, str(loc['meth'][sel[j]])[:4], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        tiles.append(t)
    grid = np.concatenate([np.concatenate(tiles[r:r + 2], 1) for r in range(0, len(tiles), 2)], 0)
    small = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(C.REPORTS, 's3_seghead_grid.png'), cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    print('격자 저장 reports/s3_seghead_grid.png')


if __name__ == '__main__':
    main()
