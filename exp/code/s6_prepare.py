"""S6-2: 위치 평가 준비 (PROTOCOL §11)
1) 방식별 마스크 이진화 임계값을 **val 정렬 마스크만으로** 결정: 1~254 값 히스토그램(평활)에서 [4, 100] 구간의 골짜기(최소 밀도).
   (마스크 영상은 F2F·NT·FS가 음영 있는 얼굴 렌더링, DF는 사각형 → 배경 0 근처와 얼굴 값 사이의 골짜기가 자연스러운 경계)
2) 정렬 통과(ok) 프레임의 GT: gt224(이진) + gt11(11×11 셀별 조작 픽셀 비율, INTER_AREA)
3) b_3·이미지 행: test는 기존 test 캐시 행(영상·프레임 i로 찾음), val은 정렬 프레임 baseline을 새로 계산(cache/masks/val_b3.npy)
출력: cache/masks/{split}_loc.npz, reports/s6_thresholds.json, reports/s6_mask_hist.png"""
import json
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

MD = os.path.join(C.CACHE, 'masks')


def valley_threshold(vals):
    h = np.bincount(vals.ravel(), minlength=256).astype(np.float64)[1:255]      # 1..254
    hs = np.convolve(np.log1p(h), np.ones(5) / 5, mode='same')
    lo, hi = 4 - 1, 100 - 1
    return int(np.argmin(hs[lo:hi]) + lo + 1), h


def main():
    thr, hists = {}, {}
    za = {sp: np.load(os.path.join(MD, f'{sp}_align.npz')) for sp in ('val', 'test')}
    mk = {sp: np.load(os.path.join(MD, f'{sp}_mask.npy'), mmap_mode='r') for sp in ('val', 'test')}
    for meth in C.METHODS:
        sel = (za['val']['meth'] == meth) & za['val']['ok']
        v = np.asarray(mk['val'][np.where(sel)[0]])
        thr[meth], hists[meth] = valley_threshold(v)
    # 히스토그램 그림(로그 축)
    W, H = 600, 240
    img = np.full((H * 4, W, 3), 255, np.uint8)
    for r, meth in enumerate(C.METHODS):
        h = np.log1p(hists[meth])
        h = (h / h.max() * (H - 30)).astype(int)
        for x in range(254):
            px = int(x / 254 * (W - 20)) + 10
            cv2.line(img, (px, r * H + H - 10), (px, r * H + H - 10 - h[x]), (60, 60, 200), 2)
        tx = int((thr[meth] - 1) / 254 * (W - 20)) + 10
        cv2.line(img, (tx, r * H + 5), (tx, r * H + H - 10), (0, 160, 0), 1)
        cv2.putText(img, f'{meth} thr={thr[meth]} (log count, 1..254)', (12, r * H + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(C.REPORTS, 's6_mask_hist.png'), img)
    out = {'thresholds': thr, 'rule': 'val 정렬 마스크 값 1~254 히스토그램(log1p, 5칸 이동평균)의 [4,100] 최소 위치',
           'area_frac_at_thr': {}}
    te = C.load_index('test')
    row_of = {(c, v, int(f)): r for r, (c, v, f) in enumerate(zip(te['cat'], te['video'], te['frame']))}
    for sp in ('val', 'test'):
        z = za[sp]
        ok = np.where(z['ok'])[0]
        meth = z['meth'][ok]
        t = np.array([thr[m] for m in meth])
        m224 = np.asarray(mk[sp][ok]) >= t[:, None, None]
        gt11 = np.stack([cv2.resize(x.astype(np.float32), (11, 11), interpolation=cv2.INTER_AREA) for x in m224])
        d = {'meth': meth, 'key': z['key'][ok], 'i': z['i'][ok], 'gt11': gt11.astype(np.float16),
             'gt224': np.packbits(m224, axis=-1)}
        if sp == 'test':
            d['cache_row'] = np.array([row_of[(c, k, int(i))] for c, k, i in zip(meth, z['key'][ok], z['i'][ok])])
        np.savez(os.path.join(MD, f'{sp}_loc.npz'), **d)
        out['area_frac_at_thr'][sp] = {mm: float(m224[meth == mm].mean()) for mm in C.METHODS}
        out[f'{sp}_frames'] = int(len(ok))
        out[f'{sp}_frames_total'] = int(len(z['ok']))
    # val 정렬 프레임 baseline (이미지·b3·logit)
    zv = np.load(os.path.join(MD, 'val_loc.npz'))
    imgs = []
    for meth, key, i in zip(zv['meth'], zv['key'], zv['i']):
        with h5py.File(C.h5_path(meth, 'val'), 'r') as f:
            imgs.append(f[key][int(i)])
    imgs = np.stack(imgs)
    import torch
    model = C.load_baseline('cuda', verbose=False)
    b3s, lgs = [], []
    for s in range(0, len(imgs), 32):
        x = torch.from_numpy(imgs[s:s + 32].astype(np.float32) / 255.).permute(0, 3, 1, 2).cuda()
        lg, b = C.baseline_forward(model, x, amp=True)
        b3s.append(b.cpu().numpy())
        lgs.append(lg.cpu().numpy())
    np.save(os.path.join(MD, 'val_img.npy'), imgs)
    np.save(os.path.join(MD, 'val_b3.npy'), np.concatenate(b3s))
    np.save(os.path.join(MD, 'val_logit.npy'), np.concatenate(lgs))
    json.dump(out, open(os.path.join(C.REPORTS, 's6_thresholds.json'), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
