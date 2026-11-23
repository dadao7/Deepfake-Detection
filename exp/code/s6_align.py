"""S6-1: FF++ 위조 마스크를 h5 얼굴 프레임(224, 회전·배율 정렬된 crop)에 정렬 (PROTOCOL §11)
탐색 결과(s6_probe3): h5 프레임 i ↔ c40 영상 프레임 t = floor(i·T/N) (균등 추출, N = h5 프레임 수, 보통 100), 얼굴은 닮음변환(회전 ~±8°, 배율 ~1.4)으로 정렬된 crop.
방법: 프레임 t의 1/4 해상도 다중 스케일 매칭으로 ROI → SIFT + RANSAC 닮음변환 A(영상→h5) → 같은 A로 마스크 프레임을 224에 옮김.
품질: 인라이어 수, 정렬한 영상 프레임과 h5 프레임의 NCC(가운데 192×192). 기준 미달 프레임은 제외(목록 기록).
대상: val·test의 가짜 영상(방식 4 × 140), 영상당 h5 프레임 i ∈ {0,10,…,90}. 진짜 영상 마스크는 전부 0.
출력: cache/masks/{split}_mask.npy (N,224,224) uint8 (정렬된 원 마스크 값), {split}_align.npz (방식·영상·i·t·인라이어·NCC·A·ok)
실행: python code/s6_align.py [split ...] [--workers 8] [--limit 영상수]"""
import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

DL = os.path.join(C.DATA, 'FFPP_dl', 'manipulated_sequences')
OUT = os.path.join(C.CACHE, 'masks')
FRAMES_I = list(range(0, 100, 10))
MIN_INL, MIN_NCC = 6, 0.95   # 탐색: 흐린 얼굴은 인라이어 9~10이어도 NCC 1.000(정확) → NCC 위주 판정


def read_frames(path, want):
    cap = cv2.VideoCapture(path)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out, t, mx = {}, 0, max(want)
    while t <= mx:
        ok, fr = cap.read()
        if not ok:
            break
        if t in want:
            out[t] = fr
        t += 1
    cap.release()
    return out, T


def coarse_box(gray_full, tmpl_gray, f=4):
    small = cv2.resize(gray_full, (gray_full.shape[1] // f, gray_full.shape[0] // f), interpolation=cv2.INTER_AREA)
    best = (-1, 0, 0, 0)
    for s in range(96, min(gray_full.shape), 12):
        ts = int(round(s / f))
        if ts >= min(small.shape):
            break
        tm = cv2.resize(tmpl_gray, (ts, ts), interpolation=cv2.INTER_AREA)
        r = cv2.matchTemplate(small, tm, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (mx, s, loc[0] * f, loc[1] * f)
    return best


def ncc_center(a, b, m=16):
    a = a[m:-m, m:-m].astype(np.float64)
    b = b[m:-m, m:-m].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


_SIFT = None


def similarity(h5_gray, vid_gray, box):
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=1000)
    H, W = vid_gray.shape
    _, s, x, y = box
    pad = int(s * 0.6)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    roi = vid_gray[y0:min(H, y + s + pad), x0:min(W, x + s + pad)]
    kq, dq = _SIFT.detectAndCompute(h5_gray, None)
    kt, dt = _SIFT.detectAndCompute(roi, None)
    if dq is None or dt is None or len(kq) < 6 or len(kt) < 6:
        return None, 0
    ms = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dq, dt, k=2)
    good = [p[0] for p in ms if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 6:
        return None, len(good)
    src = np.float32([kt[g.trainIdx].pt for g in good]) + np.float32([x0, y0])
    dst = np.float32([kq[g.queryIdx].pt for g in good])
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=3000,
                                         confidence=0.999)
    return A, int(inl.sum()) if inl is not None else 0


def align_video(args):
    split, meth, key = args
    with h5py.File(C.h5_path(meth, split), 'r') as f:
        N = f[key].shape[0]                       # 보통 100, 짧은 영상은 더 적음(val에 58장 영상 있음)
        h5 = f[key][:min(N, 100)]
    cap = cv2.VideoCapture(os.path.join(DL, meth, 'c40', 'videos', f'{key}.mp4'))
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    want = {int(i * T / N): i for i in FRAMES_I if i < N}     # 균등 추출 t = floor(i·T/N)
    vfr, T1 = read_frames(os.path.join(DL, meth, 'c40', 'videos', f'{key}.mp4'), set(want))
    mfr, T2 = read_frames(os.path.join(DL, meth, 'masks', 'videos', f'{key}.mp4'), set(want))
    rows = []
    for t, i in sorted(want.items(), key=lambda z: z[1]):
        rec = {'meth': meth, 'key': key, 'i': i, 't': t, 'T': T, 'T_mask': T2, 'inl': 0, 'ncc': -1.0, 'ok': False,
               'A': np.zeros((2, 3), np.float32), 'mask': np.zeros((224, 224), np.uint8)}
        if t in vfr and t in mfr:
            hg = cv2.cvtColor(h5[i], cv2.COLOR_RGB2GRAY)
            vg = cv2.cvtColor(vfr[t], cv2.COLOR_BGR2GRAY)
            H, W = vg.shape
            # 1차: 1/4 해상도 매칭 ROI 안 SIFT, 실패하면 2차: 화면 전체 SIFT(ROI 오검출 대비)
            for box in (coarse_box(vg, hg), (0, max(H, W), 0, 0)):
                A, inl = similarity(hg, vg, box)
                if A is None:
                    continue
                warped = cv2.warpAffine(vg, A, (224, 224), flags=cv2.INTER_LINEAR)
                ncc = ncc_center(warped, hg)
                if ncc > rec['ncc']:
                    mg = mfr[t] if mfr[t].ndim == 2 else cv2.cvtColor(mfr[t], cv2.COLOR_BGR2GRAY)
                    rec.update(inl=inl, ncc=ncc, A=A.astype(np.float32),
                               mask=cv2.warpAffine(mg, A, (224, 224), flags=cv2.INTER_LINEAR),
                               ok=bool(inl >= MIN_INL and ncc >= MIN_NCC))
                if rec['ok']:
                    break
        rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('splits', nargs='*', default=['val', 'test'])
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for split in args.splits:
        jobs = []
        for meth in C.METHODS:
            keys = C.videos_of_pairs(C.load_pairs(split), meth)
            if args.limit:
                keys = keys[:args.limit]
            jobs += [(split, meth, k) for k in keys]
        t0 = time.time()
        rows = []
        with Pool(args.workers) as pool:
            for n, r in enumerate(pool.imap(align_video, jobs, chunksize=2)):
                rows += r
                if (n + 1) % 80 == 0:
                    print(f'[{split}] {n + 1}/{len(jobs)} 영상, {(time.time() - t0) / 60:.1f}분', flush=True)
        tag = split if not args.limit else f'{split}_probe{args.limit}'
        np.save(os.path.join(OUT, f'{tag}_mask.npy'), np.stack([r['mask'] for r in rows]))
        np.savez(os.path.join(OUT, f'{tag}_align.npz'), meth=np.array([r['meth'] for r in rows]),
                 key=np.array([r['key'] for r in rows]), i=np.array([r['i'] for r in rows], np.int16),
                 t=np.array([r['t'] for r in rows], np.int32), T=np.array([r['T'] for r in rows], np.int32),
                 T_mask=np.array([r['T_mask'] for r in rows], np.int32), inl=np.array([r['inl'] for r in rows], np.int32),
                 ncc=np.array([r['ncc'] for r in rows], np.float32), ok=np.array([r['ok'] for r in rows]),
                 A=np.stack([r['A'] for r in rows]))
        ok = np.array([r['ok'] for r in rows])
        ncc = np.array([r['ncc'] for r in rows])
        print(f'[{split}] 끝 {len(rows)}프레임, 통과 {ok.mean() * 100:.1f}%, NCC 중앙값 {np.median(ncc):.3f}, '
              f'{(time.time() - t0) / 60:.1f}분', flush=True)
        for meth in C.METHODS:
            sel = np.array([r['meth'] == meth for r in rows])
            print(f'   {meth}: 통과 {ok[sel].mean() * 100:.1f}%  NCC 중앙값 {np.median(ncc[sel]):.3f}', flush=True)


if __name__ == '__main__':
    main()
