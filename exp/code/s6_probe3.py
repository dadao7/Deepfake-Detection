"""S6 탐색 3: SIFT + RANSAC 닮음변환(estimateAffinePartial2D)으로 h5 프레임 ↔ 영상 프레임 t 대응과 변환(배율·회전·이동) 추정.
정답 t에서 인라이어가 뚜렷이 많고 변환이 일관적이면 → 마스크를 같은 변환으로 옮겨 정렬할 수 있다.
  python code/s6_probe3.py Deepfakes 953_974 test"""
import math
import os
import sys
import time

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from s6_probe import DL, best_match, read_video  # noqa: E402

SIFT = cv2.SIFT_create(nfeatures=800)
BF = cv2.BFMatcher(cv2.NORM_L2)


def feats(gray):
    kp, des = SIFT.detectAndCompute(gray, None)
    return kp, des


def match(kq, dq, kt, dt, off):
    if dq is None or dt is None or len(kt) < 8:
        return 0, None
    ms = BF.knnMatch(dq, dt, k=2)
    good = [m for m, n in (p for p in ms if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 6:
        return len(good), None
    src = np.float32([kt[m.trainIdx].pt for m in good]) + np.float32(off)      # 영상 전체 좌표
    dst = np.float32([kq[m.queryIdx].pt for m in good])                        # h5 좌표
    A, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000)
    return (int(inl.sum()) if inl is not None else 0), A


def main():
    meth, key, split = sys.argv[1], sys.argv[2], sys.argv[3]
    small = read_video(os.path.join(DL, meth, 'c40', 'videos', f'{key}.mp4'), f=4)
    full = read_video(os.path.join(DL, meth, 'c40', 'videos', f'{key}.mp4'), f=1, gray=True)
    H, W = full[0].shape[:2]
    with h5py.File(C.h5_path(meth, split), 'r') as f:
        h5 = f[key][:]
    m, t, s, x, y = best_match(small[::10], cv2.cvtColor(h5[0], cv2.COLOR_RGB2GRAY), list(range(96, min(H, W), 16)))[0]
    pad = int(s * 0.6)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + s + pad), min(H, y + s + pad)
    t0 = time.time()
    vf = [feats(fr[y0:y1, x0:x1]) for fr in full]
    print(f'{meth}/{key} {W}x{H} T={len(full)} | 상자 대략 s={s} ({x},{y}) | 영상 특징 {time.time() - t0:.1f}s', flush=True)
    res = []
    for i in list(range(0, len(h5), max(1, len(h5) // 10))):
        kq, dq = feats(cv2.cvtColor(h5[i], cv2.COLOR_RGB2GRAY))
        sc = []
        for t, (kt, dt) in enumerate(vf):
            n, A = match(kq, dq, kt, dt, (x0, y0))
            sc.append((n, t, A))
        sc.sort(key=lambda z: -z[0])
        n1, tb, A = sc[0]
        far = next(z for z in sc if abs(z[1] - tb) > 3)
        if A is not None:
            scale = math.hypot(A[0, 0], A[1, 0])
            rot = math.degrees(math.atan2(A[1, 0], A[0, 0]))
            info = f'배율 {scale:.3f} 회전 {rot:+.2f}° 이동 ({A[0, 2]:.1f},{A[1, 2]:.1f})'
        else:
            info = '변환 없음'
        print(f'  i={i:3d} → t={tb:3d} 인라이어 {n1:3d} (h5 특징 {len(kq)}) | ±3 밖 최고 t={far[1]} {far[0]} | {info}', flush=True)
        res.append((i, tb, n1, far[0]))
    print('i→t:', [(a, b) for a, b, _, _ in res])


if __name__ == '__main__':
    main()
