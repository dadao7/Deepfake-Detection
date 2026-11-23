"""S6 탐색 2: 원 해상도에서 (1/4 해상도로 찾은) 상자 주변 ROI만 잘라, 모든 영상 프레임 t에 대해 h5 프레임 템플릿 매칭.
정답 t가 뚜렷한 최대(다른 t와 점수 차)인지, h5 번호 i → t 규칙이 무엇인지 본다.
  python code/s6_probe2.py Deepfakes 953_974 test"""
import os
import sys
import time

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from s6_probe import DL, best_match, read_video  # noqa: E402


def main():
    meth, key, split = sys.argv[1], sys.argv[2], sys.argv[3]
    small = read_video(os.path.join(DL, meth, 'c40', 'videos', f'{key}.mp4'), f=4)
    full = read_video(os.path.join(DL, meth, 'c40', 'videos', f'{key}.mp4'), f=1, gray=False)
    H, W = full[0].shape[:2]
    with h5py.File(C.h5_path(meth, split), 'r') as f:
        h5 = f[key][:]
    # 1) 1/4 해상도로 상자 대략값 (h5 0번, 크기 범위 전체)
    g0 = cv2.cvtColor(h5[0], cv2.COLOR_RGB2GRAY)
    m, t, s, x, y = best_match(small[::10], g0, list(range(96, min(H, W), 16)))[0]
    print(f'{meth}/{key} {W}x{H} T={len(full)} h5={len(h5)} | 대략 상자 s={s} xy=({x},{y}) {m:.3f}', flush=True)
    pad = 48
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + s + pad), min(H, y + s + pad)
    rois = [fr[y0:y1, x0:x1] for fr in full]
    out = []
    for i in list(range(0, len(h5), max(1, len(h5) // 12))):
        t1 = time.time()
        best = []
        for t, roi in enumerate(rois):
            bt = (-1, 0, 0, 0)
            for ss in range(s - 12, s + 13, 4):
                if ss >= min(roi.shape[:2]):
                    continue
                tm = cv2.resize(h5[i][:, :, ::-1], (ss, ss), interpolation=cv2.INTER_AREA)
                r = cv2.matchTemplate(roi, tm, cv2.TM_CCOEFF_NORMED)
                _, mx, _, loc = cv2.minMaxLoc(r)
                if mx > bt[0]:
                    bt = (mx, ss, loc[0] + x0, loc[1] + y0)
            best.append((bt[0], t, bt[1], bt[2], bt[3]))
        sc = np.array([b[0] for b in best])
        o = np.argsort(-sc)
        t_best = int(o[0])
        far = [j for j in o if abs(j - t_best) > 3]
        print(f'  i={i:3d} → t={t_best:3d} s={best[t_best][2]} xy=({best[t_best][3]},{best[t_best][4]}) '
              f'점수 {sc[t_best]:.4f} | 2위(±3 밖) t={far[0]} {sc[far[0]]:.4f} | 이웃 t±1 '
              f'{sc[max(0, t_best - 1)]:.4f}/{sc[min(len(sc) - 1, t_best + 1)]:.4f} | {time.time() - t1:.1f}s', flush=True)
        out.append((i, t_best))
    print('i→t:', out)


if __name__ == '__main__':
    main()
