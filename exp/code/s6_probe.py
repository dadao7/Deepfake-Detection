"""S6 탐색: h5 224 얼굴 프레임 ↔ 위조 c40 영상의 (프레임 번호 t, 정사각 crop 상자) 추정 — 1/4 해상도 다중 스케일 템플릿 매칭
  python code/s6_probe.py Deepfakes 953_974 test"""
import os
import sys
import time

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

DL = os.path.join(C.DATA, 'FFPP_dl', 'manipulated_sequences')


def read_video(path, f=4, gray=True):
    cap = cv2.VideoCapture(path)
    out = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if gray:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        out.append(cv2.resize(fr, (fr.shape[1] // f, fr.shape[0] // f), interpolation=cv2.INTER_AREA) if f > 1 else fr)
    return out


def best_match(frames, tmpl224, sizes, f=4):
    res = []
    for t, fr in enumerate(frames):
        for s in sizes:
            ts = max(8, int(round(s / f)))
            if ts >= min(fr.shape):
                continue
            tm = cv2.resize(tmpl224, (ts, ts), interpolation=cv2.INTER_AREA)
            r = cv2.matchTemplate(fr, tm, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            res.append((mx, t, s, loc[0] * f, loc[1] * f))
    res.sort(reverse=True)
    return res


def main():
    meth, key, split = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    vids = read_video(os.path.join(DL, meth, 'c40', 'videos', f'{key}.mp4'))
    H, W = vids[0].shape[0] * 4, vids[0].shape[1] * 4
    with h5py.File(C.h5_path(meth, split), 'r') as f:
        h5 = f[key][:]
    print(f'{meth}/{key}: 영상 {len(vids)}프레임 {W}x{H}, h5 {h5.shape}, 읽기 {time.time() - t0:.1f}s', flush=True)
    sizes = list(range(120, min(H, W), 16))
    for i in (0, 1, 2, 10, 50, 99):
        if i >= len(h5):
            continue
        g = cv2.cvtColor(h5[i], cv2.COLOR_RGB2GRAY)
        t1 = time.time()
        r = best_match(vids, g, sizes)
        top = r[:3]
        # 같은 t에서 두 번째로 좋은 다른 t와의 점수 차
        best_t = top[0][1]
        other = next((x for x in r if abs(x[1] - best_t) > 2), None)
        print(f'  h5 {i:3d}: top ' + ' | '.join(f't={t} s={s} xy=({x},{y}) {m:.3f}' for m, t, s, x, y in top) +
              f' | 다른 t 최고 {other[0]:.3f}(t={other[1]})  {time.time() - t1:.1f}s', flush=True)


if __name__ == '__main__':
    main()
