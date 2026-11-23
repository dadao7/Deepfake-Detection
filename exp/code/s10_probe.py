"""S10 탐색: 우리 정렬 crop vs h5 crop NCC (점 집합 비교), FF++ test 위조 c40 일부"""
import os, sys, time
import numpy as np, h5py, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C, face_align as FA
DL = os.path.join(C.DATA, 'FFPP_dl', 'manipulated_sequences')
def ncc(a, b, m=16):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)[m:-m, m:-m].astype(np.float64); b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)[m:-m, m:-m].astype(np.float64)
    a -= a.mean(); b -= b.mean(); return float((a*b).sum() / np.sqrt((a*a).sum()*(b*b).sum()))
res = {k: [] for k in FA.SUBSETS}; nofail = 0; t0 = time.time()
test = C.load_pairs('test')
for meth in C.METHODS:
    for key in C.videos_of_pairs(test, meth)[:4]:
        with h5py.File(C.h5_path(meth, 'test'), 'r') as f:
            N = f[key].shape[0]; h5 = f[key][:100]
        cap = cv2.VideoCapture(os.path.join(DL, meth, 'c40', 'videos', f'{key}.mp4')); T = int(cap.get(7)); cap.release()
        idx = list(range(0, 100, 12)); want = {int(i * T / N): i for i in idx}
        frames, _ = FA.read_frames(os.path.join(DL, meth, 'c40', 'videos', f'{key}.mp4'), set(want))
        for t, i in want.items():
            pts = FA.landmarks(frames[t])
            if pts is None: nofail += 1; continue
            for k in FA.SUBSETS:
                crop, _ = FA.align(frames[t], k, pts=pts); res[k].append(ncc(crop, h5[i]))
for k, v in res.items():
    v = np.array(v); print(f'{k:7s} n={len(v)} NCC 중앙값 {np.median(v):.4f} 평균 {v.mean():.4f} 10%분위 {np.quantile(v,0.1):.4f} ≥0.95 비율 {(v>=0.95).mean():.2f}')
print('얼굴 못 찾음', nofail, f'{time.time()-t0:.0f}s')
