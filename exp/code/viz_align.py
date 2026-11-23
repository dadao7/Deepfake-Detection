"""S6 게이트용: 정렬 샘플 격자 (h5 프레임 | 정렬한 마스크 원값 | 겹쳐 그리기) — 로컬 확인용(FF++ 이미지 공개 금지)
  python code/viz_align.py test_probe3 [n=16]"""
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT = os.path.join(C.CACHE, 'masks')


def main():
    tag = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    split = tag.split('_')[0]
    z = np.load(os.path.join(OUT, f'{tag}_align.npz'))
    masks = np.load(os.path.join(OUT, f'{tag}_mask.npy'), mmap_mode='r')
    rng = np.random.default_rng(0)
    bad = np.where(~z['ok'])[0]
    good = np.where(z['ok'])[0]
    idx = list(rng.choice(good, min(n, len(good)), replace=False)) + list(bad[:6])
    tiles = []
    for j in idx:
        meth, key, i = str(z['meth'][j]), str(z['key'][j]), int(z['i'][j])
        with h5py.File(C.h5_path(meth, split), 'r') as f:
            fr = f[key][i]
        m = np.asarray(masks[j])
        ov = fr.copy()
        edge = cv2.Canny((m > 20).astype(np.uint8) * 255, 50, 150) > 0
        ov[edge] = (0, 255, 0)
        ov[(m > 20) & ~edge] = (ov[(m > 20) & ~edge] * 0.6 + np.array([255, 0, 0]) * 0.4).astype(np.uint8)
        mv = cv2.cvtColor(cv2.equalizeHist(m) if m.max() > 0 else m, cv2.COLOR_GRAY2RGB)
        row = np.concatenate([fr, mv, ov], 1)
        cv2.putText(row, f'{C.METHOD_SHORT[meth]} {key} i{i} t{int(z["t"][j])} inl{int(z["inl"][j])} '
                    f'ncc{float(z["ncc"][j]):.3f} max{int(m.max())} {"OK" if z["ok"][j] else "FAIL"}',
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)
        tiles.append(row)
    cols = 2
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.concatenate([np.concatenate(tiles[r:r + cols], 1) for r in range(0, len(tiles), cols)], 0)
    small = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2), interpolation=cv2.INTER_AREA)
    p = os.path.join(C.REPORTS, f's6_align_grid_{tag}.png')
    cv2.imwrite(p, cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    print('저장', p, small.shape, '| 실패', len(bad), '개')


if __name__ == '__main__':
    main()
