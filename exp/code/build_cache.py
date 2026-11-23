"""S1-2: 이미지·baseline 출력 캐시 (PROTOCOL §4.1). 분리 실행 + 이어받기.
작업 순서: test → val → cdf → train → train_flip (train_flip은 train 이미지를 좌우반전해 baseline만 다시 계산)
출력: cache/frames/{split}_img.npy (uint8 N×224×224×3, RGB), cache/base/{name}_b3.npy (fp16 N×272×11×11),
      cache/base/{name}_logit.npy (fp32 N×2), cache/progress/{name}.json, 끝나면 cache/MANIFEST.json
실행: python code/build_cache.py [작업이름 ...]"""
import hashlib
import json
import os
import sys
import time

import cv2
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

BS = 32
JOBS = [('test', 'test', False), ('val', 'val', False), ('cdf', 'cdf', False),
        ('train', 'train', False), ('train_flip', 'train', True)]
PROG = os.path.join(C.CACHE, 'progress')
os.makedirs(PROG, exist_ok=True)
os.makedirs(C.FRAME_DIR, exist_ok=True)
os.makedirs(C.BASE_DIR, exist_ok=True)


def open_arr(path, dtype, shape):
    if os.path.exists(path):
        a = np.lib.format.open_memmap(path, mode='r+')
        assert a.shape == shape and a.dtype == dtype, (path, a.shape, shape, a.dtype, dtype)
        return a
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def read_ffpp(ix, rows, files):
    """rows(연속 구간)의 프레임을 h5에서 읽음. 같은 영상 프레임은 한 번에."""
    out = np.empty((len(rows), 224, 224, 3), np.uint8)
    i = 0
    while i < len(rows):
        r = rows[i]
        cat, vid = ix['cat'][r], ix['video'][r]
        j = i
        while j < len(rows) and ix['video'][rows[j]] == vid and ix['cat'][rows[j]] == cat:
            j += 1
        fr = ix['frame'][rows[i:j]]
        assert np.all(np.diff(fr) > 0)
        out[i:j] = files[cat][vid][fr.tolist()]
        i = j
    return out


def read_cdf(ix, rows):
    out = np.empty((len(rows), 224, 224, 3), np.uint8)
    for i, r in enumerate(rows):
        im = cv2.imread(os.path.join(C.CDF_DIR, ix['relpath'][r]))
        assert im is not None, ix['relpath'][r]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        out[i] = cv2.resize(im, (224, 224))          # INTER_LINEAR (1학기 평가와 같음)
    return out


def run_job(model, name, split, flip):
    ix = C.load_index(split)
    N = len(ix['label'])
    pfile = os.path.join(PROG, f'{name}.json')
    done = json.load(open(pfile))['done'] if os.path.exists(pfile) else 0
    if done >= N:
        print(f'[{name}] 이미 완료 ({N})', flush=True)
        return
    img_path = os.path.join(C.FRAME_DIR, f'{split}_img.npy')
    if flip:
        assert json.load(open(os.path.join(PROG, f'{split}.json')))['done'] == N, 'train 이미지 먼저'
        img = np.load(img_path, mmap_mode='r')
    else:
        img = open_arr(img_path, np.uint8, (N, 224, 224, 3))
    b3 = open_arr(os.path.join(C.BASE_DIR, f'{name}_b3.npy'), np.float16, (N, 272, 11, 11))
    lg = open_arr(os.path.join(C.BASE_DIR, f'{name}_logit.npy'), np.float32, (N, 2))
    files = {}
    if split != 'cdf' and not flip:
        files = {c: h5py.File(C.h5_path(c, split), 'r') for c in C.CATS}
    print(f'[{name}] {done}/{N}부터 시작', flush=True)
    t0 = time.time()
    last_report = done
    for s in range(done, N, BS):
        rows = np.arange(s, min(N, s + BS))
        if flip:
            frames = np.ascontiguousarray(img[rows])
        elif split == 'cdf':
            frames = read_cdf(ix, rows)
        else:
            frames = read_ffpp(ix, rows, files)
        x = torch.from_numpy(frames.astype(np.float32) / 255.).permute(0, 3, 1, 2).to('cuda')
        if flip:
            x = torch.flip(x, dims=[3])               # = torchvision hflip
        logit, b = C.baseline_forward(model, x, amp=True)
        assert torch.isfinite(logit).all() and torch.isfinite(b).all(), f'{name} 비유한 값 rows {s}'
        if not flip:
            img[rows] = frames
        b3[rows] = b.cpu().numpy()
        lg[rows] = logit.cpu().numpy()
        e = rows[-1] + 1
        if e - last_report >= max(2000, N // 20) or e == N:
            if not flip:
                img.flush()
            b3.flush()
            lg.flush()
            json.dump({'done': int(e), 'N': N, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}, open(pfile, 'w'))
            el = time.time() - t0
            print(f'[{name}] {e}/{N}  {el / 60:.1f}분, 남은 약 {el / (e - done) * (N - e) / 60:.1f}분', flush=True)
            last_report = e
    for f in files.values():
        f.close()
    del img, b3, lg


def sample_sha(path, rows):
    a = np.load(path, mmap_mode='r')
    h = hashlib.sha256()
    for r in rows:
        h.update(np.ascontiguousarray(a[r]).tobytes())
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'sample_sha': h.hexdigest()[:16]}


def write_manifest():
    man = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'index_sha': C.index_sha(), 'versions': C.env_versions(),
           'baseline': 'current_model_180.pth + vision_tower.pth, eval, autocast fp16, BS=32', 'arrays': {}}
    for name, split, flip in JOBS:
        N = len(C.load_index(split)['label'])
        rows = [0, N // 2, N - 1]
        paths = [os.path.join(C.BASE_DIR, f'{name}_b3.npy'), os.path.join(C.BASE_DIR, f'{name}_logit.npy')]
        if not flip:
            paths.append(os.path.join(C.FRAME_DIR, f'{split}_img.npy'))
        for p in paths:
            man['arrays'][os.path.relpath(p, C.CACHE)] = sample_sha(p, rows)
    json.dump(man, open(os.path.join(C.CACHE, 'MANIFEST.json'), 'w'), indent=1)
    print('MANIFEST 작성, cache_sha', C.cache_sha(), flush=True)


def main():
    names = sys.argv[1:] or [j[0] for j in JOBS]
    torch.backends.cudnn.benchmark = False
    model = C.load_baseline('cuda')
    t0 = time.time()
    for name, split, flip in JOBS:
        if name in names:
            run_job(model, name, split, flip)
    if all(os.path.exists(os.path.join(PROG, f'{j[0]}.json')) and
           json.load(open(os.path.join(PROG, f'{j[0]}.json')))['done'] == json.load(open(os.path.join(PROG, f'{j[0]}.json')))['N']
           for j in JOBS):
        write_manifest()
    print(f'전체 {(time.time() - t0) / 60:.1f}분, 끝', flush=True)


if __name__ == '__main__':
    main()
