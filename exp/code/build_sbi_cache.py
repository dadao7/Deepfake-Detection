"""S3-1: SBI 쌍 생성 + baseline 출력 캐시 (분리 실행 + 이어받기)
train: 공식 train 원본 중 랜드마크 있는 523영상의 모든 랜드마크 프레임(h5 프레임 10k) × 변형 V=4
val  : 공식 val 원본 88영상 × 변형 1 (고정 seed)
표본 난수 = default_rng([SBI_SEED, 영상번호, k, 변형]) → 작업 순서·프로세스 수와 무관하게 결정적.
출력(cache/sbi/): {split}_img.npy (N,2,224,224,3) uint8 [진짜, SBI], {split}_mask.npy (N,224,224) uint8 (비율 곱하기 전 M×255),
      {split}_gt.npy (N,11,11) fp16, {split}_meta.npy (ratio,use68,src_aug,flip,jpeg), {split}_meta_rows.npz (video,k,variant), {split}_b3.npy (N,2,272,11,11) fp16, {split}_logit.npy (N,2,2) fp32
실행: python code/build_sbi_cache.py [gen [프로세스 수] | base]   (기본: gen → base)"""
import json
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SBI_DIR = os.path.join(C.CACHE, 'sbi')
SBI_SEED = 20261015
V_TRAIN = 4
GEN_CHUNK = 256
BS = 32


def sample_list(split):
    from viz_sbi import load_landmarks
    lms = load_landmarks()
    ids = sorted({x for p in C.load_pairs(split) for x in p} & set(lms))
    V = V_TRAIN if split == 'train' else 1
    rows = [(v, k, var) for v in ids for k in sorted(lms[v]) for var in range(V)]
    return rows, lms


def _gen_chunk(args):
    split, rows = args
    if C.ORIG_CODE not in sys.path:
        sys.path.append(C.ORIG_CODE)
    import sbi
    from viz_sbi import load_landmarks
    global _LMS
    if '_LMS' not in globals():
        _LMS = load_landmarks()
    out_img = np.empty((len(rows), 2, 224, 224, 3), np.uint8)
    out_mask = np.empty((len(rows), 224, 224), np.uint8)
    out_gt = np.empty((len(rows), 11, 11), np.float16)
    meta = []
    with h5py.File(C.h5_path('original', 'train'), 'r') as f:     # train·val 원본 모두 전체 h5에 있음
        for i, (v, k, var) in enumerate(rows):
            img = f[v][10 * k]
            rng = np.random.default_rng([SBI_SEED, int(v), int(k), int(var)])
            real, fake, m, info = sbi.self_blend(img, _LMS[v][k], rng)
            out_img[i, 0], out_img[i, 1] = real, fake
            out_mask[i] = np.round(m * 255).astype(np.uint8)
            out_gt[i] = sbi.boundary_gt(m).astype(np.float16)
            meta.append([info['ratio'], int(info['use68']), int(info['src_aug']), int(info.get('flip', False)),
                         int(info.get('jpeg', 0))])
    return out_img, out_mask, out_gt, np.array(meta, np.float32)


def open_arr(path, dtype, shape):
    if os.path.exists(path):
        a = np.lib.format.open_memmap(path, mode='r+')
        assert a.shape == shape and a.dtype == dtype, (path, a.shape, shape)
        return a
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def prog(name):
    p = os.path.join(SBI_DIR, f'progress_{name}.json')
    return json.load(open(p))['done'] if os.path.exists(p) else 0


def set_prog(name, done, N):
    json.dump({'done': int(done), 'N': int(N), 'time': time.strftime('%H:%M:%S')},
              open(os.path.join(SBI_DIR, f'progress_{name}.json'), 'w'))


def gen(split, workers=12):
    rows, _ = sample_list(split)
    N = len(rows)
    np.savez(os.path.join(SBI_DIR, f'{split}_meta_rows.npz'), video=np.array([r[0] for r in rows]),
             k=np.array([r[1] for r in rows], np.int16), variant=np.array([r[2] for r in rows], np.int8))
    done = prog(f'gen_{split}')
    if done >= N:
        print(f'[gen {split}] 이미 완료 {N}', flush=True)
        return
    img = open_arr(os.path.join(SBI_DIR, f'{split}_img.npy'), np.uint8, (N, 2, 224, 224, 3))
    mask = open_arr(os.path.join(SBI_DIR, f'{split}_mask.npy'), np.uint8, (N, 224, 224))
    gt = open_arr(os.path.join(SBI_DIR, f'{split}_gt.npy'), np.float16, (N, 11, 11))
    mpath = os.path.join(SBI_DIR, f'{split}_meta.npy')
    meta = open_arr(mpath, np.float32, (N, 5))
    chunks = [(split, rows[s:s + GEN_CHUNK]) for s in range(done, N, GEN_CHUNK)]
    t0 = time.time()
    print(f'[gen {split}] {done}/{N}부터, 프로세스 {workers}', flush=True)
    with Pool(workers) as pool:
        s = done
        for oi, om, og, ome in pool.imap(_gen_chunk, chunks):      # 순서 유지
            e = s + len(oi)
            img[s:e], mask[s:e], gt[s:e], meta[s:e] = oi, om, og, ome
            s = e
            if (s - done) % (GEN_CHUNK * 20) < GEN_CHUNK or s == N:
                img.flush(); mask.flush(); gt.flush(); meta.flush()
                set_prog(f'gen_{split}', s, N)
                print(f'[gen {split}] {s}/{N} {(time.time() - t0) / 60:.1f}분', flush=True)
    img.flush(); mask.flush(); gt.flush(); meta.flush()
    set_prog(f'gen_{split}', N, N)


def base(split, model):
    import torch
    N = len(np.load(os.path.join(SBI_DIR, f'{split}_meta_rows.npz'))['k'])
    assert prog(f'gen_{split}') == N
    done = prog(f'base_{split}')
    if done >= N:
        print(f'[base {split}] 이미 완료 {N}', flush=True)
        return
    img = np.load(os.path.join(SBI_DIR, f'{split}_img.npy'), mmap_mode='r')
    b3 = open_arr(os.path.join(SBI_DIR, f'{split}_b3.npy'), np.float16, (N, 2, 272, 11, 11))
    lg = open_arr(os.path.join(SBI_DIR, f'{split}_logit.npy'), np.float32, (N, 2, 2))
    t0 = time.time()
    last = done
    half = BS // 2
    for s in range(done, N, half):                       # 쌍 16개 = 이미지 32장
        e = min(N, s + half)
        x = torch.from_numpy(np.ascontiguousarray(img[s:e]).reshape(-1, 224, 224, 3).astype(np.float32) / 255.)
        x = x.permute(0, 3, 1, 2).cuda()
        logit, b = C.baseline_forward(model, x, amp=True)
        assert torch.isfinite(logit).all() and torch.isfinite(b).all()
        b3[s:e] = b.cpu().numpy().reshape(e - s, 2, 272, 11, 11)
        lg[s:e] = logit.cpu().numpy().reshape(e - s, 2, 2)
        if e - last >= max(1000, N // 20) or e == N:
            b3.flush(); lg.flush()
            set_prog(f'base_{split}', e, N)
            el = time.time() - t0
            print(f'[base {split}] {e}/{N} {el / 60:.1f}분, 남은 약 {el / (e - done) * (N - e) / 60:.1f}분', flush=True)
            last = e


def main():
    """인자: gen [프로세스 수] | base | (없음 = 둘 다)"""
    os.makedirs(SBI_DIR, exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if phase in ('gen', 'all'):
        workers = int(sys.argv[2]) if len(sys.argv) > 2 else 12
        for split in ('val', 'train'):
            gen(split, workers)
    if phase in ('base', 'all'):
        import torch
        torch.backends.cudnn.benchmark = False
        model = C.load_baseline('cuda')
        for split in ('val', 'train'):
            base(split, model)
    print(f'SBI 캐시 {phase} 끝', flush=True)


if __name__ == '__main__':
    main()
