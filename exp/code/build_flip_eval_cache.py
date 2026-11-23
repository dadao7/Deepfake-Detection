"""TASKS5 M5(TTA 준비): 평가·선택 세트 이미지를 좌우 반전해 저자 baseline 출력(b_3, logit, CLIP CLS)을 다시 계산한 캐시.
학습용 train_flip과 같은 방식(이미 캐시된 224 이미지를 torch.flip(dims=[3]) → C.baseline_forward, autocast fp16, BS 32).
CLS는 build_newdata_cache.py와 같은 vision_proj 입력 훅으로 받는다. 분리 실행 + 이어받기(progress json).
게이트: 시작 전에 val 앞 64장을 반전 없이 계산해 기존 val 캐시(b_3·logit·CLS)와 같은지 확인.
출력: cache/base/{val|test|cdf}_flip_{b3,logit,cls}.npy, cache/newdata/{fsh|dfd}_flip_{b3,logit,cls}.npy
  python code/build_flip_eval_cache.py [val test cdf fsh dfd]"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ND = os.path.join(C.CACHE, 'newdata')
PROG = os.path.join(C.CACHE, 'progress')
JOBS = {'val': ('ff', 'val'), 'test': ('ff', 'test'), 'cdf': ('ff', 'cdf'), 'fsh': ('nd', 'fsh'), 'dfd': ('nd', 'dfd')}


def paths(name):
    kind, sp = JOBS[name]
    if kind == 'ff':
        return os.path.join(C.FRAME_DIR, f'{sp}_img.npy'), os.path.join(C.BASE_DIR, f'{sp}_flip_{{}}.npy')
    return os.path.join(ND, f'{sp}_img.npy'), os.path.join(ND, f'{sp}_flip_{{}}.npy')


def arr(p, dt, shp):
    if os.path.exists(p):
        a = np.lib.format.open_memmap(p, mode='r+')
        assert a.shape == shp and a.dtype == dt, (p, a.shape, shp)
        return a
    return np.lib.format.open_memmap(p, mode='w+', dtype=dt, shape=shp)


def forward(model, buf, frames, flip):
    x = torch.from_numpy(np.ascontiguousarray(frames).astype(np.float32) / 255.).permute(0, 3, 1, 2).cuda()
    if flip:
        x = torch.flip(x, dims=[3])
    logit, b = C.baseline_forward(model, x, amp=True)
    assert torch.isfinite(logit).all() and torch.isfinite(b).all()
    return b.cpu().numpy(), logit.cpu().numpy(), buf['cls'].half().cpu().numpy()


def gate(model, buf):
    img = np.load(os.path.join(C.FRAME_DIR, 'val_img.npy'), mmap_mode='r')
    b, lg, cl = forward(model, buf, img[:32], False)
    b2, lg2, cl2 = forward(model, buf, img[32:64], False)
    b, lg, cl = np.concatenate([b, b2]), np.concatenate([lg, lg2]), np.concatenate([cl, cl2])
    rb = np.load(os.path.join(C.BASE_DIR, 'val_b3.npy'), mmap_mode='r')[:64]
    rl = np.load(os.path.join(C.BASE_DIR, 'val_logit.npy'), mmap_mode='r')[:64]
    rc = np.load(os.path.join(C.BASE_DIR, 'val_cls.npy'), mmap_mode='r')[:64]
    d = [float(np.abs(b.astype(np.float32) - rb.astype(np.float32)).max()), float(np.abs(lg - rl).max()),
         float(np.abs(cl.astype(np.float32) - rc.astype(np.float32)).max())]
    print('게이트(반전 없이 val 64장 재계산 vs 기존 캐시) 최대 차 b3/logit/cls:', d, flush=True)
    assert max(d) < 1e-2, '기존 캐시와 다름 → 멈춤'


def run(model, buf, name):
    ip, op = paths(name)
    img = np.load(ip, mmap_mode='r')
    n = len(img)
    pf = os.path.join(PROG, f'{name}_flip_eval.json')
    done = json.load(open(pf))['done'] if os.path.exists(pf) else 0
    if done >= n:
        print(f'[{name}] 이미 완료 ({n})', flush=True)
        return
    b3, lg, cl = arr(op.format('b3'), np.float16, (n, 272, 11, 11)), arr(op.format('logit'), np.float32, (n, 2)), \
        arr(op.format('cls'), np.float16, (n, 1024))
    t0, last = time.time(), done
    for s in range(done, n, 32):
        e = min(n, s + 32)
        b3[s:e], lg[s:e], cl[s:e] = forward(model, buf, img[s:e], True)
        if e - last >= max(3200, n // 20) or e == n:
            b3.flush(); lg.flush(); cl.flush()
            json.dump({'done': int(e), 'N': n, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}, open(pf, 'w'))
            el = time.time() - t0
            print(f'[{name}] {e}/{n} {el / 60:.1f}분, 남은 약 {el / (e - done) * (n - e) / 60:.1f}분', flush=True)
            last = e


def main():
    names = sys.argv[1:] or list(JOBS)
    torch.backends.cudnn.benchmark = False
    model = C.load_baseline('cuda')
    buf = {}
    model.vision_proj.register_forward_pre_hook(lambda m, inp: buf.__setitem__('cls', inp[0][:, 0].detach()))
    gate(model, buf)
    t0 = time.time()
    for name in names:
        run(model, buf, name)
    print(f'끝 {(time.time() - t0) / 60:.1f}분', flush=True)


if __name__ == '__main__':
    main()
