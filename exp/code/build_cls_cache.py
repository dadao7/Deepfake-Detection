"""S9 준비: CLIP CLS 특징(vision_proj 입력, 투영 전 1024차원) 캐시 + bridge adapter 상수 (PROTOCOL §15). 분리 실행 + 이어받기.
기존 캐시 이미지로 저자 baseline을 같은 배치(32장, 같은 순서)로 다시 돌려 CLS를 뽑고, 다시 계산한 logit이 기존 캐시와 같은지 검사한다.
출력: cache/base/{name}_cls.npy (N,1024) fp16 [name: test, val, cdf, train, train_flip], cache/sbi/{split}_cls.npy (P,2,1024) fp16,
      cache/base/adapter_const.npy (128,) fp32 = clip_text_alpha·bridge_adapter_proj 출력(입력과 무관한 상수), logs/cls_check_{name}.json
  python code/build_cls_cache.py [작업이름 ...]"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

BS = 32
SBI_DIR = os.path.join(C.CACHE, 'sbi')
PROG = os.path.join(C.CACHE, 'progress')
JOBS = ['test', 'val', 'cdf', 'train', 'train_flip', 'sbi_val', 'sbi_train']
CHECK_DIR = C.LOGS   # 작업별 logs/cls_check_{name}.json (병행 실행 시 덮어쓰기 방지)


def open_arr(path, dtype, shape):
    if os.path.exists(path):
        a = np.lib.format.open_memmap(path, mode='r+')
        assert a.shape == shape and a.dtype == dtype, (path, a.shape, shape)
        return a
    return np.lib.format.open_memmap(path, mode='w+', dtype=dtype, shape=shape)


def main():
    names = sys.argv[1:] or JOBS
    torch.backends.cudnn.benchmark = False
    model = C.load_baseline('cuda')
    buf = {}
    model.vision_proj.register_forward_pre_hook(lambda m, inp: buf.__setitem__('cls', inp[0][:, 0].detach()))
    model.bridge_adapter_proj.register_forward_hook(lambda m, i, o: buf.__setitem__('ad', o.detach()))
    for name in names:
        if name.startswith('sbi_'):
            split = name[4:]
            img = np.load(os.path.join(SBI_DIR, f'{split}_img.npy'), mmap_mode='r')
            ref = np.load(os.path.join(SBI_DIR, f'{split}_logit.npy'), mmap_mode='r').reshape(-1, 2)
            P = img.shape[0]
            N = 2 * P
            out = open_arr(os.path.join(SBI_DIR, f'{split}_cls.npy'), np.float16, (P, 2, 1024))
            out2 = out.reshape(N, 1024)
            get = lambda s, e: np.ascontiguousarray(img[s // 2:e // 2]).reshape(-1, 224, 224, 3)  # noqa: E731
            step, flip = BS, False                                   # 쌍 16개 = 32장 (build_sbi_cache와 같은 묶음)
        else:
            split = 'train' if name == 'train_flip' else name
            img = np.load(os.path.join(C.FRAME_DIR, f'{split}_img.npy'), mmap_mode='r')
            ref = np.load(os.path.join(C.BASE_DIR, f'{name}_logit.npy'), mmap_mode='r')
            N = img.shape[0]
            out2 = open_arr(os.path.join(C.BASE_DIR, f'{name}_cls.npy'), np.float16, (N, 1024))
            out = out2
            get = lambda s, e: np.ascontiguousarray(img[s:e])  # noqa: E731
            step, flip = BS, name == 'train_flip'
        pfile = os.path.join(PROG, f'cls_{name}.json')
        done = json.load(open(pfile))['done'] if os.path.exists(pfile) else 0
        if done >= N:
            print(f'[{name}] 이미 완료', flush=True)
            continue
        print(f'[{name}] {done}/{N}부터', flush=True)
        cpath = os.path.join(CHECK_DIR, f'cls_check_{name}.json')
        check = json.load(open(cpath)) if os.path.exists(cpath) else {}
        t0, last, maxd = time.time(), done, check.get('logit_maxdiff', 0.0)
        for s in range(done, N, step):
            e = min(N, s + step)
            x = torch.from_numpy(get(s, e).astype(np.float32) / 255.).permute(0, 3, 1, 2).cuda()
            if flip:
                x = torch.flip(x, dims=[3])
            logit, _ = C.baseline_forward(model, x, amp=True)
            out2[s:e] = buf['cls'].half().cpu().numpy()
            maxd = max(maxd, float(np.abs(logit.cpu().numpy() - np.asarray(ref[s:e])).max()))
            a = (model.clip_text_alpha * buf['ad'].float()).float()
            assert float(a.std(0).max()) == 0.0, 'adapter 출력이 상수가 아님'
            if not os.path.exists(os.path.join(C.BASE_DIR, 'adapter_const.npy')):
                np.save(os.path.join(C.BASE_DIR, 'adapter_const.npy'), a[0].cpu().numpy())
            if e - last >= max(2000, N // 20) or e == N:
                out.flush()
                json.dump({'done': int(e), 'N': int(N)}, open(pfile, 'w'))
                json.dump({'logit_maxdiff': maxd, 'done': int(e)}, open(cpath, 'w'), indent=1)
                el = time.time() - t0
                print(f'[{name}] {e}/{N} {el / 60:.1f}분, 남은 약 {el / (e - done) * (N - e) / 60:.1f}분, '
                      f'logit 최대 차 {maxd:.2e}', flush=True)
                last = e
    print('CLS 캐시 끝', flush=True)


if __name__ == '__main__':
    main()
