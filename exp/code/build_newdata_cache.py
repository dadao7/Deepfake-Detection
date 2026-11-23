"""S10: 새 평가 데이터 캐시 (PROTOCOL §16). 1단계(CPU 다중 프로세스): 영상 → 얼굴 정렬 crop(face_align, 눈+바깥 입술 닮음변환)
2단계(GPU): crop → 저자 baseline 출력(b_3 fp16, logit fp32, CLIP CLS fp16). 이어받기 가능.
데이터셋: ffv = FF++ test(원본 140 + DF·F2F·FS·NT 560)를 우리 파이프라인으로(재현 검증용, i∈{0,3,…,93}, t=floor(i·T/N), N=h5 프레임 수)
          fsh = FF++ 원본 test 140 + FaceShifter test 140 (같은 프레임 규칙, N=100)
          dfd = DFD 원본 363 + 가짜 3,068 (t=floor((j+0.5)·T/32), 32장)
출력: cache/newdata/{ds}_img.npy (최대 크기 미리 할당, 앞 n행만 유효), {ds}_meta.npz, {ds}_b3.npy, {ds}_logit.npy, {ds}_cls.npy, {ds}_progress.json
  python code/build_newdata_cache.py crop ffv fsh dfd  /  python code/build_newdata_cache.py base ffv fsh dfd"""
import json
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

OUT = os.path.join(C.CACHE, 'newdata')
DL = os.path.join(C.DATA, 'FFPP_dl')
FF_I = list(range(0, 96, 3))           # 32장
WORKERS = 12


def video_list(ds):
    """→ [(경로, 라벨, 방식, 영상키, 프레임 규칙('h5', N) 또는 ('uni', 32))]"""
    test = C.load_pairs('test')
    rows = []
    if ds in ('ffv', 'fsh'):
        with h5py.File(C.h5_path('original', 'test'), 'r') as f:
            for k in C.videos_of_pairs(test, 'original'):
                N = f[k].shape[0] if ds == 'ffv' else 100
                rows.append((os.path.join(DL, 'original_sequences', 'youtube', 'c40', 'videos', f'{k}.mp4'), 0,
                             'original', k, ('h5', N)))
        meths = C.METHODS if ds == 'ffv' else ['FaceShifter']
        for m in meths:
            hf = h5py.File(C.h5_path(m, 'test'), 'r') if ds == 'ffv' else None
            for k in C.videos_of_pairs(test, m):
                N = hf[k].shape[0] if hf is not None else 100
                rows.append((os.path.join(DL, 'manipulated_sequences', m, 'c40', 'videos', f'{k}.mp4'), 1, m, k, ('h5', N)))
            if hf is not None:
                hf.close()
    elif ds == 'dfd':
        d = json.load(open(os.path.join(C.DATA, 'dl_meta', 'deepfake_detection_filenames.json')))
        rows += [(os.path.join(DL, 'original_sequences', 'actors', 'c40', 'videos', f'{n}.mp4'), 0, 'DFD_real', n,
                  ('uni', 32)) for n in d['actors']]
        rows += [(os.path.join(DL, 'manipulated_sequences', 'DeepFakeDetection', 'c40', 'videos', f'{n}.mp4'), 1,
                  'DFD_fake', n, ('uni', 32)) for n in d['DeepFakesDetection']]
    return rows


def _work(args):
    vi, (path, lab, meth, key, rule) = args
    import cv2
    import face_align as FA
    cap = cv2.VideoCapture(path)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if T <= 0:
        return vi, None, [], [], T, 'open_fail'
    if rule[0] == 'h5':
        pairs = [(int(i * T / rule[1]), i) for i in FF_I if i < rule[1]]
    else:
        pairs = [(int((j + 0.5) * T / rule[1]), j) for j in range(rule[1])]
    want = {t for t, _ in pairs}
    frames, _ = FA.read_frames(path, want)
    crops, ts, iis = [], [], []
    for t, i in pairs:
        if t not in frames:
            continue
        crop, _ = FA.align(frames[t])
        if crop is None:
            continue
        crops.append(crop)
        ts.append(t)
        iis.append(i)
    return vi, (np.stack(crops) if crops else None), ts, iis, T, 'ok'


def crop_phase(ds):
    os.makedirs(OUT, exist_ok=True)
    vids = video_list(ds)
    missing = [v[0] for v in vids if not os.path.exists(v[0])]
    assert not missing, f'{ds}: 없는 영상 {len(missing)}개 예 {missing[:2]}'
    maxn = sum(32 for _ in vids)
    pfile = os.path.join(OUT, f'{ds}_progress.json')
    prog = json.load(open(pfile)) if os.path.exists(pfile) else {'video_done': 0, 'n': 0, 'crop_done': False}
    if prog.get('crop_done'):
        print(f'[{ds}] crop 이미 완료 n={prog["n"]}', flush=True)
        return
    ip = os.path.join(OUT, f'{ds}_img.npy')
    img = np.lib.format.open_memmap(ip, mode='r+' if os.path.exists(ip) else 'w+', dtype=np.uint8, shape=(maxn, 224, 224, 3))
    mpath = os.path.join(OUT, f'{ds}_meta_partial.json')
    meta = json.load(open(mpath)) if os.path.exists(mpath) else {'vid': [], 'frame': [], 'i': [], 'T': {}, 'fail': {}}
    n, start = prog['n'], prog['video_done']
    t0 = time.time()
    print(f'[{ds}] crop: 영상 {start}/{len(vids)}부터, 프로세스 {WORKERS}', flush=True)
    with Pool(WORKERS) as pool:
        for k, (vi, crops, ts, iis, T, st) in enumerate(pool.imap(_work, [(vi, vids[vi]) for vi in range(start, len(vids))]), 1):
            meta['T'][str(vi)] = T
            if crops is not None:
                img[n:n + len(crops)] = crops
                meta['vid'] += [vi] * len(crops)
                meta['frame'] += ts
                meta['i'] += iis
                n += len(crops)
            meta['fail'][str(vi)] = (32 if crops is None else 32 - len(crops)) if st == 'ok' else st
            if k % 50 == 0 or vi == len(vids) - 1:
                img.flush()
                json.dump(meta, open(mpath, 'w'))
                json.dump({'video_done': vi + 1, 'n': n, 'crop_done': False}, open(pfile, 'w'))
                el = time.time() - t0
                print(f'[{ds}] 영상 {vi + 1}/{len(vids)} 프레임 {n}, {el / 60:.1f}분, 남은 약 '
                      f'{el / k * (len(vids) - vi - 1) / 60:.1f}분', flush=True)
    img.flush()
    vid = np.array(meta['vid'], np.int32)
    np.savez(os.path.join(OUT, f'{ds}_meta.npz'), vid=vid, frame=np.array(meta['frame'], np.int32),
             i=np.array(meta['i'], np.int16), label=np.array([vids[v][1] for v in vid], np.int8),
             method=np.array([vids[v][2] for v in vid]), key=np.array([vids[v][3] for v in vid]),
             video_label=np.array([v[1] for v in vids], np.int8), video_method=np.array([v[2] for v in vids]),
             video_key=np.array([v[3] for v in vids]), video_T=np.array([meta['T'].get(str(i), -1) for i in range(len(vids))]),
             video_fail=np.array([meta['fail'].get(str(i), 'x') for i in range(len(vids))], dtype=str))
    json.dump({'video_done': len(vids), 'n': n, 'crop_done': True}, open(pfile, 'w'))
    print(f'[{ds}] crop 끝: 프레임 {n}/{maxn}, {(time.time() - t0) / 60:.1f}분', flush=True)


def base_phase(ds):
    import torch
    prog = json.load(open(os.path.join(OUT, f'{ds}_progress.json')))
    assert prog['crop_done'], f'{ds} crop 먼저'
    n = prog['n']
    img = np.load(os.path.join(OUT, f'{ds}_img.npy'), mmap_mode='r')

    def arr(name, dt, shp):
        p = os.path.join(OUT, f'{ds}_{name}.npy')
        return np.lib.format.open_memmap(p, mode='r+' if os.path.exists(p) else 'w+', dtype=dt, shape=shp)
    b3, lg, cl = arr('b3', np.float16, (n, 272, 11, 11)), arr('logit', np.float32, (n, 2)), arr('cls', np.float16, (n, 1024))
    done = prog.get('base_done', 0)
    if done >= n:
        print(f'[{ds}] base 이미 완료', flush=True)
        return
    model = base_phase.model
    buf = base_phase.buf
    t0 = time.time()
    for s in range(done, n, 32):
        e = min(n, s + 32)
        x = torch.from_numpy(np.ascontiguousarray(img[s:e]).astype(np.float32) / 255.).permute(0, 3, 1, 2).cuda()
        logit, b = C.baseline_forward(model, x, amp=True)
        assert torch.isfinite(logit).all() and torch.isfinite(b).all()
        b3[s:e], lg[s:e], cl[s:e] = b.cpu().numpy(), logit.cpu().numpy(), buf['cls'].half().cpu().numpy()
        if (e - done) % 3200 < 32 or e == n:
            b3.flush(); lg.flush(); cl.flush()
            prog['base_done'] = int(e)
            json.dump(prog, open(os.path.join(OUT, f'{ds}_progress.json'), 'w'))
            el = time.time() - t0
            print(f'[{ds}] base {e}/{n} {el / 60:.1f}분, 남은 약 {el / (e - done) * (n - e) / 60:.1f}분', flush=True)


def main():
    phase, dss = sys.argv[1], sys.argv[2:]
    if phase == 'crop':
        for ds in dss:
            crop_phase(ds)
    else:
        import torch
        torch.backends.cudnn.benchmark = False
        model = C.load_baseline('cuda')
        buf = {}
        model.vision_proj.register_forward_pre_hook(lambda m, inp: buf.__setitem__('cls', inp[0][:, 0].detach()))
        base_phase.model, base_phase.buf = model, buf
        for ds in dss:
            base_phase(ds)
    print('끝', flush=True)


if __name__ == '__main__':
    main()
