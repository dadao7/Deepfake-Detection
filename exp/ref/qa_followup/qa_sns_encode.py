"""질문 방어 ⑫(축소판): SNS 업로드 모사 열화 (2026-10-21). 기본 miniconda Python + imageio-ffmpeg.
최신 생성 방식 데이터셋은 이 PC에 없고 받으려면 별도 신청이 필요해, 현업 질문의 다른 축인 'SNS에 올라간 영상'을 모사한다.
열화: 224 얼굴 crop → 112로 축소(INTER_AREA) → 영상별로 이어 H.264 crf 28 → 복원 → 224로 확대(INTER_LINEAR).
대상: FF++ test(첫 100프레임 연속), Celeb-DF, FaceShifter, DFD-test(평가에 쓰는 행만). 출력: cache/qa_sns/{set}_img.npy, {set}_rows.npy
"""
import json
import os
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

EXP = r'C:\m2f2\exp'
OUT = os.path.join(EXP, 'cache', 'qa_sns')
FF = imageio_ffmpeg.get_ffmpeg_exe()
os.makedirs(OUT, exist_ok=True)


def degrade(frames, tmp):
    small = np.stack([cv2.resize(f, (112, 112), interpolation=cv2.INTER_AREA) for f in frames])
    enc = [FF, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '112x112', '-r', '25', '-i', '-',
           '-c:v', 'libx264', '-crf', '28', '-pix_fmt', 'yuv420p', tmp]
    subprocess.run(enc, input=small.tobytes(), check=True)
    raw = subprocess.run([FF, '-loglevel', 'error', '-i', tmp, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                         capture_output=True, check=True).stdout
    dec = np.frombuffer(raw, np.uint8).reshape(-1, 112, 112, 3)
    assert len(dec) == len(frames)
    return np.stack([cv2.resize(f, (224, 224)) for f in dec])


def job(name, img, vid, rows):
    arr = np.lib.format.open_memmap(os.path.join(OUT, f'{name}_img.npy'), mode='w+', dtype=np.uint8, shape=(len(rows), 224, 224, 3))
    np.save(os.path.join(OUT, f'{name}_rows.npy'), rows)
    tmp = os.path.join(OUT, f'_tmp_{name}.mp4')
    v = vid[rows]
    for u in np.unique(v):
        pos = np.where(v == u)[0]
        arr[pos] = degrade(np.ascontiguousarray(img[rows[pos]]), tmp)
    arr.flush()
    os.remove(tmp)
    ref = np.asarray(img[rows[:500]], np.float64)
    mse = ((np.asarray(arr[:500], np.float64) - ref) ** 2).mean()
    print(f'{name}: {len(rows)}행, 원본 대비 PSNR {10 * np.log10(255 ** 2 / mse):.1f} dB', flush=True)


C = os.path.join(EXP, 'cache')
ix = np.load(os.path.join(C, 'index', 'test.npz'), allow_pickle=True)
job('fftest', np.load(os.path.join(C, 'frames', 'test_img.npy'), mmap_mode='r'), ix['vid'], np.arange(len(ix['vid'])))
ix = np.load(os.path.join(C, 'index', 'cdf.npz'), allow_pickle=True)
job('cdf', np.load(os.path.join(C, 'frames', 'cdf_img.npy'), mmap_mode='r'), ix['vid'], np.arange(len(ix['vid'])))
for ds in ('fsh', 'dfd'):
    m = np.load(os.path.join(C, 'newdata', f'{ds}_meta.npz'), allow_pickle=True)
    vid = m['vid']
    ok = np.bincount(vid, minlength=len(m['video_label'])) >= 8
    if ds == 'dfd':
        split = json.load(open(os.path.join(C, 'index', 'dfd_split.json'), encoding='utf-8'))
        ok &= np.isin(m['video_key'], split['test'])
    job('fsh' if ds == 'fsh' else 'dfdtest', np.load(os.path.join(C, 'newdata', f'{ds}_img.npy'), mmap_mode='r'), vid,
        np.where(ok[vid])[0])
