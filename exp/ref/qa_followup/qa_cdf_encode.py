"""질문 방어 ①: Celeb-DF 재압축 (2026-10-21). 기본 miniconda Python + imageio-ffmpeg(ffmpeg 7.1).
FF++ c23/c40과 같은 H.264(libx264) 고정 품질 crf 23·40으로, Celeb-DF 테스트 영상의 평가 프레임(256×256 PNG)을
영상별로 묶어 압축 → 복원 → 기존 규칙대로 224×224(INTER_LINEAR) 축소. 순서·행은 cache/index/cdf.npz와 같다.
주의: 원본 영상 전체가 아니라 평가용으로 뽑힌 프레임(영상당 ≤32장)만 이어 붙여 압축하므로 프레임 간 예측이 실제 영상보다 약하다.
출력: cache/qa_cdf/cdf_crf{23,40}_img.npy
"""
import os
import subprocess

import cv2
import imageio_ffmpeg
import numpy as np

EXP = r'C:\m2f2\exp'
CDF = r'C:\m2f2\data\Celeb-DF-v2'
OUT = os.path.join(EXP, 'cache', 'qa_cdf')
FF = imageio_ffmpeg.get_ffmpeg_exe()
os.makedirs(OUT, exist_ok=True)
ix = np.load(os.path.join(EXP, 'cache', 'index', 'cdf.npz'), allow_pickle=True)
vid, rel = ix['vid'], ix['relpath']


def roundtrip(frames, crf, tmp):
    h, w = frames.shape[1:3]
    enc = [FF, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '25', '-i', '-',
           '-c:v', 'libx264', '-crf', str(crf), '-pix_fmt', 'yuv420p', tmp]
    subprocess.run(enc, input=frames.tobytes(), check=True)
    dec = [FF, '-loglevel', 'error', '-i', tmp, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-']
    raw = subprocess.run(dec, capture_output=True, check=True).stdout
    out = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    assert len(out) == len(frames), (len(out), len(frames))
    return out


for crf in (23, 40):
    path = os.path.join(OUT, f'cdf_crf{crf}_img.npy')
    arr = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8, shape=(len(vid), 224, 224, 3))
    tmp = os.path.join(OUT, f'_tmp{crf}.mp4')
    for v in np.unique(vid):
        rows = np.where(vid == v)[0]
        fr = np.stack([cv2.cvtColor(cv2.imread(os.path.join(CDF, rel[r])), cv2.COLOR_BGR2RGB) for r in rows])
        rt = roundtrip(fr, crf, tmp)
        arr[rows] = np.stack([cv2.resize(f, (224, 224)) for f in rt])
    arr.flush()
    os.remove(tmp)
    print(f'crf {crf} 완료: {path}', flush=True)

# 검증: 압축 없이 같은 읽기·축소를 하면 기존 캐시(cdf_img.npy)와 같아야 한다(앞 64장)
ref = np.load(os.path.join(EXP, 'cache', 'frames', 'cdf_img.npy'), mmap_mode='r')
same = all(np.array_equal(cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(CDF, rel[r])), cv2.COLOR_BGR2RGB), (224, 224)), ref[r])
           for r in range(64))
print('원본 읽기 재현:', same, flush=True)
for crf in (23, 40):
    a = np.load(os.path.join(OUT, f'cdf_crf{crf}_img.npy'), mmap_mode='r')
    d = np.abs(a[:2000].astype(np.int16) - ref[:2000].astype(np.int16))
    mse = float((d.astype(np.float64) ** 2).mean())
    print(f'crf {crf}: 원본 대비 PSNR {10 * np.log10(255 ** 2 / mse):.1f} dB (앞 2000장)', flush=True)
