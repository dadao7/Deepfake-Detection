"""S3 게이트용: SBI 합성 샘플 격자 (진짜 | SBI | 블렌드 마스크 | 경계 B | 11×11 GT 확대 | |차이|×4).
출력: reports/s3_sbi_grid.png (로컬 확인용 — FF++ 이미지는 공개 게시 금지)"""
import json
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

if C.ORIG_CODE not in sys.path:
    sys.path.append(C.ORIG_CODE)
import sbi  # noqa: E402

LM_JSON = os.path.join(C.DATA, 'lm_json', 'train_81_FF++_processed_720.json')


def load_landmarks():
    d = json.load(open(LM_JSON))['data']
    out = {}
    for e in d:
        _, v, f = e['image_path'].split('/')
        out.setdefault(v, {})[int(f.split('.')[0])] = np.asarray(e['aligned_lms'], np.float32)
    return out


def main(n=12, seed=0):
    lms = load_landmarks()
    train_ids = sorted({x for p in C.load_pairs('train') for x in p} & set(lms))
    rng = np.random.default_rng(seed)
    rows = []
    with h5py.File(C.h5_path('original', 'train'), 'r') as f:
        for v in rng.choice(train_ids, n, replace=False):
            k = int(rng.choice(sorted(lms[v])))
            img = f[v][10 * k]
            real, fake, mb, info = sbi.self_blend(img, lms[v][k], rng)
            gt = sbi.boundary_gt(mb)
            b = 4 * mb * (1 - mb)
            gt_up = cv2.resize(gt, (224, 224), interpolation=cv2.INTER_NEAREST)
            diff = np.clip(np.abs(fake.astype(np.int16) - real.astype(np.int16)) * 4, 0, 255).astype(np.uint8)

            def g(a):
                return cv2.cvtColor((np.clip(a, 0, 1) * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            ov = real.copy()
            ov[gt_up >= 0.99] = (255, 0, 0)
            row = np.concatenate([real, fake, g(mb), g(b), g(gt_up), diff, ov], 1)
            cv2.putText(row, f'{v} k{k} r{info["ratio"]} {"68" if info["use68"] else "81"} '
                        f'{"src" if info["src_aug"] else "tgt"} {"F" if info.get("flip") else ""} '
                        f'{"q" + str(info["jpeg"]) if "jpeg" in info else ""}',
                        (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
            rows.append(row)
    grid = np.concatenate(rows, 0)
    os.makedirs(C.REPORTS, exist_ok=True)
    out = os.path.join(C.REPORTS, 's3_sbi_grid.png')
    small = cv2.resize(grid, (grid.shape[1] // 2, grid.shape[0] // 2), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out, cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
    print('저장', out, small.shape)


if __name__ == '__main__':
    main()
