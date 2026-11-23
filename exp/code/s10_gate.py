"""S10 재현 검증 게이트 (PROTOCOL §16): FF++ test를 우리 정렬 파이프라인으로 만든 crop(ffv) vs h5 crop, 같은 영상·같은 프레임.
① 프레임별 NCC(가운데 192×192, 회색) ② P0 영상 AUC(우리 crop vs h5 crop, 같은 영상·같은 32프레임, 양쪽 10% 제외 평균)
통과 기준: |ΔAUC| ≤ 1%p. 출력: reports/s10_gate.json"""
import json
import os
import sys

import cv2
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ND = os.path.join(C.CACHE, 'newdata')


def ncc(a, b, m=16):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)[m:-m, m:-m].astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)[m:-m, m:-m].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum() + 1e-9))


def main():
    z = np.load(os.path.join(ND, 'ffv_meta.npz'))
    meta = {k: z[k] for k in z.files}
    img = np.load(os.path.join(ND, 'ffv_img.npy'), mmap_mode='r')
    lg = np.load(os.path.join(ND, 'ffv_logit.npy'))
    te = C.load_index('test')
    row_of = {(c, v, int(f)): r for r, (c, v, f) in enumerate(zip(te['cat'], te['video'], te['frame']))}
    tl = np.load(os.path.join(C.BASE_DIR, 'test_logit.npy'))
    n = len(meta['vid'])
    nccs, p_ours, p_h5, vids = [], [], [], []
    files = {c: h5py.File(C.h5_path(c, 'test'), 'r') for c in C.CATS}
    for r in range(n):
        c, k, i = str(meta['method'][r]), str(meta['key'][r]), int(meta['i'][r])
        cr = row_of.get((c, k, i))
        if cr is None:
            continue
        nccs.append(ncc(np.asarray(img[r]), files[c][k][i]))
        p_ours.append(1 / (1 + np.exp(lg[r, 0] - lg[r, 1])))
        p_h5.append(1 / (1 + np.exp(tl[cr, 0] - tl[cr, 1])))
        vids.append(int(meta['vid'][r]))
    for f in files.values():
        f.close()
    nccs, p_ours, p_h5, vids = map(np.asarray, (nccs, p_ours, p_h5, vids))
    vl = meta['video_label']
    out = {'frames_compared': int(len(nccs)), 'frames_total': int(n),
           'ncc_median': float(np.median(nccs)), 'ncc_mean': float(nccs.mean()),
           'ncc_p10': float(np.quantile(nccs, 0.1)), 'ncc_ge_095': float((nccs >= 0.95).mean()),
           'frame_prob_mean_absdiff': float(np.abs(p_ours - p_h5).mean())}
    res = {}
    for name, p in (('ours', p_ours), ('h5', p_h5)):
        V = len(vl)
        sc, ok = np.full(V, np.nan), np.zeros(V, bool)
        for v in range(V):
            q = p[vids == v]
            if len(q) >= 8:
                sc[v], ok[v] = C.trimmed_mean(q, int(0.1 * len(q))), True
        res[name] = (sc, ok)
    ok = res['ours'][1] & res['h5'][1]
    for name in ('ours', 'h5'):
        m = C.summarize(res[name][0][ok], vl[ok])
        out[f'P0_{name}'] = m
    out['n_videos_both'] = int(ok.sum())
    out['delta_auc_ours_minus_h5'] = out['P0_ours']['auc'] - out['P0_h5']['auc']
    out['pass'] = bool(abs(out['delta_auc_ours_minus_h5']) <= 1.0)
    json.dump(out, open(os.path.join(C.REPORTS, 's10_gate.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
