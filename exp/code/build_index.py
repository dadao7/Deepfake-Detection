"""S1-1: 분할별 프레임 인덱스·nested 부분집합·1:1 균형 스케줄 생성 (PROTOCOL §3)
출력: cache/index/{train,val,test,cdf}.npz, schedule_d{25,50,100}.npz, train_pairs_perm.json, index_summary.json"""
import hashlib
import json
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

os.makedirs(C.IDX_DIR, exist_ok=True)


def frame_counts(cat, split, keys):
    with h5py.File(C.h5_path(cat, split), 'r') as f:
        miss = [k for k in keys if k not in f]
        assert not miss, f'{cat} {split} 없는 키 {miss[:5]}'
        return {k: f[k].shape[0] for k in keys}


def build_ffpp(split, pairs, k_frames, pair_rank=None):
    rows = {'cat': [], 'video': [], 'label': [], 'frame': [], 'vid': [], 'pair_rank': [], 'vrank': [], 'chunk': []}
    vid_no = 0
    for cat in C.CATS:
        vids = C.videos_of_pairs(pairs, cat)
        n_of = frame_counts(cat, split, vids)
        for r, v in enumerate(vids):
            n = n_of[v]
            fr = list(range(min(100, n))) if k_frames is None else C.frame_idx(n, k_frames)
            if k_frames is None:
                assert n >= 100, f'{cat}/{v} 프레임 {n} < 100'
            pr = pair_rank[v.split('_')[0]] if pair_rank is not None else -1
            if cat != 'original' and pair_rank is not None:
                # 영상별 고정 순열 → chunk = (p + r) mod 4  (PROTOCOL §3.4)
                seed = int(hashlib.sha256(f'{cat}/{v}'.encode()).hexdigest()[:8], 16)
                perm = np.random.default_rng(seed).permutation(len(fr))
                pos = np.argsort(perm)
                chunks = [(int(pos[j]) + r) % 4 for j in range(len(fr))]
            else:
                chunks = [-1] * len(fr)
            for j, fi in enumerate(fr):
                rows['cat'].append(cat)
                rows['video'].append(v)
                rows['label'].append(0 if cat == 'original' else 1)
                rows['frame'].append(fi)
                rows['vid'].append(vid_no)
                rows['pair_rank'].append(pr)
                rows['vrank'].append(r)
                rows['chunk'].append(chunks[j])
            vid_no += 1
    out = {'cat': np.array(rows['cat']), 'video': np.array(rows['video']),
           'label': np.array(rows['label'], np.int8), 'frame': np.array(rows['frame'], np.int16),
           'vid': np.array(rows['vid'], np.int32), 'pair_rank': np.array(rows['pair_rank'], np.int16),
           'vrank': np.array(rows['vrank'], np.int32), 'chunk': np.array(rows['chunk'], np.int8)}
    np.savez(os.path.join(C.IDX_DIR, f'{split}.npz'), **out)
    return out


def build_cdf():
    lines = [ln.split() for ln in open(os.path.join(C.CDF_DIR, 'List_of_testing_videos.txt')) if ln.strip()]
    video, label, ffile, vid = [], [], [], []
    for vi, (lab, rel) in enumerate(lines):
        sub, name = rel.split('/')
        d = os.path.join(C.CDF_DIR, sub, 'frames', os.path.splitext(name)[0])
        fs = sorted([f for f in os.listdir(d) if f.lower().endswith('.png')], key=lambda s: int(re.sub(r'\D', '', s)))
        assert fs, d
        for f in fs:
            video.append(rel)
            label.append(1 if lab == '0' else 0)   # 목록 라벨 1=진짜 → 우리 라벨 가짜=1
            ffile.append(f'{sub}/frames/{os.path.splitext(name)[0]}/{f}')
            vid.append(vi)
    out = {'video': np.array(video), 'label': np.array(label, np.int8), 'relpath': np.array(ffile),
           'vid': np.array(vid, np.int32)}
    np.savez(os.path.join(C.IDX_DIR, 'cdf.npz'), **out)
    return out


def main():
    train_pairs = C.load_pairs('train')
    perm = np.random.default_rng(C.PAIR_PERM_SEED).permutation(len(train_pairs))
    pairs_perm = [train_pairs[i] for i in perm]
    n_pairs = {d: len(train_pairs) * d // 100 for d in C.DATA_PCTS}
    json.dump({'seed': C.PAIR_PERM_SEED, 'pairs_perm': pairs_perm, 'n_pairs': n_pairs},
              open(os.path.join(C.IDX_DIR, 'train_pairs_perm.json'), 'w'), indent=0)
    pair_rank = {}
    for r, p in enumerate(pairs_perm):
        pair_rank[p[0]] = r
        pair_rank[p[1]] = r

    tr = build_ffpp('train', pairs_perm, C.N_TRAIN_FR, pair_rank)
    va = build_ffpp('val', C.load_pairs('val'), C.N_VAL_FR)
    te = build_ffpp('test', C.load_pairs('test'), None)
    cdf = build_cdf()

    # 스케줄 + 균형 검사
    summ = {'train_rows': len(tr['label']), 'val_rows': len(va['label']), 'test_rows': len(te['label']),
            'cdf_rows': len(cdf['label']), 'cdf_videos': int(cdf['vid'].max()) + 1,
            'cdf_real_videos': int(len(set(cdf['vid'][cdf['label'] == 0]))),
            'cdf_fake_videos': int(len(set(cdf['vid'][cdf['label'] == 1]))), 'schedule': {}}
    for d in C.DATA_PCTS:
        pool = tr['pair_rank'] < n_pairs[d]
        real = np.where(pool & (tr['label'] == 0))[0]
        chunks = [np.where(pool & (tr['label'] == 1) & (tr['chunk'] == c))[0] for c in range(4)]
        R = len(real)
        assert R == n_pairs[d] * 20, (d, R)
        for c in range(4):
            assert len(chunks[c]) == R, (d, c, len(chunks[c]), R)
            for m in C.METHODS:
                cnt = int((tr['cat'][chunks[c]] == m).sum())
                assert cnt == R // 4, (d, c, m, cnt)
            # 영상마다 epoch당 2~3장
            vv, cc = np.unique(tr['vid'][chunks[c]], return_counts=True)
            assert cc.min() >= 2 and cc.max() <= 3 and len(vv) == n_pairs[d] * 8, (d, c, cc.min(), cc.max(), len(vv))
        np.savez(os.path.join(C.IDX_DIR, f'schedule_d{d}.npz'), real=real,
                 **{f'fake{c}': chunks[c] for c in range(4)})
        summ['schedule'][d] = {'pairs': n_pairs[d], 'real_frames': R, 'fake_per_chunk': R,
                               'steps_per_epoch': 2 * R // 16}
    # nested 검사: 작은 풀의 프레임은 큰 풀의 같은 chunk에 들어 있다
    s = {d: np.load(os.path.join(C.IDX_DIR, f'schedule_d{d}.npz')) for d in C.DATA_PCTS}
    for a, b in ((25, 50), (50, 100)):
        assert set(s[a]['real']) <= set(s[b]['real'])
        for c in range(4):
            assert set(s[a][f'fake{c}']) <= set(s[b][f'fake{c}'])
    for split, ix in (('train', tr), ('val', va), ('test', te)):
        summ[f'{split}_videos'] = {cat: int(len(set(ix['vid'][ix['cat'] == cat]))) for cat in C.CATS}
    summ['train_frames_per_video_n'] = {int(n): int(c) for n, c in zip(*np.unique(
        [tr['frame'][tr['vid'] == v].max() for v in range(0, tr['vid'].max() + 1, 1)], return_counts=True))}
    summ['index_sha'] = C.index_sha()
    json.dump(summ, open(os.path.join(C.IDX_DIR, 'index_summary.json'), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False)
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
