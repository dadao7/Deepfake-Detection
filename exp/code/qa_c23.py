"""질문 방어 ⑬(축소판): FF++ c23(고화질) test 평가 (2026-10-21). 학습 없음. 기존 S10 파이프라인(build_newdata_cache)을 고치지 않고,
영상 목록의 c40 경로만 c23으로 바꿔 끼워 같은 얼굴 정렬·프레임 규칙(ffv: i∈{0,3,…,93}, t=floor(i·T/N))으로 crop·캐시를 만든다.
c40 짝은 기존 cache/newdata/ffv_*(같은 프레임 위치) → 같은 영상·같은 프레임에서 압축만 다른 짝 비교.
  python code/qa_c23.py crop   (CPU)   → cache/qa_c23/ffv_img.npy, ffv_meta.npz
  python code/qa_c23.py base   (GPU)   → cache/qa_c23/ffv_{b3,logit,cls}.npy
  python code/qa_c23.py eval   (GPU)   → cache/qa_c23/scores.npz (c23·c40 모두)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import build_newdata_cache as BN  # noqa: E402
import common as C  # noqa: E402

QD = os.path.join(C.CACHE, 'qa_c23')
_orig_list = BN.video_list


def video_list_c23(ds):
    assert ds == 'ffv'
    sep = os.sep
    return [(p.replace(f'{sep}c40{sep}', f'{sep}c23{sep}'), *rest) for p, *rest in _orig_list('ffv')]


BN.video_list = video_list_c23
BN.OUT = QD


def evaluate():
    import torch
    import t5_common as T5
    import t5_merge as M

    class P0Wrap(torch.nn.Module):
        def forward(self, x, b3, lg):
            return {'pred': lg}
    ms, names = [P0Wrap().cuda()], ['P0']
    for seed in (0, 1, 2):
        s0 = M.init_state(seed)
        p3l = M.expert_state('P3', 'last', seed, s0)
        mg = M.combine(s0, M.expert_state('P3', 'best', seed, s0), M.expert_state('U3s', 'best', seed, s0), 0.5, 0.75)
        ms += [M.model_from(p3l, seed), M.model_from(mg, seed)]
        names += [f'P3|{seed}', f'final|{seed}']
    res = {'names': np.array(names)}
    for tag, d in (('c23', QD), ('c40', os.path.join(C.CACHE, 'newdata'))):
        meta = dict(np.load(os.path.join(d, 'ffv_meta.npz'), allow_pickle=True))
        n = len(meta['vid'])
        A = {k: np.load(os.path.join(d, f'ffv_{k}.npy'), mmap_mode='r') for k in ('img', 'b3', 'logit', 'cls')}

        def g(r, A=A):
            r = np.asarray(r)
            t = lambda a: torch.from_numpy(np.ascontiguousarray(a[r[0]:r[-1] + 1])).cuda()  # noqa: E731
            return t(A['img']).permute(0, 3, 1, 2).float().div_(255.), t(A['b3']), t(A['logit']).float(), t(A['cls'])
        P = T5._probs(ms, n, g)
        vid, vl, keys = meta['vid'], meta['video_label'], meta['video_key']
        u = np.unique(vid)
        res[f'{tag}|key'] = np.array([f'{meta["video_method"][v]}|{keys[v]}' for v in u])
        res[f'{tag}|label'] = vl[u]
        res[f'{tag}|score'] = np.stack([[C.trimmed_mean(p[vid == v], int(0.1 * (vid == v).sum())) for v in u] for p in P])
        print(f'[{tag}] 평가 영상 {len(u)}', flush=True)
    np.savez(os.path.join(QD, 'scores.npz'), **res)
    print('C23_EVAL_OK', flush=True)


if __name__ == '__main__':
    os.makedirs(QD, exist_ok=True)
    if sys.argv[1] == 'crop':
        BN.crop_phase('ffv')
    elif sys.argv[1] == 'base':
        sys.argv = ['build_newdata_cache.py', 'base', 'ffv']
        BN.main()
    else:
        evaluate()
