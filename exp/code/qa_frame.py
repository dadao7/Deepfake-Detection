"""질문 방어 ⑭: 단일 프레임(이미지 1장) 입력 성능 (2026-10-21). 학습 없음, 기존 코드는 함수만 불러 쓴다.
평가 단위를 영상 → 프레임으로 바꿔, 사진 한 장만 들어오는 조건의 성능을 잰다.
모델: P0(캐시된 baseline logit), P3 last(seed 0~2), 최종 병합(a0.50b0.75, seed 0~2). 보고 세트 4개.
  python code/qa_frame.py   (m2f2 환경, GPU) → cache/qa_frame_probs.npz : 프레임 확률·프레임 라벨·영상 id
분석은 ref/qa_followup/qa_frame_stats.py (기본 miniconda Python).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common as C  # noqa: E402
import t34_common as T  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_merge as M  # noqa: E402

OUT = os.path.join(C.CACHE, 'qa_frame_probs.npz')


class P0Wrap(torch.nn.Module):
    """논문 모델 = 캐시된 baseline logit 그대로"""
    def forward(self, x, b3, lg):
        return {'pred': lg}


def models():
    ms, names = [P0Wrap().cuda()], ['P0']
    for seed in (0, 1, 2):
        s0 = M.init_state(seed)
        p3l = M.expert_state('P3', 'last', seed, s0)
        mg = M.combine(s0, M.expert_state('P3', 'best', seed, s0), M.expert_state('U3s', 'best', seed, s0), 0.5, 0.75)
        ms += [M.model_from(p3l, seed), M.model_from(mg, seed)]
        names += [f'P3|{seed}', f'final|{seed}']
    return ms, names


def main():
    ms, names = models()
    lab = T5.labels()
    out = {'names': np.array(names)}
    for s, sp in (('fftest', 'test'), ('cdf', 'cdf')):
        D, g = T5._ff(sp)
        P = T5._probs(ms, D.N, g)
        vid = np.asarray(D.ix['vid'])
        out[f'{s}|p'], out[f'{s}|y'], out[f'{s}|vid'] = P, lab[s][vid], vid
        print(s, P.shape, flush=True)
    D, g = T5._nd('fsh')
    vid, vl = D.meta['vid'], D.meta['video_label']
    ok = np.bincount(vid, minlength=len(vl)) >= T5.EV.MIN_FRAMES
    rows = np.where(ok[vid])[0]
    out['fsh|p'], out['fsh|y'], out['fsh|vid'] = T5._probs(ms, D.N, g, rows), np.asarray(vl)[vid[rows]], vid[rows]
    print('fsh', len(rows), flush=True)
    split = T.dfd_split()
    D, g = T5._nd('dfd')
    vid, vl, keys = D.meta['vid'], D.meta['video_label'], D.meta['video_key']
    ok = (np.bincount(vid, minlength=len(vl)) >= T5.EV.MIN_FRAMES) & np.isin(keys, split['test'])
    rows = np.where(ok[vid])[0]
    out['dfdtest|p'], out['dfdtest|y'], out['dfdtest|vid'] = T5._probs(ms, D.N, g, rows), np.asarray(vl)[vid[rows]], vid[rows]
    print('dfdtest', len(rows), flush=True)
    np.savez(OUT, **out)
    print('FRAME_OK', flush=True)


if __name__ == '__main__':
    main()
