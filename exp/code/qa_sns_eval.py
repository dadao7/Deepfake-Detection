"""질문 방어 ⑫(축소판): SNS 모사 열화 평가 (2026-10-21). 학습 없음, 기존 코드는 함수만 불러 쓴다. m2f2 환경, GPU.
열화 프레임(ref/qa_followup/qa_sns_encode.py)으로 논문 모델 출력 캐시(b3·logit·CLS)를 새로 만들고, 원본 캐시와 같은 행에서 모델을 평가한다.
영상 점수 = 프레임 확률에서 위·아래 floor(0.1n)개를 뺀 평균(모든 세트 공통 규칙과 같음).
모델: P0, P3 last, 최종 병합 — seed 0~2. 출력: cache/qa_sns/{set}_{b3,logit,cls}.npy, cache/qa_sns/scores.npz
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common as C  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_merge as M  # noqa: E402

QD = os.path.join(C.CACHE, 'qa_sns')
ND = os.path.join(C.CACHE, 'newdata')
SETS = ('fftest', 'cdf', 'fsh', 'dfdtest')


def src(s):
    """원본 이미지·캐시·영상 id (행 순서 = {set}_rows.npy)"""
    rows = np.load(os.path.join(QD, f'{s}_rows.npy'))
    if s in ('fftest', 'cdf'):
        sp = 'test' if s == 'fftest' else 'cdf'
        arr = {k: np.load(os.path.join(C.BASE_DIR, f'{sp}_{k}.npy'), mmap_mode='r') for k in ('b3', 'logit', 'cls')}
        img = np.load(os.path.join(C.FRAME_DIR, f'{sp}_img.npy'), mmap_mode='r')
        vid = C.load_index(sp)['vid']
        lab_v = T5.labels()[s]
        vl = lab_v                                         # 영상 순서 = vid 순서
    else:
        ds = 'fsh' if s == 'fsh' else 'dfd'
        arr = {k: np.load(os.path.join(ND, f'{ds}_{k}.npy'), mmap_mode='r') for k in ('b3', 'logit', 'cls')}
        img = np.load(os.path.join(ND, f'{ds}_img.npy'), mmap_mode='r')
        meta = np.load(os.path.join(ND, f'{ds}_meta.npz'), allow_pickle=True)
        vid, vl = meta['vid'], meta['video_label']
    return rows, img, arr, vid, vl


def build_cache(s):
    if os.path.exists(os.path.join(QD, f'{s}_cls.npy')):
        return
    model = C.load_baseline('cuda', verbose=False)
    buf = {}
    model.vision_proj.register_forward_pre_hook(lambda m, inp: buf.__setitem__('cls', inp[0][:, 0].detach()))
    img = np.load(os.path.join(QD, f'{s}_img.npy'), mmap_mode='r')
    N = len(img)
    b3 = np.lib.format.open_memmap(os.path.join(QD, f'{s}_b3.npy'), mode='w+', dtype=np.float16, shape=(N, 272, 11, 11))
    lg = np.lib.format.open_memmap(os.path.join(QD, f'{s}_logit.npy'), mode='w+', dtype=np.float32, shape=(N, 2))
    cl = np.lib.format.open_memmap(os.path.join(QD, f'{s}_cls.npy'), mode='w+', dtype=np.float16, shape=(N, 1024))
    for a in range(0, N, 32):
        x = torch.from_numpy(np.ascontiguousarray(img[a:a + 32]).astype(np.float32) / 255.).permute(0, 3, 1, 2).cuda()
        logit, b = C.baseline_forward(model, x, amp=True)
        b3[a:a + 32], lg[a:a + 32], cl[a:a + 32] = b.cpu().numpy(), logit.cpu().numpy(), buf['cls'].half().cpu().numpy()
    for z in (b3, lg, cl):
        z.flush()
    del model
    torch.cuda.empty_cache()
    print(f'[{s}] 열화 캐시 완료', flush=True)


class P0Wrap(torch.nn.Module):
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


def getter(img, b3, lg, cl, rows=None):
    def g(r):
        r = np.asarray(r)
        idx = r if rows is None else rows[r]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[idx])).cuda()  # noqa: E731
        return t(img).permute(0, 3, 1, 2).float().div_(255.), t(b3), t(lg).float(), t(cl)
    return g


def video(p, v):
    u = np.unique(v)
    out = np.empty(len(u))
    for j, w in enumerate(u):
        q = p[v == w]
        out[j] = C.trimmed_mean(q, int(0.1 * len(q)))
    return u, out


def main():
    for s in SETS:
        build_cache(s)
    ms, names = models()
    res = {'names': np.array(names)}
    for s in SETS:
        rows, img, arr, vid, vl = src(s)
        n = len(rows)
        Po = T5._probs(ms, n, getter(img, arr['b3'], arr['logit'], arr['cls'], rows))
        dimg = np.load(os.path.join(QD, f'{s}_img.npy'), mmap_mode='r')
        dc = {k: np.load(os.path.join(QD, f'{s}_{k}.npy'), mmap_mode='r') for k in ('b3', 'logit', 'cls')}
        Pd = T5._probs(ms, n, getter(dimg, dc['b3'], dc['logit'], dc['cls']))
        v = vid[rows]
        u, _ = video(Po[0], v)
        res[f'{s}|label'] = np.asarray(vl)[u]
        res[f'{s}|orig'] = np.stack([video(p, v)[1] for p in Po])
        res[f'{s}|sns'] = np.stack([video(p, v)[1] for p in Pd])
        np.savez(os.path.join(QD, 'scores.npz'), **res)
        print(f'[{s}] 평가 완료 (영상 {len(u)})', flush=True)
    print('SNS_EVAL_OK', flush=True)


if __name__ == '__main__':
    main()
