"""질문 방어 ①: 재압축 Celeb-DF 평가 (2026-10-21). 학습 없음, 기존 코드는 함수만 불러 쓴다.
가설: "SBI 학습이 DFD·FaceShifter(c40)엔 이득인데 Celeb-DF엔 손해"인 이유가 화질·압축 차이라면,
Celeb-DF를 c40 수준으로 압축하면 SBI 학습 모델의 손해가 줄거나 이득으로 바뀌어야 한다.
변형: orig(기존 캐시) · crf23 · crf40 (ref/qa_followup/qa_cdf_encode.py가 만든 프레임)
모델: P0, P3 last, U0s best(분기 없이 SBI 미세조정), U3s best(분기 + SBI), 최종 병합 — seed 0~2
출력: cache/qa_cdf/{crf23,crf40}_{b3,logit,cls}.npy, cache/qa_cdf/scores.npz
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common as C  # noqa: E402
import run_train as R  # noqa: E402
import run_train_unfreeze as U  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_merge as M  # noqa: E402

QD = os.path.join(C.CACHE, 'qa_cdf')
ix = C.load_index('cdf')
N = len(ix['label'])


def build_cache(v):
    if os.path.exists(os.path.join(QD, f'{v}_cls.npy')):
        return
    model = C.load_baseline('cuda', verbose=False)
    buf = {}
    model.vision_proj.register_forward_pre_hook(lambda m, inp: buf.__setitem__('cls', inp[0][:, 0].detach()))
    img = np.load(os.path.join(QD, f'cdf_{v}_img.npy'), mmap_mode='r')
    b3 = np.lib.format.open_memmap(os.path.join(QD, f'{v}_b3.npy'), mode='w+', dtype=np.float16, shape=(N, 272, 11, 11))
    lg = np.lib.format.open_memmap(os.path.join(QD, f'{v}_logit.npy'), mode='w+', dtype=np.float32, shape=(N, 2))
    cl = np.lib.format.open_memmap(os.path.join(QD, f'{v}_cls.npy'), mode='w+', dtype=np.float16, shape=(N, 1024))
    for s in range(0, N, 32):
        x = torch.from_numpy(np.ascontiguousarray(img[s:s + 32]).astype(np.float32) / 255.).permute(0, 3, 1, 2).cuda()
        logit, b = C.baseline_forward(model, x, amp=True)
        b3[s:s + 32], lg[s:s + 32], cl[s:s + 32] = b.cpu().numpy(), logit.cpu().numpy(), buf['cls'].half().cpu().numpy()
    for a in (b3, lg, cl):
        a.flush()
    del model
    torch.cuda.empty_cache()
    print(f'[{v}] 캐시 완료', flush=True)


def sanity():
    """같은 코드로 원본 프레임 앞 256장을 다시 계산 → 기존 cdf 캐시와 비교"""
    model = C.load_baseline('cuda', verbose=False)
    img = np.load(os.path.join(C.FRAME_DIR, 'cdf_img.npy'), mmap_mode='r')
    ref = np.load(os.path.join(C.BASE_DIR, 'cdf_logit.npy'), mmap_mode='r')
    d = 0.0
    for s in range(0, 256, 32):
        x = torch.from_numpy(np.ascontiguousarray(img[s:s + 32]).astype(np.float32) / 255.).permute(0, 3, 1, 2).cuda()
        logit, _ = C.baseline_forward(model, x, amp=True)
        d = max(d, float(np.abs(logit.cpu().numpy() - ref[s:s + 32]).max()))
    print(f'sanity: 원본 프레임 logit 최대 차이 {d:.2e}', flush=True)
    assert d < 1e-2
    del model
    torch.cuda.empty_cache()


class P0Wrap(torch.nn.Module):
    def forward(self, x, b3, lg):
        return {'pred': lg}


def models():
    ms, names = [P0Wrap().cuda()], ['P0']
    for seed in (0, 1, 2):
        s0 = M.init_state(seed)
        p3l = M.expert_state('P3', 'last', seed, s0)
        u3s = M.expert_state('U3s', 'best', seed, s0)
        mg = M.combine(s0, M.expert_state('P3', 'best', seed, s0), u3s, 0.5, 0.75)
        u0 = U.UModel('U0', seed)
        u0.load_state_dict(torch.load(os.path.join(C.RUNS, f'U0s_s{seed}', 'ckpt_best.pt'), map_location='cpu', weights_only=False)['model'])
        ms += [M.model_from(p3l, seed), u0.cuda().eval(), M.model_from(u3s, seed), M.model_from(mg, seed)]
        names += [f'P3|{seed}', f'U0s|{seed}', f'U3s|{seed}', f'final|{seed}']
    return ms, names


def arrays(v):
    if v == 'orig':
        img = np.load(os.path.join(C.FRAME_DIR, 'cdf_img.npy'), mmap_mode='r')
        b3, lg, cl = [np.load(os.path.join(C.BASE_DIR, f'cdf_{k}.npy'), mmap_mode='r') for k in ('b3', 'logit', 'cls')]
    else:
        img = np.load(os.path.join(QD, f'cdf_{v}_img.npy'), mmap_mode='r')
        b3, lg, cl = [np.load(os.path.join(QD, f'{v}_{k}.npy'), mmap_mode='r') for k in ('b3', 'logit', 'cls')]

    def g(r):
        r = np.asarray(r)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a[r[0]:r[-1] + 1])).cuda()  # noqa: E731
        x = t(img).permute(0, 3, 1, 2).float().div_(255.)
        return x, t(b3), t(lg).float(), t(cl)
    return g


def main():
    sanity()
    for v in ('crf23', 'crf40'):
        build_cache(v)
    ms, names = models()
    out = {'names': np.array(names)}
    for v in ('orig', 'crf23', 'crf40'):
        P = T5._probs(ms, N, arrays(v))
        out[v] = np.stack([C.video_scores(p, ix['vid'], 'frac') for p in P])
        print(f'[{v}] 평가 완료', flush=True)
    np.savez(os.path.join(QD, 'scores.npz'), **out)
    print('CDF_EVAL_OK', flush=True)


if __name__ == '__main__':
    main()
