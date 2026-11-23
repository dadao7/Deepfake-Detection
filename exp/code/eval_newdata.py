"""S10: 새 데이터(ffv·fsh·dfd)에서 모든 체크포인트 평가 (PROTOCOL §16). 데이터를 한 번 읽으며 배치마다 모든 모델을 돌린다.
대상: P0(캐시 logit), S4 9설정×3seed×best/last, S8 P2·P3·P4a(SBI)×3×2, S9 U0f·U3f·U0s·U3s×3×2.
영상 점수: 유효 프레임에서 양쪽 floor(0.1n) 제외 평균. 유효 프레임 8장 미만 영상은 제외(수 보고).
출력: reports/s10_eval.csv (모델×체크포인트×데이터셋 1줄), reports/s10_frame_probs_{ds}.npz
  python code/eval_newdata.py ffv fsh dfd"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import expmodel as E  # noqa: E402
import run_train as R  # noqa: E402

ND = os.path.join(C.CACHE, 'newdata')
MIN_FRAMES = 8
S4_SETS = ['P1', 'P1p', 'P2a', 'P2b', 'P2', 'P3a', 'P3b', 'P3', 'P4a']


class NewData:
    def __init__(self, ds):
        z = np.load(os.path.join(ND, f'{ds}_meta.npz'))
        self.meta = {k: z[k] for k in z.files}
        self.N = len(self.meta['vid'])
        self.img = np.load(os.path.join(ND, f'{ds}_img.npy'), mmap_mode='r')
        self.b3 = np.load(os.path.join(ND, f'{ds}_b3.npy'), mmap_mode='r')
        self.lg = np.load(os.path.join(ND, f'{ds}_logit.npy'), mmap_mode='r')
        self.cls = np.load(os.path.join(ND, f'{ds}_cls.npy'), mmap_mode='r')

    def batch(self, s, e):
        x = torch.from_numpy(np.ascontiguousarray(self.img[s:e])).cuda().permute(0, 3, 1, 2).float().div_(255.)
        return (x, torch.from_numpy(np.ascontiguousarray(self.b3[s:e])).cuda(),
                torch.from_numpy(np.ascontiguousarray(self.lg[s:e])).cuda(),
                torch.from_numpy(np.ascontiguousarray(self.cls[s:e])).cuda())


def model_specs():
    specs = []
    for st in S4_SETS:
        for s in (0, 1, 2):
            for ck in ('best', 'last'):
                specs.append(('S4', st, s, ck, os.path.join(C.RUNS, f'{st}_d100_s{s}', f'ckpt_{ck}.pt')))
    for st in ('P2', 'P3', 'P4a'):
        for s in (0, 1, 2):
            for ck in ('best', 'last'):
                specs.append(('S8', f'{st}_sbi', s, ck, os.path.join(C.RUNS, f'{st}_sbi_s{s}', f'ckpt_{ck}.pt')))
    for st in ('U0f', 'U3f', 'U0s', 'U3s'):
        for s in (0, 1, 2):
            for ck in ('best', 'last'):
                specs.append(('S9', st, s, ck, os.path.join(C.RUNS, f'{st}_s{s}', f'ckpt_{ck}.pt')))
    return [sp for sp in specs if os.path.exists(sp[4])]


def build(sp):
    stage, st, seed, ck, path = sp
    sd = torch.load(path, map_location='cpu', weights_only=False)['model']
    if stage == 'S9':
        import run_train_unfreeze as U
        m = U.UModel(st[:2], seed)
    else:
        m = E.ExpModel(E.setting_cfg(st.replace('_sbi', '')), seed=seed)
    m.load_state_dict(sd)
    return m.cuda().eval()


def video_metrics(p, D):
    vid, vl = D.meta['vid'], D.meta['video_label']
    V = len(vl)
    scores, ok = np.full(V, np.nan), np.zeros(V, bool)
    for v in range(V):
        q = p[vid == v]
        if len(q) >= MIN_FRAMES:
            scores[v] = C.trimmed_mean(q, int(0.1 * len(q)))
            ok[v] = True
    m = C.summarize(scores[ok], vl[ok])
    out = {f'{k}': v for k, v in m.items()}
    out['n_videos'] = int(ok.sum())
    out['n_excluded'] = int((~ok).sum())
    meth = D.meta['video_method']
    for mm in sorted(set(meth[vl == 1])):
        sel = ok & ((vl == 0) | (meth == mm))
        if len(set(meth[vl == 1])) > 1:
            from sklearn.metrics import roc_auc_score
            out[f'auc_{mm}'] = roc_auc_score(vl[sel], scores[sel]) * 100
    return out, scores


@torch.no_grad()
def run(ds):
    D = NewData(ds)
    specs = model_specs()
    print(f'[{ds}] 프레임 {D.N}, 모델 {len(specs)}개 적재 중', flush=True)
    models = [build(sp) for sp in specs]
    probs = np.zeros((len(models), D.N), np.float32)
    t0 = time.time()
    for s in range(0, D.N, R.EVAL_BS):
        e = min(D.N, s + R.EVAL_BS)
        x, b3, lg, cls = D.batch(s, e)
        for j, (sp, m) in enumerate(zip(specs, models)):
            with torch.autocast('cuda', dtype=torch.float16):
                out = m(x, b3, cls) if sp[0] == 'S9' else m(x, b3, lg)
            probs[j, s:e] = torch.softmax(out['pred'].float(), 1)[:, 1].cpu().numpy()
        if (s // R.EVAL_BS) % 200 == 0:
            el = time.time() - t0
            print(f'[{ds}] {e}/{D.N} {el / 60:.1f}분, 남은 약 {el / e * (D.N - e) / 60:.1f}분', flush=True)
    p0 = (1 / (1 + np.exp(np.asarray(D.lg[:, 0], np.float64) - np.asarray(D.lg[:, 1], np.float64)))).astype(np.float32)
    rows = []
    m0, _ = video_metrics(p0, D)
    rows.append({'dataset': ds, 'stage': 'P0', 'setting': 'P0', 'seed': '', 'ckpt': '', **m0})
    for sp, p in zip(specs, probs):
        mm, _ = video_metrics(p, D)
        rows.append({'dataset': ds, 'stage': sp[0], 'setting': sp[1], 'seed': sp[2], 'ckpt': sp[3], **mm})
    np.savez(os.path.join(C.REPORTS, f's10_frame_probs_{ds}.npz'), p0=p0, probs=probs,
             names=np.array([f'{a}|{b}|{c}|{d}' for a, b, c, d, _ in specs]))
    return rows


def main():
    import pandas as pd
    allrows = []
    out = os.path.join(C.REPORTS, 's10_eval.csv')
    old = pd.read_csv(out) if os.path.exists(out) else None
    for ds in sys.argv[1:]:
        allrows += run(ds)
    df = pd.DataFrame(allrows)
    if old is not None:
        df = pd.concat([old[~old['dataset'].isin(sys.argv[1:])], df], ignore_index=True)
    df.to_csv(out, index=False)
    print('저장', out, len(df), '줄', flush=True)


if __name__ == '__main__':
    main()
