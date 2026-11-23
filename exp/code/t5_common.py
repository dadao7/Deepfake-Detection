"""TASKS5 공통: 임의 모델 목록을 모든 세트(ffval·fftest·cdf·fsh·dfdval·dfdtest)에서 평가해 영상 점수를 만든다.
영상 점수 규칙은 TASKS4 점수 은행과 같다(ffval 8프레임 평균, fftest 100프레임 양쪽 10개 제외, cdf·fsh·dfd 양쪽 floor(0.1n) 제외,
새 데이터는 유효 프레임 8장 미만 영상 제외). 결과 은행: cache/scorebank_t5.npz (키 score|{set}|{이름}, TASKS4 은행은 건드리지 않음).
지표: AUC, ACC(기준값 0.5), bACC, 짝 bootstrap(여러 seed는 seed 평균 지표)."""
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import eval_newdata as EV  # noqa: E402
import run_train as R  # noqa: E402
import run_train_unfreeze as U  # noqa: E402
import t34_common as T  # noqa: E402

BANK5 = os.path.join(C.CACHE, 'scorebank_t5.npz')
ALL = T.SELECT_SETS + T.REPORT_SETS


def call(m, x, b3, lg, cls):
    """UModel은 (x, b3, cls), ExpModel은 (x, b3, baseline logit)"""
    return m(x, b3, cls) if isinstance(m, U.UModel) else m(x, b3, lg)


@torch.no_grad()
def _probs(models, n, getter, rows=None):
    for m in models:
        m.eval()
    rows = np.arange(n) if rows is None else np.asarray(rows)
    out = np.zeros((len(models), len(rows)), np.float32)
    for s in range(0, len(rows), R.EVAL_BS):
        r = rows[s:s + R.EVAL_BS]
        x, b3, lg, cls = getter(r)
        for j, m in enumerate(models):
            with torch.autocast('cuda', dtype=torch.float16):
                o = call(m, x, b3, lg, cls)
            out[j, s:s + len(r)] = torch.softmax(o['pred'].float(), 1)[:, 1].cpu().numpy()
    return out


def _ff(split, flip=False):
    """flip=True: 좌우 반전 이미지 + 반전 캐시(b_3·logit·CLS, build_flip_eval_cache.py)"""
    D = U.FFData(split, flip=flip)

    def g(r):
        r = np.asarray(r)
        x, b3, lg, y, cls = D.batch(r, np.ones(len(r), bool) if flip else None)
        return x, b3, lg, cls
    return D, g


def _nd(ds, flip=False):
    D = EV.NewData(ds)
    if flip:
        D.b3, D.lg, D.cls = [np.load(os.path.join(EV.ND, f'{ds}_flip_{k}.npy'), mmap_mode='r') for k in ('b3', 'logit', 'cls')]

    def g(r):
        r = np.asarray(r)
        if len(r) and r[-1] - r[0] + 1 == len(r):
            out = D.batch(int(r[0]), int(r[-1]) + 1)
        else:
            x = torch.from_numpy(np.ascontiguousarray(D.img[r])).cuda().permute(0, 3, 1, 2).float().div_(255.)
            out = (x, torch.from_numpy(np.ascontiguousarray(D.b3[r])).cuda(),
                   torch.from_numpy(np.ascontiguousarray(D.lg[r])).cuda(),
                   torch.from_numpy(np.ascontiguousarray(D.cls[r])).cuda())
        return (out[0].flip(3),) + tuple(out[1:]) if flip else out
    return D, g


def _probs_tta(models, D_g, n, rows=None, tta=False, mk=None):
    """tta=True면 원본·반전 프레임 확률 평균"""
    P = _probs(models, n, D_g, rows)
    if tta:
        P = (P + _probs(models, n, mk(), rows)) / 2
    return P


def _nd_video(D, p_full, keep):
    """p_full: (M, N) 프레임 확률 (N 전체 또는 keep 프레임만 채움) → (M, 영상 수) 점수 (규칙 미달은 nan)"""
    vid, vl = D.meta['vid'], D.meta['video_label']
    V = len(vl)
    cnt = np.bincount(vid, minlength=V)
    ok = cnt >= EV.MIN_FRAMES
    order = np.argsort(vid, kind='stable')
    st = np.r_[0, np.cumsum(cnt)[:-1]]
    out = np.full((len(p_full), V), np.nan)
    for j, p in enumerate(p_full):
        ps = p[order]
        for v in np.where(ok & keep)[0]:
            q = ps[st[v]:st[v] + cnt[v]]
            out[j, v] = C.trimmed_mean(q, int(0.1 * len(q)))
    return out, ok


def eval_models(models, sets=ALL, verbose=True, tta=False):
    """→ {set: (M, 영상 수) 영상 점수 (확률 공간)}. 영상 순서는 TASKS4 은행과 같다.
    tta=True: 프레임마다 원본·좌우 반전 확률을 평균한 뒤 같은 규칙으로 영상 점수(반전 캐시 필요)."""
    res = {}
    if 'ffval' in sets:
        D, g = _ff('val')
        P = _probs_tta(models, g, D.N, tta=tta, mk=lambda: _ff('val', True)[1])
        vid = D.ix['vid']
        V = vid.max() + 1
        cnt = np.bincount(vid, minlength=V)
        res['ffval'] = np.stack([np.bincount(vid, weights=p.astype(np.float64), minlength=V) / cnt for p in P])
        verbose and print('  ffval', flush=True)
    for s, sp, trim in (('fftest', 'test', 'ffpp'), ('cdf', 'cdf', 'frac')):
        if s in sets:
            D, g = _ff(sp)
            P = _probs_tta(models, g, D.N, tta=tta, mk=lambda: _ff(sp, True)[1])
            res[s] = np.stack([C.video_scores(p, D.ix['vid'], trim) for p in P])
            verbose and print(' ', s, flush=True)
    if 'fsh' in sets:
        D, g = _nd('fsh')
        P = _probs_tta(models, g, D.N, tta=tta, mk=lambda: _nd('fsh', True)[1])
        vs, ok = _nd_video(D, P, np.ones(len(D.meta['video_label']), bool))
        res['fsh'] = vs[:, ok]
        verbose and print('  fsh', flush=True)
    dsets = [s for s in ('dfdval', 'dfdtest') if s in sets]
    if dsets:
        split = T.dfd_split()
        D, g = _nd('dfd')
        keys = D.meta['video_key']
        want = {'dfdval': np.isin(keys, split['val']), 'dfdtest': np.isin(keys, split['test'])}
        keep = np.zeros(len(keys), bool)
        for s in dsets:
            keep |= want[s]
        rows = np.where(keep[D.meta['vid']])[0]            # 필요한 영상의 프레임만 계산
        Pk = _probs_tta(models, g, D.N, rows, tta=tta, mk=lambda: _nd('dfd', True)[1])
        P = np.zeros((len(models), D.N), np.float32)
        P[:, rows] = Pk
        vs, ok = _nd_video(D, P, keep)
        for s in dsets:
            res[s] = vs[:, ok & want[s]]
        verbose and print(' ', '·'.join(dsets), flush=True)
    return res


def load_bank5():
    if not os.path.exists(BANK5):
        return {}
    z = np.load(BANK5, allow_pickle=True)
    return {k: z[k] for k in z.files}


def save_bank5(add):
    b = load_bank5()
    b.update(add)
    np.savez(BANK5, **b)
    return len(b)


def labels():
    b = T.load_bank()
    return {s: b[f'label|{s}'] for s in ALL}


def acc(lab, s, th=0.5):
    p = np.asarray(s) >= th
    lab = np.asarray(lab).astype(bool)
    return float((p == lab).mean() * 100), float((p[lab].mean() + (~p[~lab]).mean()) / 2 * 100)


def boot_cmp(lab, new_list, base_list, B=2000, seed=0, th_new=0.5, th_base=0.5):
    """영상 단위 짝 bootstrap. 여러 seed면 seed 평균 지표끼리 비교. → {auc|acc|bacc: (Δ, lo, hi)}"""
    rng = np.random.default_rng(seed)
    lab = np.asarray(lab).astype(int)
    new_list = [np.asarray(s) for s in new_list]
    base_list = [np.asarray(s) for s in base_list]
    n = len(lab)

    thn = list(th_new) if np.ndim(th_new) else [th_new] * len(new_list)       # 기준값: 하나 또는 seed마다
    thb = list(th_base) if np.ndim(th_base) else [th_base] * len(base_list)

    def stats(i):
        li = lab[i]
        out = []
        for L, ths in ((new_list, thn), (base_list, thb)):
            a = np.mean([T.auc(li, s[i]) for s in L])
            ab = np.array([acc(li, s[i], th) for s, th in zip(L, ths)]).mean(0)
            out.append(np.r_[a, ab])
        return out[0] - out[1]
    point = stats(np.arange(n))
    ds = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if lab[i].min() == lab[i].max():
            continue
        ds.append(stats(i))
    ds = np.array(ds)
    lo, hi = np.quantile(ds, [0.025, 0.975], axis=0)
    return {k: (float(point[j]), float(lo[j]), float(hi[j])) for j, k in enumerate(('auc', 'acc', 'bacc'))}


def fmt(d, key='auc'):
    v, lo, hi = d[key]
    return f'{v:+.2f} [{lo:+.2f}, {hi:+.2f}]{" ✔" if (lo > 0 or hi < 0) else ""}'
