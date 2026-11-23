"""TASKS4: 새로 학습한 모델(혼합 학습 MIX: U0m·U3m, anchor ANC: P4c_l*)을 모든 세트에서 평가해 점수 은행에 추가.
ffval(FF++ val 예측 새로 계산), fftest·cdf(runs pred 파일), fsh·dfdval·dfdtest(새 데이터 캐시 프레임 → 영상 점수).
은행 키: score|{set}|{MIX|ANC}|{설정}|{seed}|{ckpt}.  python code/t4_eval_new.py"""
import glob
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import eval_newdata as EV  # noqa: E402
import expmodel as E  # noqa: E402
import run_train as R  # noqa: E402
import run_train_unfreeze as U  # noqa: E402
import t34_common as T  # noqa: E402
import t34_scorebank as SB  # noqa: E402


def specs():
    out = []
    for rd in sorted(glob.glob(os.path.join(C.RUNS, 'U*m_s*'))) + sorted(glob.glob(os.path.join(C.RUNS, 'P4c_l*_s*'))):
        if not os.path.exists(os.path.join(rd, 'DONE')):
            continue
        name = os.path.basename(rd)
        st, seed = name.rsplit('_s', 1)
        stage = 'MIX' if name.startswith('U') else 'ANC'
        for ck in ('best', 'last'):
            out.append((stage, st, int(seed), ck, os.path.join(rd, f'ckpt_{ck}.pt')))
    return out


def build(sp):
    stage, st, seed, ck, path = sp
    m = U.UModel(st[:2], seed) if stage == 'MIX' else E.ExpModel(E.setting_cfg('P4a'), seed=seed)
    m.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    return m.cuda().eval()


@torch.no_grad()
def frame_probs(models, sps, D, n, getter):
    probs = np.zeros((len(models), n), np.float32)
    for s in range(0, n, R.EVAL_BS):
        e = min(n, s + R.EVAL_BS)
        x, b3, lg, cls = getter(s, e)
        for j, (sp, m) in enumerate(zip(sps, models)):
            with torch.autocast('cuda', dtype=torch.float16):
                out = m(x, b3, cls) if sp[0] == 'MIX' else m(x, b3, lg)
            probs[j, s:e] = torch.softmax(out['pred'].float(), 1)[:, 1].cpu().numpy()
    return probs


def main():
    sps = specs()
    if not sps:
        print('평가할 새 모델 없음')
        return
    bank = dict(T.load_bank())
    have = {k for k in bank if k.startswith('score|fftest|')}
    sps = [sp for sp in sps if f'score|fftest|{sp[0]}|{sp[1]}|{sp[2]}|{sp[3]}' not in have]
    print('새로 평가할 체크포인트', len(sps), flush=True)
    if not sps:
        return
    models = [build(sp) for sp in sps]
    key = lambda sp: f'{sp[0]}|{sp[1]}|{sp[2]}|{sp[3]}'  # noqa: E731
    # ffval
    VA = U.FFData('val')

    def gv(s, e):
        x, b3, lg, y, cls = VA.batch(np.arange(s, e))
        return x, b3, lg, cls
    P = frame_probs(models, sps, VA, VA.N, gv)
    vid = VA.ix['vid']
    V = vid.max() + 1
    cnt = np.bincount(vid, minlength=V)
    for sp, p in zip(sps, P):
        bank[f'score|ffval|{key(sp)}'] = np.bincount(vid, weights=p.astype(np.float64), minlength=V) / cnt
    # fftest, cdf
    for sp in sps:
        z = np.load(os.path.join(os.path.dirname(sp[4]), f'pred_{sp[3]}.npz'))
        bank[f'score|fftest|{key(sp)}'] = z['ffpp_video_score']
        bank[f'score|cdf|{key(sp)}'] = z['cdf_video_score']
    # fsh, dfd
    split = T.dfd_split()
    for ds in ('fsh', 'dfd'):
        D = EV.NewData(ds)
        P = frame_probs(models, sps, D, D.N, D.batch)
        vidn, vl, keys = D.meta['vid'], D.meta['video_label'], D.meta['video_key']
        Vn = len(vl)
        cn = np.bincount(vidn, minlength=Vn)
        ok = cn >= EV.MIN_FRAMES
        order = np.argsort(vidn, kind='stable')
        st = np.r_[0, np.cumsum(cn)[:-1]]
        masks = {'fsh': ok} if ds == 'fsh' else {'dfdval': ok & np.isin(keys, split['val']), 'dfdtest': ok & np.isin(keys, split['test'])}
        for sp, p in zip(sps, P):
            ps = p[order]
            vs = np.full(Vn, np.nan)
            for v in np.where(ok)[0]:
                q = ps[st[v]:st[v] + cn[v]]
                vs[v] = C.trimmed_mean(q, int(0.1 * len(q)))
            for name, m in masks.items():
                bank[f'score|{name}|{key(sp)}'] = vs[m]
        print(ds, '완료', flush=True)
    np.savez(T.BANK_PATH, **bank)
    print('점수 은행 갱신', len(bank), '항목', flush=True)
    for sp in sps:
        k = key(sp)
        print(k, ' | '.join(f'{s} {T.auc(bank[f"label|{s}"], bank[f"score|{s}|{k}"]):.2f}' for s in T.SELECT_SETS + T.REPORT_SETS))


if __name__ == '__main__':
    main()
