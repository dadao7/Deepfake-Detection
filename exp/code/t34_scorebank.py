"""TASKS4 A1·A2: DFD 신원 분할 + 점수 은행 생성.
은행 키: label|{set}, vid|{set}(영상 이름), score|{set}|{model} (영상 점수, 확률 공간)
set: ffval(8프레임 평균), fftest(100프레임 양쪽 10개 제외), cdf, fsh, dfdval, dfdtest (양쪽 floor(0.1n) 제외, 8장 미만 영상 제외)
model: P0 | S4|{설정}|{seed}|{ckpt} | S8|… | S9|…  (FF++ val 예측은 여기서 새로 계산)"""
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import eval_newdata as EV  # noqa: E402
import run_train as R  # noqa: E402
import t34_common as T  # noqa: E402

RUNPAT = {'S4': '{st}_d100_s{s}', 'S8': '{st}_s{s}', 'S9': '{st}_s{s}'}


def sig(lg):
    return 1 / (1 + np.exp(np.asarray(lg[:, 0], np.float64) - np.asarray(lg[:, 1], np.float64)))


@torch.no_grad()
def ffval_scores(specs):
    """FF++ val(8프레임/영상): 모든 모델의 프레임 확률 → 영상 평균"""
    import run_train_unfreeze as U
    VA = U.FFData('val')
    vid = VA.ix['vid']
    models = [EV.build(sp) for sp in specs]
    probs = np.zeros((len(models), VA.N), np.float32)
    for s in range(0, VA.N, R.EVAL_BS):
        r = np.arange(s, min(VA.N, s + R.EVAL_BS))
        x, b3, lg, y, cls = VA.batch(r)
        for j, (sp, m) in enumerate(zip(specs, models)):
            with torch.autocast('cuda', dtype=torch.float16):
                out = m(x, b3, cls) if sp[0] == 'S9' else m(x, b3, lg)
            probs[j, r] = torch.softmax(out['pred'].float(), 1)[:, 1].cpu().numpy()
    V = vid.max() + 1
    cnt = np.bincount(vid, minlength=V)
    vs = lambda p: np.bincount(vid, weights=p.astype(np.float64), minlength=V) / cnt  # noqa: E731
    lab = np.zeros(V, int)
    lab[vid] = VA.ix['label']
    p0 = sig(np.load(os.path.join(C.BASE_DIR, 'val_logit.npy')))
    return lab, vs(p0), [vs(p) for p in probs]


def newdata_scores(ds, specs, names_keep=None):
    D = EV.NewData(ds)
    z = np.load(os.path.join(C.REPORTS, f's10_frame_probs_{ds}.npz'))
    names = list(z['names'])
    vid, vl, keys = D.meta['vid'], D.meta['video_label'], D.meta['video_key']
    V = len(vl)
    cnt = np.bincount(vid, minlength=V)
    ok = cnt >= EV.MIN_FRAMES
    order = np.argsort(vid, kind='stable')
    starts = np.r_[0, np.cumsum(cnt)[:-1]]

    def vs(p):
        ps = p[order]
        out = np.full(V, np.nan)
        for v in np.where(ok)[0]:
            q = ps[starts[v]:starts[v] + cnt[v]]
            out[v] = C.trimmed_mean(q, int(0.1 * len(q)))
        return out
    res = {'P0': vs(z['p0'])}
    for sp in specs:
        res[f'{sp[0]}|{sp[1]}|{sp[2]}|{sp[3]}'] = vs(z['probs'][names.index(f'{sp[0]}|{sp[1]}|{sp[2]}|{sp[3]}')])
    return vl, keys, ok, res


def main():
    split = T.dfd_split()
    print(f'DFD 분할: val {len(split["val"])}영상, test {len(split["test"])}영상, 제외(신원 섞임) {len(split["dropped"])}', flush=True)
    specs = EV.model_specs()
    bank = {}
    # FF++ val
    lab, p0, plist = ffval_scores(specs)
    bank['label|ffval'] = lab
    bank['score|ffval|P0'] = p0
    for sp, p in zip(specs, plist):
        bank[f'score|ffval|{sp[0]}|{sp[1]}|{sp[2]}|{sp[3]}'] = p
    print('ffval 완료', flush=True)
    # FF++ test, CDF (runs의 pred 파일)
    te, cd = C.load_index('test'), C.load_index('cdf')
    for name, ix, logit, key, trim in (('fftest', te, 'test_logit.npy', 'ffpp_video_score', 'ffpp'),
                                       ('cdf', cd, 'cdf_logit.npy', 'cdf_video_score', 'frac')):
        v0 = C.video_scores(sig(np.load(os.path.join(C.BASE_DIR, logit))), ix['vid'], trim)
        vl = np.zeros(len(v0), int)
        vl[ix['vid']] = ix['label']
        bank[f'label|{name}'] = vl
        bank[f'score|{name}|P0'] = v0
        for sp in specs:
            p = os.path.join(C.RUNS, RUNPAT[sp[0]].format(st=sp[1], s=sp[2]), f'pred_{sp[3]}.npz')
            bank[f'score|{name}|{sp[0]}|{sp[1]}|{sp[2]}|{sp[3]}'] = np.load(p)[key]
    print('fftest·cdf 완료', flush=True)
    # FSH, DFD(val/test)
    for ds in ('fsh', 'dfd'):
        vl, keys, ok, res = newdata_scores(ds, specs)
        if ds == 'fsh':
            sel = {'fsh': ok}
        else:
            sv, st = set(split['val']), set(split['test'])
            sel = {'dfdval': ok & np.array([k in sv for k in keys]), 'dfdtest': ok & np.array([k in st for k in keys])}
        for name, m in sel.items():
            bank[f'label|{name}'] = vl[m]
            bank[f'vid|{name}'] = keys[m]
            for k, v in res.items():
                bank[f'score|{name}|{k}'] = v[m]
            print(f'{name}: 영상 {m.sum()} (진짜 {(vl[m] == 0).sum()}, 가짜 {(vl[m] == 1).sum()})', flush=True)
    np.savez(T.BANK_PATH, **bank)
    print('저장', T.BANK_PATH, len(bank), '항목', flush=True)


if __name__ == '__main__':
    main()
