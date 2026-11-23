"""TASKS6 N4: 영상 점수 집계 방식. 프레임 확률 → 영상 점수를 여러 방식으로 만들고, 모델마다 선택 세트(FF++ val·DFD-val)로 하나를 고른다.
공정성: P0·P3도 같은 후보에서 각자 고른다. 기본(지금까지): 양쪽 10%(FF++ test는 10장) 제외 평균(trimmed).
후보: trimmed, mean, median, logit_mean(logit 평균 → sigmoid), top50(상위 50% 평균), trim_logit(양쪽 10% 제외 logit 평균)
결과: cache/t6_frames/{이름}.npz(프레임 확률), scorebank_t5 키 score|{set}|AGG|{이름}|{방식}|{seed}, reports/t6_agg.md
  python code/t6_agg.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common as C  # noqa: E402
import eval_newdata as EV  # noqa: E402
import t34_common as T  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_merge as M  # noqa: E402
import t5_tta as TT  # noqa: E402

FR = os.path.join(C.CACHE, 't6_frames')
AGGS = ('trimmed', 'mean', 'median', 'logit_mean', 'top50', 'trim_logit')
SEEDS = (0, 1, 2)


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def agg(q, how, trim):
    q = np.sort(np.asarray(q, np.float64))
    n = len(q)
    t = trim if n > 2 * trim else 0
    if how == 'trimmed':
        return q[t:n - t].mean()
    if how == 'mean':
        return q.mean()
    if how == 'median':
        return float(np.median(q))
    if how == 'logit_mean':
        return 1 / (1 + np.exp(-lg(q).mean()))
    if how == 'top50':
        return q[n // 2:].mean()
    if how == 'trim_logit':
        return 1 / (1 + np.exp(-lg(q[t:n - t]).mean()))
    raise ValueError(how)


def frame_probs(model):
    """→ {'ffval','fftest','cdf','fsh','dfd'}: 프레임 확률 (TASKS5 평가와 같은 경로)"""
    out = {}
    for s, sp in (('ffval', 'val'), ('fftest', 'test'), ('cdf', 'cdf')):
        D, g = T5._ff(sp)
        out[s] = T5._probs([model], D.N, g)[0]
    for ds in ('fsh', 'dfd'):
        D, g = T5._nd(ds)
        out[ds] = T5._probs([model], D.N, g)[0]
    return out


def video_scores(P, how):
    """프레임 확률 → 보고·선택 세트 영상 점수(영상 순서·제외 규칙은 TASKS4 은행과 같음)"""
    res = {}
    for s, sp, rule in (('ffval', 'val', 'none'), ('fftest', 'test', 'ffpp'), ('cdf', 'cdf', 'frac')):
        vid = C.load_index(sp)['vid']
        V = vid.max() + 1
        order = np.argsort(vid, kind='stable')
        cnt = np.bincount(vid, minlength=V)
        st = np.r_[0, np.cumsum(cnt)[:-1]]
        ps = P[s][order]
        res[s] = np.array([agg(ps[st[v]:st[v] + cnt[v]], how, 0 if rule == 'none' else (10 if rule == 'ffpp' else int(0.1 * cnt[v])))
                           for v in range(V)])
    split = T.dfd_split()
    for ds in ('fsh', 'dfd'):
        D = EV.NewData(ds)
        vid, keys = D.meta['vid'], D.meta['video_key']
        V = len(D.meta['video_label'])
        cnt = np.bincount(vid, minlength=V)
        ok = cnt >= EV.MIN_FRAMES
        order = np.argsort(vid, kind='stable')
        st = np.r_[0, np.cumsum(cnt)[:-1]]
        ps = P[ds][order]
        vs = np.full(V, np.nan)
        for v in np.where(ok)[0]:
            vs[v] = agg(ps[st[v]:st[v] + cnt[v]], how, int(0.1 * cnt[v]))
        if ds == 'fsh':
            res['fsh'] = vs[ok]
        else:
            res['dfdval'] = vs[ok & np.isin(keys, split['val'])]
            res['dfdtest'] = vs[ok & np.isin(keys, split['test'])]
    return res


def targets():
    out = [('P0', 0, lambda: TT.P0Model().cuda())]
    for sd in SEEDS:
        out.append(('P3', sd, lambda sd=sd: EV.build(('S4', 'P3', sd, 'last', os.path.join(C.RUNS, f'P3_d100_s{sd}', 'ckpt_last.pt')))))
        out.append(('병합TV', sd, lambda sd=sd: M.model_from(M.combine(M.init_state(sd), M.expert_state('P3', 'best', sd, M.init_state(sd)),
                                                                      M.expert_state('U3s', 'best', sd, M.init_state(sd)), 0.5, 0.75), sd)))
    return out


def main():
    os.makedirs(FR, exist_ok=True)
    lab = T5.labels()
    b5 = T5.load_bank5()
    add = {}
    for name, sd, mk in targets():
        fp = os.path.join(FR, f'{name}_{sd}.npz')
        if not os.path.exists(fp):
            P = frame_probs(mk())
            np.savez(fp, **P)
            torch.cuda.empty_cache()
        P = dict(np.load(fp))
        for how in AGGS:
            vs = video_scores(P, how)
            for s in T5.ALL:
                add[f'score|{s}|AGG|{name}|{how}|{sd}'] = vs[s]
        print(name, sd, '완료', flush=True)
    T5.save_bank5(add)
    b5 = T5.load_bank5()
    # 확인: trimmed = TASKS4·5 은행 값
    chk = np.nanmax(np.abs(b5['score|fftest|AGG|P3|trimmed|0'] - T.load_bank()['score|fftest|S4|P3|0|last']))
    md = ['# N4: 영상 점수 집계 방식 (자동 생성: code/t6_agg.py)\n',
          f'기본(trimmed) 재현 확인: P3 seed 0 FF++ test 영상 점수 최대 차 {chk:.2e}. 선택 = FF++ val·DFD-val 평균 AUC(seed 평균). 모델마다 따로 고름.\n',
          '| 모델 | 집계 | 선택 점수 | ' + ' | '.join(T.REPORT_SETS) + ' |', '|---|---|---|' + '---|' * 4]
    choice = {}
    for name in ('P0', 'P3', '병합TV'):
        sds = (0,) if name == 'P0' else SEEDS
        best = None
        for how in AGGS:
            a = {s: np.mean([T.auc(lab[s], b5[f'score|{s}|AGG|{name}|{how}|{sd}']) for sd in sds]) for s in T5.ALL}
            sel = np.mean([a[s] for s in T.SELECT_SETS])
            md.append(f'| {name} | {how} | {sel:.2f} | ' + ' | '.join(f'{a[s]:.2f}' for s in T.REPORT_SETS) + ' |')
            if best is None or sel > best[0]:
                best = (sel, how)
        choice[name] = best[1]
        md.append(f'| **{name} 선택** | **{best[1]}** | | | | | |')
    md += ['', '## 고른 집계로 비교 (P3도 자기 선택 집계 사용)\n', '| 비교 | ' + ' | '.join(T.REPORT_SETS) + ' |', '|---|' + '---|' * 4]
    def sc(name, how):
        sds = (0,) if name == 'P0' else SEEDS
        return {s: [b5[f'score|{s}|AGG|{name}|{how}|{sd}'] for sd in sds] for s in T5.ALL}
    for a_name, b_name in (('병합TV', 'P3'), ('P3', 'P0')):
        A, B = sc(a_name, choice[a_name]), sc(b_name, choice[b_name])
        cells = [T5.fmt(T5.boot_cmp(lab[s], A[s], B[s], B=1000)) for s in T.REPORT_SETS]
        md.append(f'| {a_name}({choice[a_name]}) − {b_name}({choice[b_name]}) | ' + ' | '.join(cells) + ' |')
    for name in ('P3', '병합TV'):
        A, B = sc(name, choice[name]), sc(name, 'trimmed')
        cells = [T5.fmt(T5.boot_cmp(lab[s], A[s], B[s], B=1000)) for s in T.REPORT_SETS]
        md.append(f'| {name}: {choice[name]} − trimmed | ' + ' | '.join(cells) + ' |')
    json.dump(choice, open(os.path.join(C.REPORTS, 't6_agg_choice.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    open(os.path.join(C.REPORTS, 't6_agg.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    main()
