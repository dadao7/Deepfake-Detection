"""TASKS6 T1~T3 평가: 새 SBI 전문가(runs/X{tag}_s{seed})를 P3와 병합(선형 t, task vector a·b)해 선택 세트로 평가.
X가 P3 last에서 시작했으면(Xp3) A = P3 last(같은 경로 위 보간 = WiSE-FT), 아니면 A = P3 best(TASKS5와 같음).
  python code/t6_texp.py screen            # seed 0: 완료된 모든 X 변형 → reports/t6_texp_sel.json, 기준(TASKS5 seed 0)과 비교
  python code/t6_texp.py full Xema         # seed 0·1·2 선택 → 1위 변형을 모든 세트에서 평가(scorebank_t5 키 score|{set}|T6X|{이름}|{seed})"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common as C  # noqa: E402
import t34_common as T  # noqa: E402
import t5_common as T5  # noqa: E402
import t5_merge as M  # noqa: E402

SEL = os.path.join(C.REPORTS, 't6_texp_sel.json')
TS = (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
AB = [(a, b) for a in (0.25, 0.5, 0.75, 1.0) for b in (0.5, 0.75, 1.0)]


def a_ck(x):
    cfg = json.load(open(glob.glob(os.path.join(C.RUNS, f'{x}_s*', 'config.json'))[0], encoding='utf-8'))
    return 'last' if cfg['init'] == 'p3last' else 'best'


def variants(x, seed):
    s0 = M.init_state(seed)
    A = M.expert_state('P3', a_ck(x), seed, s0)
    out = []
    for ck in ('best', 'last'):
        B = M.expert_state(x, ck, seed, s0)
        for t in TS:
            out.append((f'{x}-{ck}|lin|t{t:.1f}', lambda B=B, t=t: M.combine(s0, A, B, 1 - t, t)))
        for a, b in AB:
            out.append((f'{x}-{ck}|tv|a{a:.2f}b{b:.2f}', lambda B=B, a=a, b=b: M.combine(s0, A, B, a, b)))
    return out


def eval_sel(seed, vs, res):
    lab = T5.labels()
    todo = [(n, f) for n, f in vs if n not in res.get(str(seed), {})]
    for i in range(0, len(todo), 24):
        chunk = todo[i:i + 24]
        models = [M.model_from(f(), seed) for _, f in chunk]
        sc = T5.eval_models(models, sets=T.SELECT_SETS, verbose=False)
        res.setdefault(str(seed), {}).update({n: {s: M.safe_auc(lab[s], sc[s][j]) for s in T.SELECT_SETS} for j, (n, _) in enumerate(chunk)})
        json.dump(res, open(SEL, 'w'), indent=0)
        del models
        torch.cuda.empty_cache()


def sel_score(d):
    v = [d[s] for s in T.SELECT_SETS]
    return float(np.mean(v)) if np.isfinite(v).all() else float('nan')


def screen():
    res = json.load(open(SEL)) if os.path.exists(SEL) else {}
    xs = sorted({os.path.basename(p).rsplit('_s', 1)[0] for p in glob.glob(os.path.join(C.RUNS, 'X*_s0')) if os.path.exists(os.path.join(p, 'DONE'))})
    for x in xs:
        t0 = time.time()
        eval_sel(0, variants(x, 0), res)
        print(f'{x} seed 0 선별 {(time.time() - t0) / 60:.1f}분', flush=True)
    base = json.load(open(os.path.join(C.REPORTS, 't5_merge_sel.json')))['0']
    ref = sel_score(base['tv|P3best+U3s|a0.50b0.75'])
    ref_best = max(sel_score(v) for v in base.values() if np.isfinite(sel_score(v)))
    print(f'기준(seed 0): TASKS5 선택 병합 {ref:.3f}, TASKS5 격자 seed 0 최고 {ref_best:.3f}')
    for x in xs:
        rows = sorted(((sel_score(v), n) for n, v in res['0'].items() if n.startswith(x + '-')), reverse=True)
        alone = {ck: sel_score(res['0'][f'{x}-{ck}|lin|t1.0']) for ck in ('best', 'last')}
        print(f'{x}: 단독 best {alone["best"]:.3f} / last {alone["last"]:.3f} | 병합 상위:', ', '.join(f'{n.split("-", 1)[1]} {s:.3f}' for s, n in rows[:4]))


def full(x):
    res = json.load(open(SEL)) if os.path.exists(SEL) else {}
    for seed in (0, 1, 2):
        if os.path.exists(os.path.join(C.RUNS, f'{x}_s{seed}', 'DONE')):
            eval_sel(seed, variants(x, seed), res)
    names = [n for n in res['0'] if n.startswith(x + '-')]
    sc = {n: np.mean([sel_score(res[str(s)][n]) for s in (0, 1, 2)]) for n in names if all(n in res.get(str(s), {}) for s in (0, 1, 2))}
    best = max(sc, key=sc.get)
    print(f'{x} 3seed 선택 1위: {best} {sc[best]:.3f}', flush=True)
    b5 = T5.load_bank5()
    for seed in (0, 1, 2):
        if f'score|fftest|T6X|{best}|{seed}' in b5:
            continue
        f = dict(variants(x, seed))[best]
        sc_ = T5.eval_models([M.model_from(f(), seed)], verbose=False)
        T5.save_bank5({f'score|{s}|T6X|{best}|{seed}': sc_[s][0] for s in T5.ALL})
        b5 = T5.load_bank5()
    json.dump({'x': x, 'best': best, 'sel': sc[best], 'top5': sorted(sc.items(), key=lambda kv: -kv[1])[:5]},
              open(os.path.join(C.REPORTS, f't6_texp_choice_{x}.json'), 'w'), indent=1)


if __name__ == '__main__':
    screen() if sys.argv[1] == 'screen' else full(sys.argv[2])
