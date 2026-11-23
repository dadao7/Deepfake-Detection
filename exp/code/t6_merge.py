"""TASKS6 N1~N3: 학습 없는 병합 확장. 모든 모델은 같은 seed의 θ0(UModel('U3') 초기값)를 공유한다고 보고 task vector τ = θ − θ0로 다룬다.
 N1 모듈별 비율: θ = θ0 + b_top·τ_U3s[top] + a_br·τ_P3[branch] + b_br·τ_U3s[branch]  (P3의 상단 τ는 0)
 N2 TIES(텐서별 상위 k% 크기만 남김 → 부호 선출 → 같은 부호끼리 평균, ×λ) · DARE(확률 p로 버리고 1/(1−p) 재척도, 그 뒤 a·τ_P3 + b·τ_U3s)
 N3 다중 전문가: 기준(TASKS5 선택 task vector 병합)에 c·τ_e를 탐욕적으로 추가(c ∈ {0.25, 0.5}, 선택 점수 +0.10 이상일 때만 채택, 최대 2회)
BN running 통계: 계수가 양수인 전문가들의 볼록 결합(계수 비례), 모두 0이면 θ0.
선택 = FF++ val·DFD-val 평균 AUC(seed 3개 평균)만. 결과: reports/t6_sel.json, scorebank_t5.npz(키 score|{set}|T6|{이름}|{seed})
  python code/t6_merge.py sel | n3 | full | hyb
 N1b(hyb) 하이브리드: 분해에서 이득이 주로 상단에서 왔으므로 상단·분기 처리를 따로 고른다.
   상단 ∈ {선형 0.75·τ_U3s, TIES식 상위 k%(k 0.1·0.2·0.3, λ 1.0)·(k 0.2, λ 0.75), 0.75·τ_U0s(상단만 SBI 학습)}
   분기 ∈ {P3 그대로, 0.5·τ_P3 + 0.5·τ_U3s, TIES k0.2 λ1.0}  → 18개/seed, 선택 1위를 전체 평가"""
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

SEEDS = (0, 1, 2)
SEL = os.path.join(C.REPORTS, 't6_sel.json')
BASE = {'P3': ('P3', 'best'), 'U3s': ('U3s', 'best')}          # TASKS5 선택 쌍
EXTRA = {'U3f': ('U3f', 'best'), 'P3sbi': ('P3_sbi', 'best'), 'U3m': ('U3m', 'best'), 'Ds': ('Ds_b1', 'last'),
         'P4c': ('P4c_l100', 'best'), 'U0s': ('U0s', 'best'), 'U0f': ('U0f', 'best')}
STAT = ('running_mean', 'running_var')


def load_any(name, ck, seed, s0):
    """runs/의 체크포인트를 UModel 키 공간으로. ExpModel(분기만)·U0(상단만)은 없는 부분을 θ0로 채움(τ = 0)."""
    if name == 'P3':
        return M.expert_state('P3', ck, seed, s0)
    sd = torch.load(os.path.join(C.RUNS, f'{name}_s{seed}', f'ckpt_{ck}.pt'), map_location='cpu', weights_only=False)['model']
    if not any(k.startswith(('top.', 'branch.')) for k in sd):
        sd = {'branch.' + k: v for k, v in sd.items()}
    out = {k: v.clone() for k, v in s0.items()}
    for k, v in sd.items():
        assert k in out, (name, k)
        out[k] = v
    return out


def module_of(k):
    return 'top' if k.startswith('top.') else 'branch'


def mix(s0, experts, coef):
    """experts: {이름: state}, coef: {이름: {'top': c, 'branch': c}} → θ0 + Σ c·τ (BN 통계는 볼록 결합)"""
    out = {}
    for k, v0 in s0.items():
        if not torch.is_floating_point(v0):
            out[k] = v0
            continue
        mod = module_of(k)
        cs = {n: coef[n].get(mod, 0.0) for n in experts}
        if k.endswith(STAT):
            w = {n: c for n, c in cs.items() if c > 0}
            out[k] = v0 if not w else (sum(c * experts[n][k].float() for n, c in w.items()) / sum(w.values())).to(v0.dtype)
        else:
            out[k] = (v0.float() + sum(c * (experts[n][k].float() - v0.float()) for n, c in cs.items() if c)).to(v0.dtype)
    return out


def ties(s0, experts, k_frac, lam):
    out = {}
    for key, v0 in s0.items():
        if not torch.is_floating_point(v0):
            out[key] = v0
            continue
        if key.endswith(STAT):
            out[key] = (sum(e[key].float() for e in experts.values()) / len(experts)).to(v0.dtype)
            continue
        taus = []
        for e in experts.values():
            t = e[key].float() - v0.float()
            if t.numel() > 1 and k_frac < 1:
                th = t.abs().flatten().kthvalue(max(1, int(round((1 - k_frac) * t.numel())))).values
                t = torch.where(t.abs() >= th, t, torch.zeros_like(t))
            taus.append(t)
        st = torch.stack(taus)
        sign = torch.sign(st.sum(0))
        agree = (torch.sign(st) == sign) & (st != 0)
        merged = (st * agree).sum(0) / agree.sum(0).clamp(min=1)
        out[key] = (v0.float() + lam * merged).to(v0.dtype)
    return out


def dare(s0, experts, p, coefs, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    dropped = {}
    for n, e in experts.items():
        d = {}
        for key, v0 in s0.items():
            if torch.is_floating_point(v0) and not key.endswith(STAT):
                t = e[key].float() - v0.float()
                m = (torch.rand(t.shape, generator=g) >= p).float()
                d[key] = (v0.float() + t * m / (1 - p)).to(v0.dtype)
            else:
                d[key] = e[key]
        dropped[n] = d
    return mix(s0, dropped, {n: {'top': c, 'branch': c} for n, c in coefs.items()})


def n1n2_variants(seed):
    s0 = M.init_state(seed)
    ex = {n: load_any(*BASE[n], seed, s0) for n in BASE}
    vs = []
    for bt in (0.0, 0.5, 0.75, 1.0):
        for ab, bb in ((1.0, 0.0), (0.0, 1.0), (0.5, 0.75), (0.5, 0.5), (0.75, 0.5), (0.25, 0.75), (0.5, 1.0), (0.75, 0.75)):
            vs.append((f'mod|top{bt:.2f}|br{ab:.2f}_{bb:.2f}', lambda bt=bt, ab=ab, bb=bb: mix(
                s0, ex, {'P3': {'top': 0.0, 'branch': ab}, 'U3s': {'top': bt, 'branch': bb}})))
    for kf in (0.2, 0.5):
        for lam in (0.75, 1.0, 1.25, 1.5):
            vs.append((f'ties|k{kf:.1f}|l{lam:.2f}', lambda kf=kf, lam=lam: ties(s0, ex, kf, lam)))
    for p in (0.5, 0.9):
        for a, b in ((0.5, 0.75), (1.0, 1.0)):
            vs.append((f'dare|p{p:.1f}|a{a:.2f}b{b:.2f}', lambda p=p, a=a, b=b: dare(s0, ex, p, {'P3': a, 'U3s': b}, seed)))
    return vs


def eval_sel(seed, variants, res, tag):
    lab = T5.labels()
    todo = [(n, f) for n, f in variants if n not in res.get(str(seed), {})]
    if not todo:
        return
    t0 = time.time()
    for i in range(0, len(todo), 24):                              # 메모리: 24개씩
        chunk = todo[i:i + 24]
        models = [M.model_from(f(), seed) for _, f in chunk]
        sc = T5.eval_models(models, sets=T.SELECT_SETS, verbose=False)
        res.setdefault(str(seed), {}).update({n: {s: M.safe_auc(lab[s], sc[s][j]) for s in T.SELECT_SETS}
                                             for j, (n, _) in enumerate(chunk)})
        json.dump(res, open(SEL, 'w'), indent=0)
        del models
        torch.cuda.empty_cache()
    print(f'[{tag}] seed {seed}: {len(todo)}개 {(time.time() - t0) / 60:.1f}분', flush=True)


def score(res, n):
    v = [np.mean([res[str(s)][n][k] for k in T.SELECT_SETS]) for s in SEEDS]
    return float(np.mean(v)) if np.isfinite(v).all() else float('nan')


def sel_stage():
    res = json.load(open(SEL)) if os.path.exists(SEL) else {}
    for seed in SEEDS:
        eval_sel(seed, n1n2_variants(seed), res, 'N1N2')
    names = [n for n, _ in n1n2_variants(0)]
    rows = sorted(((score(res, n), n) for n in names), key=lambda x: -x[0] if np.isfinite(x[0]) else 1e9)
    for sc_, n in rows[:12]:
        print(f'{n:32s} {sc_:.3f}')
    return res


def base_coef():
    return {'P3': {'top': 0.0, 'branch': 0.5}, 'U3s': {'top': 0.75, 'branch': 0.75}}   # = tv|P3best+U3s|a0.50b0.75


def n3_stage():
    res = json.load(open(SEL)) if os.path.exists(SEL) else {}
    chosen = []                                   # [(전문가, c)]
    base_name = 'n3|base'
    for seed in SEEDS:
        s0 = M.init_state(seed)
        ex = {n: load_any(*BASE[n], seed, s0) for n in BASE}
        eval_sel(seed, [(base_name, lambda: mix(s0, ex, base_coef()))], res, 'N3 기준')
    cur = score(res, base_name)
    print(f'N3 기준 {cur:.3f}', flush=True)
    for rnd in (1, 2):
        cands = [(e, c) for e in EXTRA for c in (0.25, 0.5) if e not in [x for x, _ in chosen]]
        for seed in SEEDS:
            s0 = M.init_state(seed)
            ex = {n: load_any(*BASE[n], seed, s0) for n in BASE}
            for e, _ in chosen:
                ex[e] = load_any(*EXTRA[e], seed, s0)
            vs = []
            for e, c in cands:
                if e not in ex:
                    ex[e] = load_any(*EXTRA[e], seed, s0)
                name = 'n3|' + '+'.join(f'{x}{cc:.2f}' for x, cc in chosen + [(e, c)])

                def f(e=e, c=c):
                    coef = base_coef()
                    for x, cc in chosen + [(e, c)]:
                        coef[x] = {'top': cc, 'branch': cc}
                    return mix(s0, {n: ex[n] for n in list(BASE) + [x for x, _ in chosen + [(e, c)]]}, coef)
                vs.append((name, f))
            eval_sel(seed, vs, res, f'N3 {rnd}회')
        scored = []
        for e, c in cands:
            name = 'n3|' + '+'.join(f'{x}{cc:.2f}' for x, cc in chosen + [(e, c)])
            scored.append((score(res, name), e, c, name))
        scored.sort(key=lambda x: -x[0] if np.isfinite(x[0]) else 1e9)
        for x in scored[:6]:
            print(f'  {x[3]:40s} {x[0]:.3f}')
        if scored and scored[0][0] >= cur + 0.10:
            cur = scored[0][0]
            chosen.append((scored[0][1], scored[0][2]))
            print(f'N3 {rnd}회 채택: {scored[0][3]} → {cur:.3f}', flush=True)
        else:
            print(f'N3 {rnd}회: +0.10 이상 개선 없음 → 멈춤', flush=True)
            break
    json.dump({'chosen': chosen, 'score': cur}, open(os.path.join(C.REPORTS, 't6_n3_choice.json'), 'w'), indent=1)


def build_named(name, seed):
    s0 = M.init_state(seed)
    ex = {n: load_any(*BASE[n], seed, s0) for n in BASE}
    for f in n1n2_variants(seed):
        if f[0] == name:
            return f[1]()
    assert name.startswith('n3|'), name
    coef = base_coef()
    if name != 'n3|base':
        for part in name[3:].split('+'):
            e, c = part[:-4], float(part[-4:])
            ex[e] = load_any(*EXTRA[e], seed, s0)
            coef[e] = {'top': c, 'branch': c}
    return mix(s0, ex, coef)


def full_stage():
    res = json.load(open(SEL))
    names = [n for n, _ in n1n2_variants(0)]
    pick = {}
    for fam in ('mod', 'ties', 'dare'):
        cand = [(score(res, n), n) for n in names if n.startswith(fam + '|') and np.isfinite(score(res, n))]
        pick[fam] = max(cand)
    # 분해(상단만 / 분기만 병합)도 보고용으로 전체 평가
    pick['mod_top_only'] = (score(res, 'mod|top0.75|br1.00_0.00'), 'mod|top0.75|br1.00_0.00')
    pick['mod_branch_only'] = (score(res, 'mod|top0.00|br0.50_0.75'), 'mod|top0.00|br0.50_0.75')
    n3 = json.load(open(os.path.join(C.REPORTS, 't6_n3_choice.json'))) if os.path.exists(os.path.join(C.REPORTS, 't6_n3_choice.json')) else None
    if n3 and n3['chosen']:
        nm = 'n3|' + '+'.join(f'{x}{cc:.2f}' for x, cc in n3['chosen'])
        pick['n3'] = (n3['score'], nm)
    print('전체 평가 대상:', pick, flush=True)
    b5 = T5.load_bank5()
    for seed in SEEDS:
        todo = [n for _, n in pick.values() if f'score|fftest|T6|{n}|{seed}' not in b5]
        if not todo:
            continue
        t0 = time.time()
        models = [M.model_from(build_named(n, seed), seed) for n in todo]
        sc = T5.eval_models(models, verbose=False)
        T5.save_bank5({f'score|{s}|T6|{n}|{seed}': sc[s][j] for j, n in enumerate(todo) for s in T5.ALL})
        b5 = T5.load_bank5()
        print(f'seed {seed} 전체 평가 {len(todo)}개 {(time.time() - t0) / 60:.1f}분', flush=True)
        del models
        torch.cuda.empty_cache()
    json.dump(pick, open(os.path.join(C.REPORTS, 't6_merge_choice.json'), 'w'), indent=1, ensure_ascii=False)


TOPS = {'lin0.75': None, 'tk0.1': (0.1, 1.0), 'tk0.2': (0.2, 1.0), 'tk0.3': (0.3, 1.0), 'tk0.2l0.75': (0.2, 0.75), 'U0s0.75': 'U0s'}
BRS = ('P3', 'avg', 'ties')


def hybrid(seed, top, br, s0=None, ex=None):
    s0 = s0 or M.init_state(seed)
    ex = ex or {n: load_any(*BASE[n], seed, s0) for n in BASE}
    if TOPS[top] is None:
        st = mix(s0, ex, {'P3': {'top': 0.0}, 'U3s': {'top': 0.75}})
    elif TOPS[top] == 'U0s':
        st = mix(s0, {'U0s': load_any(*EXTRA['U0s'], seed, s0)}, {'U0s': {'top': 0.75}})
    else:
        st = ties(s0, {'U3s': ex['U3s']}, *TOPS[top])
    if br == 'P3':
        sb = ex['P3']
    elif br == 'avg':
        sb = mix(s0, ex, {'P3': {'branch': 0.5}, 'U3s': {'branch': 0.5}})
    else:
        sb = ties(s0, ex, 0.2, 1.0)
    return {k: (st[k] if k.startswith('top.') else sb[k]) for k in s0}


def hyb_stage():
    res = json.load(open(SEL)) if os.path.exists(SEL) else {}
    for seed in SEEDS:
        s0 = M.init_state(seed)
        ex = {n: load_any(*BASE[n], seed, s0) for n in BASE}
        vs = [(f'hyb|{t}|{b}', lambda t=t, b=b: hybrid(seed, t, b, s0, ex)) for t in TOPS for b in BRS]
        eval_sel(seed, vs, res, 'N1b')
    names = [f'hyb|{t}|{b}' for t in TOPS for b in BRS]
    rows = sorted(((score(res, n), n) for n in names), reverse=True)
    for sc_, n in rows:
        print(f'{n:28s} {sc_:.3f}')
    best = rows[0]
    b5 = T5.load_bank5()
    for seed in SEEDS:
        if f'score|fftest|T6|{best[1]}|{seed}' in b5:
            continue
        t, b = best[1].split('|')[1:]
        sc = T5.eval_models([M.model_from(hybrid(seed, t, b), seed)], verbose=False)
        T5.save_bank5({f'score|{s}|T6|{best[1]}|{seed}': sc[s][0] for s in T5.ALL})
        b5 = T5.load_bank5()
    json.dump({'best': best, 'rows': rows}, open(os.path.join(C.REPORTS, 't6_hyb_choice.json'), 'w'), indent=1)


if __name__ == '__main__':
    {'sel': sel_stage, 'n3': n3_stage, 'full': full_stage, 'hyb': hyb_stage}[sys.argv[1]]()
