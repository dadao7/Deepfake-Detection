"""TASKS5 M1: 데이터 전문가 가중치 병합 (학습 없음, 결과는 단일 모델 = P3와 같은 추론 비용).
근거: 같은 seed의 P3·U3f·U3s는 초기값이 같다(저자 baseline 상단 + SBI 사전학습 분기 + seed 규칙 gating 초기화).
  θ0 = UModel('U3', seed) 초기 상태, P3는 baseline 상단이 θ0(고정)인 U3로 본다.
  선형: θ(t) = (1−t)·θ_A + t·θ_B   (A = FF++ 전문가, B = SBI 전문가 U3s best)
  task vector: θ = θ0 + a·(θ_A − θ0) + b·(θ_B − θ0)
BN running 통계는 두 전문가 값의 볼록 결합 (wa·A + wb·B)/(wa+wb) (외삽 시 분산 음수 방지, num_batches_tracked는 A 값).
점수에 NaN/inf가 있으면 그 변형의 AUC는 nan으로 기록하고 순위에서 뺀다. 선택 = FF++ val·DFD-val 평균 AUC(seed 평균)만.
  python code/t5_merge.py sel      # 1단계: 선택 세트에서 격자 → reports/t5_merge_sel.json
  python code/t5_merge.py full     # 2단계: 고른 쌍의 t 곡선 전체 + task vector 최고를 모든 세트에서 → scorebank_t5.npz
  python code/t5_merge.py report   # 3단계: reports/t5_merge.md
  python code/t5_merge.py ablate   # 대조: WiSE-FT형 θ0 + b·(θ_U3s − θ0) (FF++ 전문가 없이 SBI 학습량만 줄임) — 선택 세트·보고 세트 모두
                                   #       → 병합 이득이 FF++ 전문가 덕분인지, 초기값 쪽으로 줄인 효과인지 구분"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common as C  # noqa: E402
import run_train_unfreeze as U  # noqa: E402
import t34_common as T  # noqa: E402
import t5_common as T5  # noqa: E402

SEEDS = (0, 1, 2)
TS = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
PAIRS = {'P3last+U3s': (('P3', 'last'), ('U3s', 'best')),
         'P3best+U3s': (('P3', 'best'), ('U3s', 'best')),
         'U3f+U3s': (('U3f', 'best'), ('U3s', 'best'))}
TV_AB = [(a, b) for a in (0.5, 0.75, 1.0, 1.25) for b in (0.25, 0.5, 0.75, 1.0)]
SEL_PATH = os.path.join(C.REPORTS, 't5_merge_sel.json')


def safe_auc(lab, sc):
    return T.auc(lab, sc) if np.isfinite(sc).all() else float('nan')


def init_state(seed):
    torch.manual_seed(0)
    return {k: v.clone() for k, v in U.UModel('U3', seed).state_dict().items()}


def expert_state(name, ck, seed, s0):
    if name == 'P3':
        sd = torch.load(os.path.join(C.RUNS, f'P3_d100_s{seed}', f'ckpt_{ck}.pt'), map_location='cpu', weights_only=False)['model']
        out = {k: v.clone() for k, v in s0.items()}
        for k, v in sd.items():
            assert 'branch.' + k in out, k
            out['branch.' + k] = v
        return out
    return torch.load(os.path.join(C.RUNS, f'{name}_s{seed}', f'ckpt_{ck}.pt'), map_location='cpu', weights_only=False)['model']


def combine(s0, sa, sb, wa, wb):
    """θ0 + wa(θA−θ0) + wb(θB−θ0). 선형 보간은 wa = 1−t, wb = t (θ0 항이 사라짐)"""
    out = {}
    for k, v0 in s0.items():
        if not torch.is_floating_point(v0):
            out[k] = sa[k]
        elif k.endswith('running_mean') or k.endswith('running_var'):
            # BN 통계는 task vector가 아니라 데이터 통계 → 두 전문가 값의 볼록 결합(외삽하면 분산이 음수가 될 수 있음)
            out[k] = ((wa * sa[k].float() + wb * sb[k].float()) / (wa + wb)).to(v0.dtype)
        else:
            out[k] = v0.float() + wa * (sa[k].float() - v0.float()) + wb * (sb[k].float() - v0.float())
            out[k] = out[k].to(v0.dtype)
    return out


def model_from(sd, seed):
    m = U.UModel('U3', seed)
    m.load_state_dict(sd)
    return m.cuda().eval()


def variants(seed, pairs=PAIRS, ts=TS, tv=True):
    s0 = init_state(seed)
    out = []
    cache = {}
    for pname, (A, B) in pairs.items():
        for key in (A, B):
            if key not in cache:
                cache[key] = expert_state(*key, seed, s0)
        sa, sb = cache[A], cache[B]
        for t in ts:
            out.append((f'lin|{pname}|t{t:.1f}', combine(s0, sa, sb, 1 - t, t)))
        if tv:
            for a, b in TV_AB:
                out.append((f'tv|{pname}|a{a:.2f}b{b:.2f}', combine(s0, sa, sb, a, b)))
    return out


def sel_stage():
    lab = T5.labels()
    res = json.load(open(SEL_PATH)) if os.path.exists(SEL_PATH) else {}
    for seed in SEEDS:
        if str(seed) in res:
            continue
        t0 = time.time()
        vs = variants(seed)
        names = [n for n, _ in vs]
        models = [model_from(sd, seed) for _, sd in vs]
        del vs
        sc = T5.eval_models(models, sets=T.SELECT_SETS, verbose=False)
        res[str(seed)] = {n: {s: safe_auc(lab[s], sc[s][j]) for s in T.SELECT_SETS} for j, n in enumerate(names)}
        json.dump(res, open(SEL_PATH, 'w'), indent=0)
        print(f'seed {seed}: {len(names)}개 {(time.time() - t0) / 60:.1f}분', flush=True)
        del models
        torch.cuda.empty_cache()
    # 요약
    names = list(res['0'].keys())
    rows = []
    for n in names:
        v = np.mean([[res[str(s)][n][k] for k in T.SELECT_SETS] for s in SEEDS], 0)
        if not np.isfinite(v).all():
            print('제외(점수 nan):', n, flush=True)
            continue
        rows.append((float(v.mean()), n, float(v[0]), float(v[1])))
    rows.sort(reverse=True)
    for r in rows[:15]:
        print(f'{r[1]:32s} 선택 {r[0]:.2f} | ffval {r[2]:.2f} dfdval {r[3]:.2f}')
    for pname in PAIRS:
        print(pname, ' '.join(f'{t:.1f}:{np.mean([np.mean(list(res[str(s)][f"lin|{pname}|t{t:.1f}"].values())) for s in SEEDS]):.2f}' for t in TS))
    return res, rows


def full_stage():
    res, rows = sel_stage()
    best_lin = max((r for r in rows if r[1].startswith('lin|')), key=lambda r: r[0])
    best_tv = max((r for r in rows if r[1].startswith('tv|')), key=lambda r: r[0])
    pname = best_lin[1].split('|')[1]
    print('선택: 선형', best_lin[1], f'{best_lin[0]:.2f} / task vector', best_tv[1], f'{best_tv[0]:.2f}', flush=True)
    bank5 = T5.load_bank5()
    for seed in SEEDS:
        vs = variants(seed, pairs={pname: PAIRS[pname]}, tv=False)
        tvp = best_tv[1].split('|')[1]
        a, b = best_tv[1].split('|')[2][1:].split('b')
        s0 = init_state(seed)
        A, B = PAIRS[tvp]
        vs.append((best_tv[1], combine(s0, expert_state(*A, seed, s0), expert_state(*B, seed, s0), float(a), float(b))))
        vs = [(n, sd) for n, sd in vs if f'score|fftest|M1|{n}|{seed}' not in bank5]
        if not vs:
            continue
        t0 = time.time()
        names = [n for n, _ in vs]
        models = [model_from(sd, seed) for _, sd in vs]
        del vs
        sc = T5.eval_models(models, verbose=False)
        add = {f'score|{s}|M1|{n}|{seed}': sc[s][j] for j, n in enumerate(names) for s in T5.ALL}
        T5.save_bank5(add)
        bank5.update(add)
        print(f'seed {seed} 전체 평가 {len(names)}개 {(time.time() - t0) / 60:.1f}분', flush=True)
        del models
        torch.cuda.empty_cache()
    json.dump({'best_lin': best_lin, 'best_tv': best_tv}, open(os.path.join(C.REPORTS, 't5_merge_choice.json'), 'w'), indent=1)


WISE_B = (0.5, 0.6, 0.7, 0.8, 0.9)


def ablate():
    bank5 = T5.load_bank5()
    for seed in SEEDS:
        s0 = init_state(seed)
        B = expert_state('U3s', 'best', seed, s0)
        vs = [(f'wise|U3s|b{b:.1f}', combine(s0, B, B, 0.0, b)) for b in WISE_B]
        vs = [(n, sd) for n, sd in vs if f'score|fftest|M1|{n}|{seed}' not in bank5]
        if not vs:
            continue
        t0 = time.time()
        names = [n for n, _ in vs]
        models = [model_from(sd, seed) for _, sd in vs]
        del vs
        sc = T5.eval_models(models, verbose=False)
        add = {f'score|{s}|M1|{n}|{seed}': sc[s][j] for j, n in enumerate(names) for s in T5.ALL}
        T5.save_bank5(add)
        bank5.update(add)
        print(f'seed {seed} WiSE 대조 {len(names)}개 {(time.time() - t0) / 60:.1f}분', flush=True)
        del models
        torch.cuda.empty_cache()
    lab = T5.labels()
    print('WiSE 대조 (seed 평균 AUC): b | ' + ' | '.join(T5.ALL) + ' | 선택')
    for b in WISE_B:
        n = f'wise|U3s|b{b:.1f}'
        a = [np.mean([T.auc(lab[s], bank5[f'score|{s}|M1|{n}|{sd}']) for sd in SEEDS]) for s in T5.ALL]
        print(f'{b:.1f} | ' + ' | '.join(f'{x:.2f}' for x in a) + f' | {np.mean(a[:2]):.2f}')


def report():
    ch = json.load(open(os.path.join(C.REPORTS, 't5_merge_choice.json')))
    sel = json.load(open(SEL_PATH))
    lab = T5.labels()
    bank, b5 = T.load_bank(), T5.load_bank5()
    base = {s: [bank[f'score|{s}|S4|P3|{sd}|last'] for sd in SEEDS] for s in T5.ALL}
    p0 = {s: [bank[f'score|{s}|P0']] for s in T5.ALL}
    u3s = {s: [bank[f'score|{s}|S9|U3s|{sd}|best'] for sd in SEEDS] for s in T5.ALL}
    pname = ch['best_lin'][1].split('|')[1]
    md = ['# M1: 데이터 전문가 가중치 병합 (자동 생성: code/t5_merge.py)\n',
          '학습 없이 같은 seed의 FF++ 전문가(A)와 SBI 전문가(B = U3s best)의 가중치를 섞은 **단일 모델**(추론 비용 = U3 = P3 + baseline 상단 재계산 없음과 같음).',
          '선택 = FF++ val·DFD-val 평균 AUC(seed 3개 평균). 보고 세트 수치는 seed 3개 평균. Δ = [영상 짝 bootstrap 95% CI], ✔ = 0 밖.\n',
          f'- 선택된 선형 병합: `{ch["best_lin"][1]}` (선택 점수 {ch["best_lin"][0]:.2f}), task vector 최고: `{ch["best_tv"][1]}` ({ch["best_tv"][0]:.2f})',
          '- 비교: P3(last) 선택 점수 84.00, P0 83.94\n',
          '## 1. 선택 세트 격자 (seed 평균 AUC)\n', '| 쌍 | ' + ' | '.join(f't={t:.1f}' for t in TS) + ' |', '|---|' + '---|' * len(TS)]
    for pn in PAIRS:
        md.append(f'| {pn} | ' + ' | '.join(f'{np.mean([np.mean(list(sel[str(s)][f"lin|{pn}|t{t:.1f}"].values())) for s in SEEDS]):.2f}' for t in TS) + ' |')
    md += ['', f'## 2. 보고 세트: {pname} 선형 병합 t 곡선 (AUC, seed 평균) — t는 선택 세트로만 고름\n',
           '| t | ' + ' | '.join(T5.ALL) + ' |', '|---|' + '---|' * len(T5.ALL)]
    for t in TS:
        n = f'lin|{pname}|t{t:.1f}'
        md.append(f'| {t:.1f}{" ←선택" if n == ch["best_lin"][1] else ""} | ' + ' | '.join(
            f'{np.mean([T.auc(lab[s], b5[f"score|{s}|M1|{n}|{sd}"]) for sd in SEEDS]):.2f}' for s in T5.ALL) + ' |')
    md += ['', '## 3. 선택된 모델의 판정 (보고 세트, seed 평균)\n']
    for n in (ch['best_lin'][1], ch['best_tv'][1]):
        new = {s: [b5[f'score|{s}|M1|{n}|{sd}'] for sd in SEEDS] for s in T5.ALL}
        md += [f'### `{n}`\n', '| 세트 | AUC | ACC@0.5 | bACC@0.5 | ΔAUC vs P3 | ΔACC vs P3 | ΔAUC vs P0 | ΔACC vs P0 |', '|---|---|---|---|---|---|---|---|']
        for s in T.REPORT_SETS:
            a = np.mean([T.auc(lab[s], x) for x in new[s]])
            ac = np.array([T5.acc(lab[s], x) for x in new[s]]).mean(0)
            d3 = T5.boot_cmp(lab[s], new[s], base[s], B=1000)
            d0 = T5.boot_cmp(lab[s], new[s], p0[s], B=1000)
            md.append(f'| {s} | {a:.2f} | {ac[0]:.2f} | {ac[1]:.2f} | {T5.fmt(d3)} | {T5.fmt(d3, "acc")} | {T5.fmt(d0)} | {T5.fmt(d0, "acc")} |')
        md.append('')
    md += ['## 참고: 기준 모델 (seed 평균)\n', '| 모델 | ' + ' | '.join(f'{s} AUC/ACC' for s in T.REPORT_SETS) + ' |', '|---|' + '---|' * 4]
    for name, d in (('P0', p0), ('P3 last', base), ('U3s best', u3s)):
        md.append(f'| {name} | ' + ' | '.join(
            f'{np.mean([T.auc(lab[s], x) for x in d[s]]):.2f} / {np.mean([T5.acc(lab[s], x)[0] for x in d[s]]):.2f}' for s in T.REPORT_SETS) + ' |')
    open(os.path.join(C.REPORTS, 't5_merge.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    {'sel': sel_stage, 'full': full_stage, 'report': report, 'ablate': ablate}[sys.argv[1]]()
