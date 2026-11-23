"""TASKS4 T4-1~T4-4: 결과에서 찾은 약점 기반 개선(학습 없이 캐시로 시험)
 ① AdaBN-top: 고정 baseline의 EfficientNet 상단(f7·f8) BN 통계를 대상 데이터(라벨 없음)로 다시 추정, 원 통계와 λ 비율로 섞음
 ② 경계 증거 결합: 고정 seg head(SBI 경계 탐지)의 heatmap 최댓값을 영상 logit에 β 가중으로 더함(β 단위 = FF++ train 표준편차)
 ③ 판정 기준 적응: 라벨 없이 대상 데이터의 영상 logit 분포에 2성분 GMM → 두 성분 사후확률이 같은 점을 기준값으로
 ④ 상보 앙상블: FF++로 학습한 쪽 점수와 SBI로 학습한 쪽 점수의 logit 가중 평균(가중치 w)
선택: FF++ val + DFD-val의 평균 AUC(①②④) / 균형정확도(③)로 하이퍼파라미터를 고른다. 보고 세트(fftest·cdf·fsh·dfdtest)는 고르는 데 쓰지 않는다.
  python code/t4_methods.py frames   (프레임 값 계산·저장 → cache/t4/)
  python code/t4_methods.py eval     (선택·보고 → reports/t4_results.md, t4_results.json)"""
import json
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import basetop as BT  # noqa: E402
import common as C  # noqa: E402
import expmodel as E  # noqa: E402
import t34_common as T  # noqa: E402

OUT = os.path.join(C.CACHE, 't4')
ND = os.path.join(C.CACHE, 'newdata')
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
SETS = ('ffval', 'dfdval', 'fftest', 'cdf', 'fsh', 'dfdtest')


def frame_sets():
    """세트마다 프레임 행·영상 번호·라벨·점수 규칙 (점수 은행과 같은 영상 순서)"""
    out = {}
    for name, split, rule in (('ffval', 'val', 'mean'), ('fftest', 'test', 'ffpp'), ('cdf', 'cdf', 'frac')):
        ix = C.load_index(split)
        V = int(ix['vid'].max()) + 1
        lab = np.zeros(V, int)
        lab[ix['vid']] = ix['label']
        out[name] = {'src': ('base', split), 'rows': np.arange(len(ix['vid'])), 'vid': ix['vid'], 'V': V, 'label': lab,
                     'rule': rule}
    split = T.dfd_split()
    for ds in ('fsh', 'dfd'):
        z = np.load(os.path.join(ND, f'{ds}_meta.npz'))
        vid, keys, vl = z['vid'], z['video_key'], z['video_label']
        ok = np.bincount(vid, minlength=len(vl)) >= 8
        masks = {'fsh': ok} if ds == 'fsh' else {
            'dfdval': ok & np.isin(keys, split['val']), 'dfdtest': ok & np.isin(keys, split['test'])}
        for name, m in masks.items():
            pos = np.full(len(vl), -1)
            pos[np.where(m)[0]] = np.arange(m.sum())
            rows = np.where(pos[vid] >= 0)[0]
            out[name] = {'src': ('newdata', ds), 'rows': rows, 'vid': pos[vid[rows]], 'V': int(m.sum()), 'label': vl[m],
                         'rule': 'frac'}
    return out


def arrays(src):
    kind, name = src
    d = C.BASE_DIR if kind == 'base' else ND
    return (np.load(os.path.join(d, f'{name}_b3.npy'), mmap_mode='r'), np.load(os.path.join(d, f'{name}_cls.npy'), mmap_mode='r'),
            np.load(os.path.join(d, f'{name}_logit.npy'), mmap_mode='r'))


def video_scores(vals, vid, V, rule):
    order = np.argsort(vid, kind='stable')
    cnt = np.bincount(vid, minlength=V)
    st = np.r_[0, np.cumsum(cnt)[:-1]]
    vs = vals[order]
    out = np.zeros(V)
    for v in range(V):
        q = vs[st[v]:st[v] + cnt[v]]
        t = 0 if rule == 'mean' else (10 if rule == 'ffpp' else int(0.1 * len(q)))
        out[v] = C.trimmed_mean(q, t)
    return out


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


# ── ① AdaBN-top ─────────────────────────────────────────────────────
def bn_layers(top):
    return [m for m in list(top.f7.modules()) + list(top.f8.modules()) if isinstance(m, torch.nn.BatchNorm2d)]


@torch.no_grad()
def target_bn_stats(top, b3, rows):
    """대상 프레임으로 f7·f8 BN 통계(누적 평균) 추정 — 라벨 사용 안 함. 원 통계는 그대로 두고 새 통계만 반환"""
    import copy
    t = copy.deepcopy(top)
    bns = bn_layers(t)
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    for m in t.modules():                         # BN만 학습 모드(통계 갱신), 나머지는 eval
        m.training = isinstance(m, torch.nn.BatchNorm2d)
    for s in range(0, len(rows), 512):
        r = np.sort(rows[s:s + 512])
        with torch.autocast('cuda', dtype=torch.float16):
            t.f8(t.f7(torch.from_numpy(np.ascontiguousarray(b3[r])).cuda()))
    return [(m.running_mean.clone(), m.running_var.clone()) for m in bns]


@torch.no_grad()
def top_logits(top, b3, cls, rows):
    out = np.zeros(len(rows), np.float32)
    for s in range(0, len(rows), 512):
        r = rows[s:s + 512]
        with torch.autocast('cuda', dtype=torch.float16):
            lg = top(torch.from_numpy(np.ascontiguousarray(cls[r])).cuda(), torch.from_numpy(np.ascontiguousarray(b3[r])).cuda())
        lg = lg.float().cpu().numpy()
        out[s:s + len(r)] = lg[:, 1] - lg[:, 0]
    return out


# ── ② 경계 증거 ─────────────────────────────────────────────────────
@torch.no_grad()
def seg_evidence(head, b3, rows):
    out = np.zeros((len(rows), 2), np.float32)
    for s in range(0, len(rows), 1024):
        r = rows[s:s + 1024]
        with torch.autocast('cuda', dtype=torch.float16):
            hm = head(torch.from_numpy(np.ascontiguousarray(b3[r])).cuda()).float().flatten(1)
        top3 = hm.topk(3, dim=1).values
        out[s:s + len(r), 0] = hm.max(1).values.cpu().numpy()
        out[s:s + len(r), 1] = top3.mean(1).cpu().numpy()
    return out


def frames_stage():
    os.makedirs(OUT, exist_ok=True)
    torch.backends.cudnn.deterministic = True
    fs = frame_sets()
    top = BT.BaselineTop()
    top.load_author()
    top = top.cuda().eval()
    src_stats = [(m.running_mean.clone(), m.running_var.clone()) for m in bn_layers(top)]
    head = E.HeatmapHead(272, 64).cuda().eval()
    head.load_state_dict(torch.load(os.path.join(C.RUNS, 'sbi_pretrain', 'seg_head.pt'), map_location='cpu',
                                    weights_only=False)['heatmap_head'])
    # 경계 증거 정규화용 원(FF++ train) 통계
    tb3 = np.load(os.path.join(C.BASE_DIR, 'train_b3.npy'), mmap_mode='r')
    ev_tr = seg_evidence(head, tb3, np.arange(len(tb3)))
    np.save(os.path.join(OUT, 'seg_train_stats.npy'), np.stack([ev_tr.mean(0), ev_tr.std(0)]))
    for name, f in fs.items():
        b3, cls, lg = arrays(f['src'])
        rows = f['rows']
        res = {'p0_logit': (np.asarray(lg[rows, 1], np.float64) - np.asarray(lg[rows, 0], np.float64)).astype(np.float32),
               'seg': seg_evidence(head, b3, rows)}
        tgt = target_bn_stats(top, b3, rows)
        for lam in LAMS:
            for m, (sm, sv), (tm, tv) in zip(bn_layers(top), src_stats, tgt):
                m.running_mean.copy_((1 - lam) * sm + lam * tm)
                m.running_var.copy_((1 - lam) * sv + lam * tv)
            res[f'adabn_{lam}'] = top_logits(top, b3, cls, rows)
        for m, (sm, sv) in zip(bn_layers(top), src_stats):
            m.running_mean.copy_(sm)
            m.running_var.copy_(sv)
        # 확인: λ=0 재구성 logit == 캐시 logit
        res['check_lam0_maxdiff'] = np.array([np.abs(res['adabn_0.0'] - res['p0_logit']).max()])
        np.savez(os.path.join(OUT, f'{name}.npz'), **res, vid=f['vid'], label=f['label'], V=f['V'], rule=f['rule'])
        print(f'{name}: 프레임 {len(rows)}, 영상 {f["V"]}, λ=0 재구성 최대 차 {res["check_lam0_maxdiff"][0]:.2e}', flush=True)


# ── 평가 ────────────────────────────────────────────────────────────
def load_frames():
    out = {}
    for s in SETS:
        z = np.load(os.path.join(OUT, f'{s}.npz'))
        out[s] = {k: z[k] for k in z.files}
        out[s]['V'] = int(out[s]['V'])
        out[s]['rule'] = str(out[s]['rule'])
    return out


def vlogit(F, frame_logit):
    """프레임 logit → 프레임 확률 → 영상 점수(규칙) → 영상 logit"""
    p = 1 / (1 + np.exp(-np.asarray(frame_logit, np.float64)))
    return logit(video_scores(p, F['vid'], F['V'], F['rule']))


def gmm_threshold(x, iters=200):
    """1차원 2성분 GMM(EM, 원소별 연산만) → 두 성분 사후확률이 같은 점(두 평균 사이)"""
    x = np.asarray(x, np.float64)
    mu = np.quantile(x, [0.25, 0.75])
    sd = np.array([x.std(), x.std()]) / 2 + 1e-3
    pi = np.array([0.5, 0.5])
    for _ in range(iters):
        ll = [np.log(pi[k]) - np.log(sd[k]) - 0.5 * ((x - mu[k]) / sd[k]) ** 2 for k in range(2)]
        m = np.maximum(ll[0], ll[1])
        r1 = np.exp(ll[1] - m) / (np.exp(ll[0] - m) + np.exp(ll[1] - m))
        r = [1 - r1, r1]
        for k in range(2):
            w = r[k].sum() + 1e-9
            pi[k] = w / len(x)
            mu[k] = (r[k] * x).sum() / w
            sd[k] = np.sqrt((r[k] * (x - mu[k]) ** 2).sum() / w) + 1e-3
    lo, hi = sorted(mu)
    grid = np.linspace(lo, hi, 2001)
    ll = [np.log(pi[k]) - np.log(sd[k]) - 0.5 * ((grid - mu[k]) / sd[k]) ** 2 for k in range(2)]
    hi_k = int(np.argmax(mu))
    diff = ll[hi_k] - ll[1 - hi_k]
    i = int(np.argmax(diff > 0)) if (diff > 0).any() else len(grid) // 2
    return float(grid[i])


def bacc(lab, L, thr):
    pred = L >= thr
    tpr = pred[lab == 1].mean()
    tnr = (~pred[lab == 0]).mean()
    return (tpr + tnr) / 2 * 100, (1 - tnr) * 100, (1 - tpr) * 100


def eval_stage():
    Fr = load_frames()
    bank = T.load_bank()
    seg_mu, seg_sd = np.load(os.path.join(OUT, 'seg_train_stats.npy'))
    res = {'checks': {}, 'adabn': {}, 'seg': {}, 'thr': {}, 'ens': {}}
    # 확인: P0 영상 점수 == 점수 은행
    for s in SETS:
        v = video_scores(1 / (1 + np.exp(-Fr[s]['p0_logit'].astype(np.float64))), Fr[s]['vid'], Fr[s]['V'], Fr[s]['rule'])
        res['checks'][s] = float(np.abs(v - bank[f'score|{s}|P0']).max())
    base = {s: vlogit(Fr[s], Fr[s]['p0_logit']) for s in SETS}

    def report(name, cand):
        """cand: {set: 영상 logit} → 세트별 AUC, P0 대비 Δ와 bootstrap CI"""
        out = {}
        for s in SETS:
            lab = Fr[s]['label']
            a = T.auc(lab, cand[s])
            d, lo, hi = T.boot_delta(lab, [cand[s]], base[s], B=1000) if s in T.REPORT_SETS else (a - T.auc(lab, base[s]), np.nan, np.nan)
            out[s] = {'auc': a, 'delta': d, 'lo': lo, 'hi': hi}
        return out
    # ① AdaBN-top: λ 선택(선택 세트 평균 AUC)
    sel = {lam: np.mean([T.auc(Fr[s]['label'], vlogit(Fr[s], Fr[s][f'adabn_{lam}'])) for s in T.SELECT_SETS]) for lam in LAMS}
    lam_star = max(sel, key=sel.get)
    res['adabn'] = {'select': sel, 'lambda': lam_star,
                    'all_lambdas': {lam: {s: T.auc(Fr[s]['label'], vlogit(Fr[s], Fr[s][f'adabn_{lam}'])) for s in SETS} for lam in LAMS},
                    'report': report('adabn', {s: vlogit(Fr[s], Fr[s][f'adabn_{lam_star}']) for s in SETS})}
    print('AdaBN λ 선택', {k: round(v, 3) for k, v in sel.items()}, '→', lam_star, flush=True)
    # ② 경계 증거: 지표(최댓값/상위3 평균)·β 선택
    BETAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
    best = None
    sel2 = {}
    for j, nm in enumerate(('max', 'top3')):
        ev = {s: video_scores(Fr[s]['seg'][:, j].astype(np.float64), Fr[s]['vid'], Fr[s]['V'], Fr[s]['rule']) for s in SETS}
        ev = {s: (ev[s] - seg_mu[j]) / seg_sd[j] for s in SETS}
        res['seg'][f'evidence_alone_auc_{nm}'] = {s: T.auc(Fr[s]['label'], ev[s]) for s in SETS}
        for b in BETAS:
            sc = np.mean([T.auc(Fr[s]['label'], base[s] + b * ev[s]) for s in T.SELECT_SETS])
            sel2[f'{nm}|{b}'] = sc
            if best is None or sc > best[0]:
                best = (sc, nm, b, ev)
    res['seg'].update({'select': sel2, 'choice': f'{best[1]}|{best[2]}',
                       'report': report('seg', {s: base[s] + best[2] * best[3][s] for s in SETS})})
    print('경계 증거 선택 →', best[1], best[2], flush=True)
    # ③ 판정 기준 적응: 고정 0.5 vs GMM (P0 영상 logit)
    for s in SETS:
        lab, L = Fr[s]['label'], base[s]
        thr = gmm_threshold(L)
        b0, f0, n0 = bacc(lab, L, 0.0)
        b1, f1, n1 = bacc(lab, L, thr)
        res['thr'][s] = {'fixed05': {'bacc': b0, 'fpr': f0, 'fnr': n0}, 'gmm': {'thr_logit': thr, 'bacc': b1, 'fpr': f1, 'fnr': n1}}
    # ④ 상보 앙상블: (FF++ 쪽, SBI 쪽) 쌍, seed별로 짝지어 w 선택
    pairs = [('P0', 'S9|U0s|{s}|best'), ('P0', 'S9|U0s|{s}|last'), ('S4|P3|{s}|last', 'S9|U0s|{s}|last'),
             ('P0', 'S9|U3s|{s}|best'), ('S4|P3|{s}|last', 'S9|U3s|{s}|best'), ('P0', 'S8|P3_sbi|{s}|best')]
    W = np.round(np.arange(0, 1.01, 0.1), 2)
    for a, b in pairs:
        def comb(s, seed, w):
            ka = a.format(s=seed)
            kb = b.format(s=seed)
            return (1 - w) * logit(bank[f'score|{s}|{ka}']) + w * logit(bank[f'score|{s}|{kb}'])
        sc = {float(w): np.mean([T.auc(bank[f'label|{s}'], comb(s, seed, w)) for s in T.SELECT_SETS for seed in (0, 1, 2)])
              for w in W}
        w_star = max(sc, key=sc.get)
        rep = {}
        for s in SETS:
            lab = bank[f'label|{s}']
            cands = [comb(s, seed, w_star) for seed in (0, 1, 2)]
            b0 = logit(bank[f'score|{s}|P0'])
            if s in T.REPORT_SETS:
                d, lo, hi = T.boot_delta(lab, cands, b0, B=1000)
            else:
                d, lo, hi = np.mean([T.auc(lab, c) for c in cands]) - T.auc(lab, b0), np.nan, np.nan
            rep[s] = {'auc': float(np.mean([T.auc(lab, c) for c in cands])), 'delta': d, 'lo': lo, 'hi': hi}
        res['ens'][f'{a} + {b}'] = {'select': sc, 'w': w_star, 'report': rep}
        print('앙상블', a, b, 'w*', w_star, flush=True)
    json.dump(res, open(os.path.join(C.REPORTS, 't4_results.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False,
              default=float)
    write_md(res, Fr)


def write_md(res, Fr):
    f = lambda r: f'{r["auc"]:.2f} ({r["delta"]:+.2f}' + (f' [{r["lo"]:+.2f}, {r["hi"]:+.2f}]{" ✔" if (r["lo"] > 0 or r["hi"] < 0) else ""})' if np.isfinite(r['lo']) else ')')  # noqa: E731
    md = ['# T4 개선 방법 시험 결과 (자동 생성: code/t4_methods.py)\n',
          '선택 세트 = FF++ val + DFD-val (하이퍼파라미터 고르기), 보고 세트 = FF++ test · Celeb-DF · FaceShifter · DFD-test.',
          '표: AUC (P0 대비 Δ [영상 bootstrap 95% CI], ✔ = CI가 0 밖)\n',
          f'점수 은행 대조(P0 영상 점수 최대 차): {json.dumps({k: round(v, 8) for k, v in res["checks"].items()})}\n',
          '| 방법 | 선택값 | ' + ' | '.join(SETS) + ' |', '|---|---|' + '---|' * len(SETS)]
    md.append('| P0 | – | ' + ' | '.join(f'{T.auc(Fr[s]["label"], vlogit(Fr[s], Fr[s]["p0_logit"])):.2f}' for s in SETS) + ' |')
    md.append(f'| ① AdaBN-top | λ={res["adabn"]["lambda"]} | ' + ' | '.join(f(res['adabn']['report'][s]) for s in SETS) + ' |')
    md.append(f'| ② 경계 증거 결합 | {res["seg"]["choice"]} | ' + ' | '.join(f(res['seg']['report'][s]) for s in SETS) + ' |')
    for k, v in res['ens'].items():
        md.append(f'| ④ 앙상블 {k} | w={v["w"]} | ' + ' | '.join(f(v['report'][s]) for s in SETS) + ' |')
    md += ['\n## ① AdaBN-top: λ별 AUC\n', '| λ | ' + ' | '.join(SETS) + ' |', '|---|' + '---|' * len(SETS)]
    for lam, d in res['adabn']['all_lambdas'].items():
        md.append(f'| {lam} | ' + ' | '.join(f'{d[s]:.2f}' for s in SETS) + ' |')
    md += ['\n## ② 경계 증거 단독 AUC (seg head heatmap, FF++ train 기준 정규화)\n', '| 지표 | ' + ' | '.join(SETS) + ' |',
           '|---|' + '---|' * len(SETS)]
    for nm in ('max', 'top3'):
        d = res['seg'][f'evidence_alone_auc_{nm}']
        md.append(f'| {nm} | ' + ' | '.join(f'{d[s]:.2f}' for s in SETS) + ' |')
    md += ['\n## ③ 판정 기준 적응 (P0, 라벨 없는 2성분 GMM)\n', '| 세트 | 고정 0.5: 균형정확도 / 오탐 / 미탐 | GMM 기준: 균형정확도 / 오탐 / 미탐 (기준 logit) |',
           '|---|---|---|']
    for s in SETS:
        a, b = res['thr'][s]['fixed05'], res['thr'][s]['gmm']
        md.append(f'| {s} | {a["bacc"]:.1f} / {a["fpr"]:.1f} / {a["fnr"]:.1f} | {b["bacc"]:.1f} / {b["fpr"]:.1f} / {b["fnr"]:.1f} ({b["thr_logit"]:+.2f}) |')
    open(os.path.join(C.REPORTS, 't4_results.md'), 'w', encoding='utf-8').write('\n'.join(md))
    print('\n'.join(md))


if __name__ == '__main__':
    frames_stage() if sys.argv[1] == 'frames' else eval_stage()
