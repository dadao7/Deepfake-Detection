"""TASKS4 T3(기존 기능 최고 조합) + T4-4b(전문가 앙상블 확장). 점수 은행만 사용.
선택 점수 = AUC(FF++ val)와 AUC(DFD-val)의 평균. 보고 세트(fftest·cdf·fsh·dfdtest)는 고르는 데 쓰지 않는다.
 - 단일 설정: (단계, 설정, 체크포인트)별 seed 평균 AUC로 순위 → 상위 설정을 보고 세트에서 P0 대비 bootstrap
 - 앙상블: seed 3개 logit 평균을 한 '전문가'로 보고, 전문가 2·3개의 logit 가중 평균(가중치 0.1 간격 격자)
출력: reports/t3_best_combo.md (T3), reports/t4_ensemble.md (T4-4b), reports/t34_select.json"""
import itertools
import json
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import t34_common as T  # noqa: E402

ALL = T.SELECT_SETS + T.REPORT_SETS
NAMES = {'P1p': "P1′"}


def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


def groups(bank):
    g = {}
    for k in T.model_keys(bank):
        if k == 'P0':
            continue
        st, setting, seed, ck = k.split('|')
        g.setdefault(f'{st}|{setting}|{ck}', []).append(k)
    return g


def label(gname):
    st, setting, ck = gname.split('|')
    return f'{NAMES.get(setting, setting)} ({ck}, {st})'


def main():
    bank = T.load_bank()
    lab = {s: bank[f'label|{s}'] for s in ALL}
    p0 = {s: logit(bank[f'score|{s}|P0']) for s in ALL}
    G = groups(bank)
    sel_p0 = np.mean([T.auc(lab[s], p0[s]) for s in T.SELECT_SETS])
    # ── T3: 단일 설정 ──
    singles = []
    for gname, keys in G.items():
        per = {s: [T.auc(lab[s], logit(bank[f'score|{s}|{k}'])) for k in keys] for s in ALL}
        singles.append({'group': gname, 'n_seed': len(keys), 'select': float(np.mean([np.mean(per[s]) for s in T.SELECT_SETS])),
                        'auc': {s: float(np.mean(per[s])) for s in ALL}})
    singles.sort(key=lambda r: -r['select'])
    for r in singles[:6]:
        keys = G[r['group']]
        r['boot'] = {s: T.boot_delta(lab[s], [logit(bank[f'score|{s}|{k}']) for k in keys], p0[s], B=1000) for s in T.REPORT_SETS}
    # ── 전문가(seed 평균) ──
    experts = {}
    for gname, keys in G.items():
        if len(keys) == 3:
            experts[gname] = {s: np.mean([logit(bank[f'score|{s}|{k}']) for k in keys], 0) for s in ALL}
    experts['P0'] = p0
    cands = ['P0', 'S4|P3|last', 'S4|P3|best', 'S4|P2|last', 'S4|P2a|best', 'S9|U0f|best', 'S9|U0s|best', 'S9|U0s|last',
             'S9|U3s|best', 'S9|U3s|last', 'S8|P3_sbi|best', 'S9|U3f|last']
    cands = [c for c in cands if c in experts]
    W = np.round(np.arange(0, 1.01, 0.1), 2)
    ens = []

    def score(combo, w):
        return {s: sum(wi * experts[c][s] for c, wi in zip(combo, w)) for s in ALL}
    for r in (2, 3):
        for combo in itertools.combinations(cands, r):
            for w in itertools.product(W, repeat=r):
                if abs(sum(w) - 1) > 1e-6 or min(w) == 0:
                    continue
                sc = score(combo, w)
                ens.append((float(np.mean([T.auc(lab[s], sc[s]) for s in T.SELECT_SETS])), combo, w))
    ens.sort(key=lambda t: -t[0])
    top = []
    seen = set()
    for selv, combo, w in ens:
        if combo in seen:
            continue
        seen.add(combo)
        sc = score(combo, w)
        top.append({'combo': combo, 'w': [float(x) for x in w], 'select': selv, 'auc': {s: T.auc(lab[s], sc[s]) for s in ALL},
                    'boot': {s: T.boot_delta(lab[s], [sc[s]], p0[s], B=1000) for s in T.REPORT_SETS}})
        if len(top) == 6:
            break
    # 단일 전문가(seed 평균) 기준선도 같은 표에
    for c in ('S4|P3|last', 'S9|U3s|best'):
        sc = experts[c]
        top.append({'combo': (c,), 'w': [1.0], 'select': float(np.mean([T.auc(lab[s], sc[s]) for s in T.SELECT_SETS])),
                    'auc': {s: T.auc(lab[s], sc[s]) for s in ALL},
                    'boot': {s: T.boot_delta(lab[s], [sc[s]], p0[s], B=1000) for s in T.REPORT_SETS}, 'note': 'seed 평균 단독'})
    json.dump({'p0_select': sel_p0, 'singles': singles, 'ensembles_top': top}, open(os.path.join(C.REPORTS, 't34_select.json'), 'w',
              encoding='utf-8'), indent=1, ensure_ascii=False, default=lambda o: list(o) if isinstance(o, tuple) else float(o))
    fmtb = lambda b: f'{b[0]:+.2f} [{b[1]:+.2f}, {b[2]:+.2f}]{" ✔" if (b[1] > 0 or b[2] < 0) else ""}'  # noqa: E731
    p0auc = {s: T.auc(lab[s], p0[s]) for s in ALL}
    md = ['# T3: 지금까지 기능 중 최고 조합 (자동 생성: code/t34_select.py)\n',
          '선택 = FF++ val·DFD-val 평균 AUC(seed 평균)로만 순위. 보고 세트는 선택에 쓰지 않음. Δ = P0 대비, [영상 bootstrap 95% CI], ✔ = 0 밖.\n',
          f'P0: 선택 점수 {sel_p0:.2f} | ' + ' | '.join(f'{s} {p0auc[s]:.2f}' for s in ALL) + '\n',
          '## 단일 설정 순위 (상위 15, 선택 점수 기준)\n', '| 순위 | 설정 | 선택 점수 | ' + ' | '.join(ALL) + ' |',
          '|---|---|---|' + '---|' * len(ALL)]
    for i, r in enumerate(singles[:15], 1):
        md.append(f'| {i} | {label(r["group"])} | {r["select"]:.2f} | ' + ' | '.join(f'{r["auc"][s]:.2f}' for s in ALL) + ' |')
    md += ['\n### 상위 6개 단일 설정의 보고 세트 판정 (P0 대비)\n', '| 설정 | ' + ' | '.join(T.REPORT_SETS) + ' |',
           '|---|' + '---|' * len(T.REPORT_SETS)]
    for r in singles[:6]:
        md.append(f'| {label(r["group"])} | ' + ' | '.join(fmtb(r['boot'][s]) for s in T.REPORT_SETS) + ' |')
    open(os.path.join(C.REPORTS, 't3_best_combo.md'), 'w', encoding='utf-8').write('\n'.join(md))
    md2 = ['# T4-4b: 전문가 앙상블 (seed 평균 전문가 2·3개 logit 가중 평균, 자동 생성: code/t34_select.py)\n',
           '선택 = FF++ val·DFD-val 평균 AUC. 상위 6개 조합 + 단독 기준선. Δ = P0 대비 [bootstrap 95% CI].\n',
           '| 조합 (가중치) | 선택 점수 | ' + ' | '.join(ALL) + ' | ' + ' | '.join(f'Δ {s}' for s in T.REPORT_SETS) + ' |',
           '|---|---|' + '---|' * (len(ALL) + len(T.REPORT_SETS))]
    for r in top:
        name = ' + '.join(f'{label(c) if c != "P0" else "P0"}×{w:.1f}' for c, w in zip(r['combo'], r['w']))
        md2.append(f'| {name}{" (" + r["note"] + ")" if "note" in r else ""} | {r["select"]:.2f} | ' +
                   ' | '.join(f'{r["auc"][s]:.2f}' for s in ALL) + ' | ' + ' | '.join(fmtb(r['boot'][s]) for s in T.REPORT_SETS) + ' |')
    open(os.path.join(C.REPORTS, 't4_ensemble.md'), 'w', encoding='utf-8').write('\n'.join(md2))
    print('\n'.join(md))
    print('\n'.join(md2))


if __name__ == '__main__':
    main()
