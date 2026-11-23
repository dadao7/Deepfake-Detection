"""U0 vs U3: baseline 상단이 저자 가중치에서 얼마나 멀어졌나(1) + U3의 동적 α(2). 읽기 전용."""
import json
import os
import sys

import numpy as np
import torch

EXP = r'C:\m2f2\exp'
sys.path.insert(0, os.path.join(EXP, 'code'))
from basetop import BaselineTop  # noqa: E402

top0 = BaselineTop()
print(top0.load_author())
theta0 = {k: v.detach().float().clone() for k, v in top0.named_parameters()}
GROUPS = ['f7', 'f8', 'deepfake_proj', 'vision_proj', 'clip_vision_alpha', 'output']


def drift(sd):
    tot_d = tot_0 = 0.0
    per = {g: [0.0, 0.0] for g in GROUPS}
    for k, p0 in theta0.items():
        p = sd['top.' + k].float()
        d = float(((p - p0) ** 2).sum())
        n = float((p0 ** 2).sum())
        tot_d += d
        tot_0 += n
        g = k.split('.')[0]
        per[g][0] += d
        per[g][1] += n
    rel = {g: (v[0] ** 0.5) / max(v[1] ** 0.5, 1e-12) * 100 for g, v in per.items()}
    return (tot_d ** 0.5) / (tot_0 ** 0.5) * 100, (tot_d ** 0.5), rel


out = {}
for data in ['s', 'f', 'm']:
    for arch in ['U0', 'U3']:
        for ck in ['best', 'last']:
            for s in range(3):
                rid = f'{arch}{data}_s{s}'
                c = torch.load(os.path.join(EXP, 'runs', rid, f'ckpt_{ck}.pt'), map_location='cpu', weights_only=False)
                rel, absn, per = drift(c['model'])
                pr = np.load(os.path.join(EXP, 'runs', rid, f'pred_{ck}.npz'))
                row = {'epoch': c['epoch'], 'rel_drift_pct': rel, 'abs_drift': absn, 'per_group_pct': per}
                if arch == 'U3':
                    for ds in ['ffpp', 'cdf']:
                        row[f'alpha_{ds}'] = float(pr[f'{ds}_alpha'].mean())
                out[f'{rid}_{ck}'] = row
                print(rid, ck, 'ep', c['epoch'], f'drift {rel:.3f}%', {g: round(v, 3) for g, v in per.items()},
                      {k: round(v, 3) for k, v in row.items() if k.startswith('alpha')})

# P3(FF++, 상단 고정) α 참고
for ck in ['best', 'last']:
    for s in range(3):
        pr = np.load(os.path.join(EXP, 'runs', f'P3_d100_s{s}', f'pred_{ck}.npz'))
        out[f'P3_s{s}_{ck}'] = {f'alpha_{ds}': float(pr[f'{ds}_alpha'].mean()) for ds in ['ffpp', 'cdf']}
        print('P3', s, ck, out[f'P3_s{s}_{ck}'])

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'top_drift.json'), 'w'), indent=1)
