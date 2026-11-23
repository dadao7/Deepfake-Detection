"""⑧ 추론 시간·메모리 측정: 논문 모델(P0) vs 최종 모델(= P3 구조: P0 + 분기). 2026-10-21. 학습 없음.
병합 모델은 P3와 구조가 같아 가중치 값만 다르므로, 시간은 P3 구조로 잰다(가중치 값은 속도에 무관).
입력은 무작위 224×224 프레임(실제 캐시와 같은 형식: float [0,1]), fp16 autocast. 출력: reports/qa_timing.json
"""
import json
import os
import sys

import torch

EXP = r'C:\m2f2\exp'
sys.path.insert(0, os.path.join(EXP, 'code'))
import common as C  # noqa: E402
import expmodel as E  # noqa: E402

torch.backends.cudnn.benchmark = True
dev = 'cuda'
base = C.load_baseline(dev, verbose=False)
branch = E.ExpModel(E.setting_cfg('P3'), seed=0).to(dev).eval()
branch.load_init('sbi')
n_base = sum(p.numel() for p in base.parameters())
n_branch = sum(p.numel() for p in branch.parameters())


@torch.no_grad()
def run_p0(x):
    return C.baseline_forward(base, x, amp=True)


@torch.no_grad()
def run_final(x):
    logit, b3 = C.baseline_forward(base, x, amp=True)
    with torch.autocast('cuda', dtype=torch.float16):
        return branch(x, b3, logit)['pred']


def bench(fn, bs, iters=50, warm=10):
    x = torch.rand(bs, 3, 224, 224, device=dev)
    for _ in range(warm):
        fn(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        fn(x)
    en.record()
    torch.cuda.synchronize()
    ms = st.elapsed_time(en) / iters
    return {'ms_per_batch': ms, 'ms_per_frame': ms / bs, 'peak_mem_MB': torch.cuda.max_memory_allocated() / 2**20}


out = {'gpu': torch.cuda.get_device_name(0), 'params_baseline_M': n_base / 1e6, 'params_branch_M': n_branch / 1e6}
for bs in (1, 32):
    out[f'bs{bs}'] = {'P0': bench(run_p0, bs), '최종(P3 구조)': bench(run_final, bs)}
for bs in (1, 32):
    p0, fi = out[f'bs{bs}']['P0'], out[f'bs{bs}']['최종(P3 구조)']
    out[f'bs{bs}']['overhead_pct'] = (fi['ms_per_frame'] / p0['ms_per_frame'] - 1) * 100
json.dump(out, open(os.path.join(EXP, 'reports', 'qa_timing.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
