"""S1-0: 저자 baseline 점검 — bridge_adapter 무작위성의 영향, 재구성 간 동일성, fp16 vs fp32, P0 값 대조, 속도
결과: logs/check_baseline.json"""
import json
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

dev = 'cuda'
te = C.load_index('test')
p0 = np.load(os.path.join(C.EXP, 'ref', 'p0', 'p0_frames.npy'))
p0_idx = json.load(open(os.path.join(C.EXP, 'ref', 'p0', 'p0_index.json')))
# 우리 test 인덱스 순서가 P0과 같은지
same_order = all(a[0] == c and a[1] == v and a[3] == f for a, c, v, f in
                 zip(p0_idx, te['cat'], te['video'], te['frame']))
print('test 인덱스 순서 = P0 순서:', same_order, flush=True)

rows = np.r_[0:8, 14000:8 + 14000, 56000:56016]          # 진짜 8 + DF 8 + F2F 16
imgs = []
for r in rows:
    with h5py.File(C.h5_path(te['cat'][r], 'test'), 'r') as f:
        imgs.append(f[te['video'][r]][te['frame'][r]])
x = torch.from_numpy(np.stack(imgs).astype(np.float32) / 255.).permute(0, 3, 1, 2).to(dev)

res = {'test_order_equals_p0': bool(same_order)}
torch.manual_seed(0)
m1 = C.load_baseline(dev)
with torch.no_grad():
    a32, b32 = C.baseline_forward(m1, x, amp=False)
    a32b, _ = C.baseline_forward(m1, x, amp=False)
    a16, b16 = C.baseline_forward(m1, x, amp=True)
    a16b, b16b = C.baseline_forward(m1, x, amp=True)
# bridge_adapter 출력이 상수인지 직접 확인 (forward hook)
outs = []
h = m1.bridge_adapter_proj.register_forward_hook(lambda m, i, o: outs.append(o.detach().float().cpu()))
with torch.no_grad():
    C.baseline_forward(m1, x, amp=False)
h.remove()
bap = outs[0]
res['bridge_adapter_proj_out_std_over_batch'] = float(bap.std(0).max())
res['same_model_twice_fp32_maxdiff'] = float((a32 - a32b).abs().max())
res['same_model_twice_fp16_maxdiff'] = float((a16 - a16b).abs().max())
res['same_model_twice_fp16_b3_equal'] = bool(torch.equal(b16, b16b))
del m1
torch.cuda.empty_cache()

torch.manual_seed(12345)            # 다른 난수로 재구성 → bridge_adapter·prompt_tokens 초기값이 달라짐
m2 = C.load_baseline(dev, verbose=False)
with torch.no_grad():
    c32, _ = C.baseline_forward(m2, x, amp=False)
    c16, c16b3 = C.baseline_forward(m2, x, amp=True)
res['rebuild_other_seed_fp32_logit_maxdiff'] = float((a32 - c32).abs().max())
res['rebuild_other_seed_fp16_logit_maxdiff'] = float((a16 - c16).abs().max())
res['rebuild_other_seed_fp16_b3_equal'] = bool(torch.equal(b16, c16b3))
p32 = torch.softmax(a32, 1)[:, 1].cpu().numpy()
p16 = torch.softmax(a16, 1)[:, 1].cpu().numpy()
res['fp16_vs_fp32_prob_maxdiff'] = float(np.abs(p16 - p32).max())
res['fp32_vs_p0_prob_maxdiff'] = float(np.abs(p32 - p0[rows]).max())
res['fp16_vs_p0_prob_maxdiff'] = float(np.abs(p16 - p0[rows]).max())
res['b3_shape'] = list(b16.shape)
res['b3_dtype'] = str(b16.dtype)

# 속도 (autocast, 배치 크기별)
big = x.repeat(4, 1, 1, 1)[:64]
for bs in (32, 64):
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(3):
        C.baseline_forward(m2, big[:bs], amp=True)
    torch.cuda.synchronize()
    res[f'sec_per_frame_bs{bs}'] = (time.time() - t) / (3 * bs)
res['gpu_mem_peak_GB'] = torch.cuda.max_memory_allocated() / 1e9
json.dump(res, open(os.path.join(C.LOGS, 'check_baseline.json'), 'w'), indent=1)
for k, v in res.items():
    print(k, v)
