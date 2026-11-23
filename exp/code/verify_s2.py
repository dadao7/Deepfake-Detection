"""S2 게이트 검증. 결과: logs/verify_s2_{part}.json
  part=unit : ① P3(ExpModel+캐시) == 원본 M2F2DetImproved(실시간 baseline)  ② crop 벡터화 == 원본 crop_by_heatmap_topk
              ③ detach 제거 시 heatmap head에 gradient  + 색 변형·Method 4 원본 일치, 캐시 == 실시간 baseline, heatmap BN 동작
  part=repro: ④ 같은 seed 두 번 학습(짧게) → 가중치·곡선 비트 단위 비교, 결정적 알고리즘 검사"""
import json
import os
import random
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import expmodel as E  # noqa: E402

from topk_crop import crop_by_heatmap_topk  # noqa: E402  (원본, 읽기 전용)

R = {}


def maxdiff(a, b):
    return float((a.float() - b.float()).abs().max())


class FixedMap(nn.Module):
    """원본 crop 함수에 정해 둔 heatmap logit을 넣기 위한 시험용 head"""
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m


def test_crops():
    torch.manual_seed(0)
    imgs = torch.rand(121, 3, 224, 224, device='cuda')
    # 121개 위치 각각에 peak를 하나씩 + 다른 두 곳(가까운 것 포함)에 작은 peak → k=3 NMS 동작까지 비교
    hm = torch.full((121, 1, 11, 11), -5.0, device='cuda')
    for i in range(121):
        y, x = divmod(i, 11)
        hm[i, 0, y, x] = 3.0
        hm[i, 0, (y + 5) % 11, (x + 7) % 11] = 1.0
        hm[i, 0, (y + 1) % 11, x] = 2.0        # 인접 → NMS에 억제됨
    a_crops, a_hm, a_pv = crop_by_heatmap_topk(imgs, FixedMap(hm), None, k=3, crop_size=64, min_distance=2)
    from topk_crop import nms_heatmap
    peaks, pv = nms_heatmap(torch.sigmoid(hm), k=3, min_distance=2)
    y1, x1 = E.peaks_to_boxes(peaks, crop=64)
    b_crops = E.crop_boxes(imgs, y1, x1, crop=64)
    R['gate2_crop_equal_bitwise'] = bool(torch.equal(a_crops, b_crops))
    R['gate2_peakvals_equal'] = bool(torch.equal(a_pv, pv))
    R['gate2_box_y1_range'] = [int(y1.min()), int(y1.max())]
    # 무작위 b3 + 실제 HeatmapHead로도 비교
    head = E.HeatmapHead(272, 64).cuda().eval()
    b3 = torch.randn(64, 272, 11, 11, device='cuda')
    with torch.no_grad():
        a_crops, _, a_pv = crop_by_heatmap_topk(imgs[:64], head, b3, k=3, crop_size=64, min_distance=2)
        peaks, pv = nms_heatmap(torch.sigmoid(head(b3)), k=3, min_distance=2)
        b_crops = E.crop_boxes(imgs[:64], *E.peaks_to_boxes(peaks, crop=64), crop=64)
    R['gate2_crop_equal_random_b3'] = bool(torch.equal(a_crops, b_crops))


def test_jitter_m4():
    torch.manual_seed(1)
    x = torch.rand(8, 3, 224, 224, device='cuda')
    rng = np.random.default_rng(0)
    bf, cf, sf = rng.uniform(0.8, 1.2, 8), rng.uniform(0.8, 1.2, 8), rng.uniform(0.9, 1.1, 8)
    mine = E.color_jitter(x, bf, cf, sf)
    ref = torch.stack([TF.adjust_saturation(TF.adjust_contrast(TF.adjust_brightness(x[i], float(bf[i])),
                                                               float(cf[i])), float(sf[i])) for i in range(8)])
    R['jitter_maxdiff_vs_torchvision'] = maxdiff(mine, ref)
    R['jitter_equal_bitwise'] = bool(torch.equal(mine, ref))
    # Method 4: 원본 improved_model.forward 안의 블록과 같은 결과인지 (같은 seed)
    import torch.nn.functional as F
    import torchvision.transforms.functional as F_t
    crops = torch.rand(16, 3, 3, 64, 64, device='cuda')
    eq = []
    for s in range(20):
        random.seed(s)
        torch.manual_seed(s)
        a = E.method4(crops.clone())
        random.seed(s)
        torch.manual_seed(s)
        # 원본 코드 블록(improved_model.py 75~99행)을 그대로 옮긴 참조 구현
        B, K, C_im, H, W = crops.shape
        f = crops.clone().view(B * K, C_im, H, W)
        if random.random() < 0.5:
            ks = random.choice([3, 5])
            sg = random.uniform(0.1, 2.0)
            f = F_t.gaussian_blur(f, kernel_size=[ks, ks], sigma=[sg, sg])
        if random.random() < 0.5:
            ns = random.uniform(0.01, 0.04)
            f = torch.clamp(f + torch.randn_like(f) * ns, 0.0, 1.0)
        if random.random() < 0.3:
            f = F.dropout(f, p=0.1, training=True)
        eq.append(torch.equal(a, f.view(B, K, C_im, H, W)))
    R['method4_equal_20seeds'] = int(sum(eq))


def test_equivalence():
    """① 같은 입력(val 캐시 0..31행)에서 원본 M2F2DetImproved vs ExpModel(P3)"""
    from improved_model import M2F2DetImproved
    VA_img = np.load(os.path.join(C.FRAME_DIR, 'val_img.npy'), mmap_mode='r')
    VA_b3 = np.load(os.path.join(C.BASE_DIR, 'val_b3.npy'), mmap_mode='r')
    VA_lg = np.load(os.path.join(C.BASE_DIR, 'val_logit.npy'), mmap_mode='r')
    rows = np.arange(32)
    x = torch.from_numpy(np.ascontiguousarray(VA_img[rows])).cuda().permute(0, 3, 1, 2).float() / 255.
    b3c = torch.from_numpy(np.ascontiguousarray(VA_b3[rows])).cuda()
    lgc = torch.from_numpy(np.ascontiguousarray(VA_lg[rows])).cuda()

    base = C.load_baseline('cuda', verbose=False)
    # 캐시 == 실시간 baseline (같은 32장 배치, autocast)
    lg_live, b3_live = C.baseline_forward(base, x, amp=True)
    R['cache_vs_live_b3_equal'] = bool(torch.equal(b3_live, b3c))
    R['cache_vs_live_logit_maxdiff'] = maxdiff(lg_live, lgc)

    cfg = E.setting_cfg('P3')
    em = E.ExpModel(cfg, seed=0).cuda()
    R['init_msg'] = em.load_init('sbi1')
    om = M2F2DetImproved(nn.DataParallel(base), k=3, crop_size=64, alpha='dynamic').cuda()
    om.heatmap_head.load_state_dict(em.heatmap_head.state_dict())
    om.detail_branch.load_state_dict(em.detail_branch.state_dict())
    om.gating_network.load_state_dict(em.gating_network.state_dict())
    for blk in base.bridge_adapter:      # 시험 장치: 원본 객체의 list 모듈 dropout을 꺼서 CUDA 난수 소비를 맞춤(출력은 상수라 무관)
        blk.eval()

    # (a) eval, autocast: 원본은 실시간 baseline, ExpModel은 캐시
    om.eval()
    em.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        o = om(x, return_dict=True)
        e = em(x, b3c, lgc)
    for k_o, k_e in (('pred', 'pred'), ('detail_pred', 'detail_pred'), ('heatmap', 'heatmap'),
                     ('dynamic_alpha', 'alpha')):
        R[f'gate1_eval_amp_{k_o}_maxdiff'] = maxdiff(o[k_o], e[k_e])
    R['gate1_eval_amp_pred_equal'] = bool(torch.equal(o['pred'], e['pred']))
    # (b) eval, fp32: ExpModel에 실시간 fp32 baseline 출력을 넣어 모델 논리만 비교
    with torch.no_grad():
        o = om(x, return_dict=True)
        b3_32 = base.block_outputs['b_3']
        e = em(x, b3_32, o['baseline_pred'])
    R['gate1_eval_fp32_pred_maxdiff'] = maxdiff(o['pred'], e['pred'])
    R['gate1_eval_fp32_pred_equal'] = bool(torch.equal(o['pred'], e['pred']))
    # (c) train 모드(heatmap BN 배치 통계·Method 4·dropout), autocast, 같은 난수 상태
    bn_before = {k: v.clone() for k, v in em.heatmap_head.state_dict().items()}
    sd_om_hh = {k: v.clone() for k, v in om.heatmap_head.state_dict().items()}
    diffs, eqs = [], []
    for s in range(5):
        om.train()
        om.baseline.eval()
        em.train()
        om.heatmap_head.load_state_dict(sd_om_hh)
        em.heatmap_head.load_state_dict(sd_om_hh)
        random.seed(s)
        torch.manual_seed(s)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            o = om(x, return_dict=True)
        random.seed(s)
        torch.manual_seed(s)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            e = em(x, b3c, lgc)
        diffs.append(maxdiff(o['pred'], e['pred']))
        eqs.append(bool(torch.equal(o['pred'], e['pred'])))
        R[f'gate1_train_bn_running_var_equal_s{s}'] = bool(torch.equal(
            om.heatmap_head.head[1].running_var, em.heatmap_head.head[1].running_var))
    R['gate1_train_amp_pred_maxdiff'] = max(diffs)
    R['gate1_train_amp_pred_equal_5'] = int(sum(eqs))
    # heatmap BN 원본 동작: train 모드 forward 한 번에 running 통계 변화, 가중치는 그대로
    after = em.heatmap_head.state_dict()
    R['bn_running_var_changed_after_train_forward'] = maxdiff(after['head.1.running_var'], bn_before['head.1.running_var'])
    R['bn_weight_changed'] = maxdiff(after['head.1.weight'], bn_before['head.1.weight'])
    del om, base
    torch.cuda.empty_cache()

    # ③ gradient 도달: P3(detach) vs P4a(no detach), 같은 입력
    for name in ('P3', 'P4a'):
        m = E.ExpModel(E.setting_cfg(name), seed=0).cuda()
        m.load_init('sbi1')
        m.train()
        random.seed(0)
        torch.manual_seed(0)
        with torch.autocast('cuda', dtype=torch.float16):
            out = m(x, b3c, lgc)
            y = torch.from_numpy(C.load_index('val')['label'][rows].astype(np.int64)).cuda()
            loss = torch.nn.functional.cross_entropy(out['pred'].float(), y) + \
                0.15 * torch.nn.functional.cross_entropy(out['detail_pred'].float(), y)
        loss.backward()
        g = [p.grad for p in m.heatmap_head.parameters()]
        R[f'gate3_{name}_heatmap_grad_none'] = all(t is None for t in g)
        R[f'gate3_{name}_heatmap_grad_norm'] = float(sum(t.float().norm() ** 2 for t in g if t is not None) ** 0.5) \
            if any(t is not None for t in g) else 0.0
        R[f'gate3_{name}_detail_grad_norm'] = float(sum(p.grad.float().norm() ** 2 for p in
                                                        m.detail_branch.parameters() if p.grad is not None) ** 0.5)


def test_repro():
    """④ 같은 seed로 짧은 학습(2 epoch × 40스텝)을 두 번 → 가중치·곡선 비교"""
    out = {}
    for tag in ('a', 'b'):
        rid = f'debug_repro_{tag}'
        rdir = os.path.join(C.RUNS, rid)
        if os.path.exists(rdir):
            import shutil
            shutil.rmtree(rdir)             # 디버그 폴더만 지움(실험 결과 아님)
        cmd = [C.PY, '-X', 'utf8', os.path.join(C.EXP, 'code', 'run_train.py'), '--setting', 'P4a', '--data', '100',
               '--seed', '0', '--epochs', '2', '--max_steps', '40', '--no_test', '--kind', 'debug', '--init', 'sbi1',
               '--run_id', rid]
        p = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
        out[tag] = p.stdout[-1500:] + p.stderr[-1500:]
    a = torch.load(os.path.join(C.RUNS, 'debug_repro_a', 'resume.pt'), map_location='cpu', weights_only=False)
    b = torch.load(os.path.join(C.RUNS, 'debug_repro_b', 'resume.pt'), map_location='cpu', weights_only=False)
    neq = [k for k in a['model'] if not torch.equal(a['model'][k], b['model'][k])]
    R['gate4_params_total'] = len(a['model'])
    R['gate4_params_not_equal'] = len(neq)
    R['gate4_first_not_equal'] = neq[:5]
    R['gate4_max_absdiff'] = max([maxdiff(a['model'][k], b['model'][k]) for k in a['model']] or [0.0])
    R['gate4_curves_equal'] = a['curves'] == b['curves'] if not neq else str(
        [(x['val_loss_bal'], y['val_loss_bal']) for x, y in zip(a['curves'], b['curves'])])
    R['gate4_log_tail_a'] = out['a'][-600:]


if __name__ == '__main__':
    part = sys.argv[1] if len(sys.argv) > 1 else 'unit'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if part == 'unit':
        test_crops()
        test_jitter_m4()
        test_equivalence()
    else:
        test_repro()
    json.dump(R, open(os.path.join(C.LOGS, f'verify_s2_{part}.json'), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False, default=str)
    for k, v in R.items():
        print(k, v)
