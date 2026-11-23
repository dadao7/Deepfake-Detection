"""학습·검증·테스트·기록 (PROTOCOL §6~9). 실행 1회 = runs/{run_id}/ + registry.csv 1줄.
  python code/run_train.py --setting P3 --data 100 --seed 0
  --mode orig : (S2 시범 전용) 원본 M2F2DetImproved로 baseline을 실시간 계산하고 색 변형을 전체 이미지에 적용해 학습.
                검증·테스트는 두 방식 모두 같은 캐시 경로(ExpModel)로 한다(학습 모듈을 공유).
중단되면 같은 명령으로 다시 실행 → resume.pt에서 이어받는다. DONE이 있으면 건너뛴다."""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import expmodel as E  # noqa: E402

TRAIN_BS, EVAL_BS = 16, 64
LR, WD, ETA_MIN, LOSS_W_DETAIL, CLIP = 5e-5, 1e-3, 1e-6, 0.15, 1.0


# ── 데이터 ────────────────────────────────────────────────────────────
class Split:
    def __init__(self, split, flip=False):
        self.name = split
        self.ix = C.load_index(split)
        self.img = np.load(os.path.join(C.FRAME_DIR, f'{split}_img.npy'), mmap_mode='r')
        self.b3 = np.load(os.path.join(C.BASE_DIR, f'{split}_b3.npy'), mmap_mode='r')
        self.lg = np.load(os.path.join(C.BASE_DIR, f'{split}_logit.npy'), mmap_mode='r')
        if flip:
            self.b3f = np.load(os.path.join(C.BASE_DIR, f'{split}_flip_b3.npy'), mmap_mode='r')
            self.lgf = np.load(os.path.join(C.BASE_DIR, f'{split}_flip_logit.npy'), mmap_mode='r')
        self.N = len(self.ix['label'])

    def batch(self, rows, flips=None):
        """→ x (B,3,224,224) float [0,1] (반전 적용됨), b3 (B,272,11,11) fp16, logit (B,2) fp32, y (B,) long — 모두 GPU"""
        rows = np.asarray(rows)
        img = self.img[rows]
        if flips is None or not flips.any():
            b3, lg = self.b3[rows], self.lg[rows]
        else:
            b3 = np.empty((len(rows), 272, 11, 11), np.float16)
            lg = np.empty((len(rows), 2), np.float32)
            nf = ~flips
            if nf.any():
                b3[nf], lg[nf] = self.b3[rows[nf]], self.lg[rows[nf]]
            b3[flips], lg[flips] = self.b3f[rows[flips]], self.lgf[rows[flips]]
        x = torch.from_numpy(img).cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255.)
        if flips is not None and flips.any():
            fl = torch.from_numpy(flips).cuda()
            x = torch.where(fl[:, None, None, None], x.flip(3), x)
        y = torch.from_numpy(self.ix['label'][rows].astype(np.int64)).cuda()
        return x, torch.from_numpy(b3).cuda(), torch.from_numpy(lg).cuda(), y


def epoch_plan(sched, seed, epoch):
    """PROTOCOL §3.4·§6.3: 진짜 전부 + 가짜 chunk (e−1) mod 4, 순서·반전·색 계수는 (seed, epoch)로만 결정"""
    rows = np.concatenate([sched['real'], sched[f'fake{(epoch - 1) % 4}']])
    rows = rows[np.random.default_rng([seed, epoch, 0]).permutation(len(rows))]
    n = len(rows)
    flips = np.random.default_rng([seed, epoch, 1]).random(n) < 0.5
    r2 = np.random.default_rng([seed, epoch, 2])
    bf, cf, sf = r2.uniform(0.8, 1.2, n), r2.uniform(0.8, 1.2, n), r2.uniform(0.9, 1.1, n)
    return rows, flips, bf, cf, sf


# ── 평가 ──────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(models, S, rows=None):
    """eval 모드·autocast fp16. 반환: 모델마다 dict(prob, dprob, alpha, peaks, pv, ce, ce_d)"""
    for m in models:
        m.eval()
    rows = np.arange(S.N) if rows is None else np.asarray(rows)
    N = len(rows)
    outs = []
    for m in models:
        k = m.k
        outs.append({'prob': np.empty(N, np.float32), 'dprob': np.empty(N, np.float16),
                     'alpha': np.empty(N, np.float16), 'peaks': np.empty((N, k, 2), np.int8),
                     'pv': np.empty((N, k), np.float16), 'ce': np.empty(N, np.float32),
                     'ce_d': np.empty(N, np.float32)})
    for s in range(0, N, EVAL_BS):
        r = rows[s:s + EVAL_BS]
        x, b3, lg, y = S.batch(r)
        sl = slice(s, s + len(r))
        for m, o in zip(models, outs):
            with torch.autocast('cuda', dtype=torch.float16):
                out = m(x, b3, lg)
            pred, det = out['pred'].float(), out['detail_pred'].float()
            o['prob'][sl] = torch.softmax(pred, 1)[:, 1].cpu().numpy()
            o['dprob'][sl] = torch.softmax(det, 1)[:, 1].cpu().numpy()
            o['alpha'][sl] = out['alpha'].float().view(-1).cpu().numpy()
            o['peaks'][sl] = out['peaks'].cpu().numpy()
            o['pv'][sl] = out['peak_vals'].float().cpu().numpy()
            o['ce'][sl] = F.cross_entropy(pred, y, reduction='none').cpu().numpy()
            o['ce_d'][sl] = F.cross_entropy(det, y, reduction='none').cpu().numpy()
    return outs


def val_metrics(o, ix):
    lab = ix['label'].astype(int)
    real, fake = lab == 0, lab == 1
    p = o['prob'].astype(np.float64)
    vid = ix['vid']
    V = vid.max() + 1
    vs = np.bincount(vid, weights=p, minlength=V) / np.bincount(vid, minlength=V)
    vl = np.zeros(V, int)
    vl[vid] = lab
    pred = p >= 0.5
    ce = o['ce'].astype(np.float64)
    m = {'val_loss_bal': 0.5 * (ce[real].mean() + ce[fake].mean()), 'val_loss_raw': ce.mean(),
         'val_loss_detail': o['ce_d'].astype(np.float64).mean(), 'val_auc_frame': roc_auc_score(lab, p) * 100,
         'val_auc_video': roc_auc_score(vl, vs) * 100, 'val_acc05': (pred == lab).mean() * 100,
         'val_bacc05': (pred[fake].mean() + (~pred[real]).mean()) / 2 * 100}
    return {k: float(v) for k, v in m.items()}


# ── 모델 구성 ─────────────────────────────────────────────────────────
def build_model(cfg, seed, init):
    m = E.ExpModel(cfg, seed=seed)
    msg = m.load_init(init)
    return m.cuda(), msg


def build_orig_model(em):
    """원본 M2F2DetImproved(실시간 baseline)를 만들고 학습 모듈을 em과 공유"""
    from improved_model import M2F2DetImproved   # 원본, 읽기 전용
    base = nn.DataParallel(C.load_baseline('cuda'))
    om = M2F2DetImproved(base, k=em.k, crop_size=em.cfg['crop'], alpha='dynamic',
                         aggregate_mode='weighted', min_distance=em.cfg['min_distance']).cuda()
    om.heatmap_head = em.heatmap_head
    om.detail_branch = em.detail_branch
    om.gating_network = em.gating_network
    for p in om.baseline.parameters():
        p.requires_grad = False
    return om


def cfg_sha(cfg, args):
    d = {**cfg, 'init': args.init, 'epochs': args.epochs, 'data': args.data, 'mode': args.mode,
         'lr': LR, 'wd': WD, 'bs': TRAIN_BS, 'loss_w_detail': LOSS_W_DETAIL, 'clip': CLIP}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:12]


def set_determinism():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── 주 루프 ───────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', required=True, choices=list(E.SETTINGS))
    ap.add_argument('--data', type=int, default=100, choices=list(C.DATA_PCTS))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--kind', default='main', choices=['main', 'pilot', 'debug'])
    ap.add_argument('--mode', default='cache', choices=['cache', 'orig'])
    ap.add_argument('--init', default=None, help='기본: 설정 표의 init. 시범은 sbi1')
    ap.add_argument('--run_id', default=None)
    ap.add_argument('--max_steps', type=int, default=0, help='디버그: epoch당 스텝 제한')
    ap.add_argument('--no_test', action='store_true', help='디버그: 테스트 평가·registry 생략')
    args = ap.parse_args()
    cfg = E.setting_cfg(args.setting)
    args.init = args.init or cfg['init']
    if args.mode == 'orig':
        assert args.setting == 'P3', '원본 방식은 P3(현재 모델) 시범 전용'
    run_id = args.run_id or f'{args.setting}_d{args.data}_s{args.seed}'
    rdir = os.path.join(C.RUNS, run_id)
    os.makedirs(rdir, exist_ok=True)
    if os.path.exists(os.path.join(rdir, 'DONE')):
        print(f'[{run_id}] 이미 완료 → 건너뜀')
        return
    set_determinism()
    t_start = time.time()
    started = time.strftime('%Y-%m-%d %H:%M:%S')

    TR = Split('train', flip=True)
    VA = Split('val')
    sched = dict(np.load(os.path.join(C.IDX_DIR, f'schedule_d{args.data}.npz')))
    em, init_msg = build_model(cfg, args.seed, args.init)
    om = build_orig_model(em) if args.mode == 'orig' else None
    params = em.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    scaler = torch.amp.GradScaler('cuda')
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=ETA_MIN)
    curves, best = [], {'epoch': 0, 'val_loss_bal': float('inf')}
    start_epoch = 1
    res_path = os.path.join(rdir, 'resume.pt')
    if os.path.exists(res_path):
        st = torch.load(res_path, map_location='cpu', weights_only=False)
        em.load_state_dict(st['model'])
        opt.load_state_dict(st['opt'])
        scaler.load_state_dict(st['scaler'])
        sch.load_state_dict(st['sched'])
        curves, best, start_epoch = st['curves'], st['best'], st['epoch'] + 1
        torch.set_rng_state(st['rng']['torch'])
        torch.cuda.set_rng_state_all(st['rng']['cuda'])
        random.setstate(st['rng']['py'])
        print(f'[{run_id}] epoch {st["epoch"]}까지 완료된 상태에서 이어받기', flush=True)
    else:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    config = {'run_id': run_id, 'kind': args.kind, 'setting': args.setting, 'cfg': cfg, 'data_pct': args.data,
              'seed': args.seed, 'epochs': args.epochs, 'mode': args.mode, 'init': args.init, 'init_msg': init_msg,
              'lr': LR, 'wd': WD, 'eta_min': ETA_MIN, 'train_bs': TRAIN_BS, 'eval_bs': EVAL_BS,
              'loss_w_detail': LOSS_W_DETAIL, 'clip': CLIP, 'amp': 'fp16 autocast + GradScaler',
              'steps_per_epoch': (len(sched['real']) * 2) // TRAIN_BS, 'max_steps': args.max_steps,
              'code_sha': C.code_sha(), 'cfg_sha': cfg_sha(cfg, args), 'index_sha': C.index_sha(),
              'cache_sha': C.cache_sha(), 'versions': C.env_versions(), 'started': started,
              'n_trainable': sum(p.numel() for p in params)}
    json.dump(config, open(os.path.join(rdir, 'config.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'[{run_id}] {init_msg} | 학습 파라미터 {config["n_trainable"]:,} | epoch {start_epoch}→{args.epochs}',
          flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        em.train()
        if om is not None:
            om.train()
            om.baseline.eval()
        rows, flips, bf, cf, sf = epoch_plan(sched, args.seed, epoch)
        n_steps = len(rows) // TRAIN_BS
        if args.max_steps:
            n_steps = min(n_steps, args.max_steps)
        acc_l = torch.zeros(3, device='cuda', dtype=torch.float64)   # 손실 합(합계·결합·detail), 스텝마다 동기화 안 함
        n_seen = 0
        probs, labs = [], []
        for st_i in range(n_steps):
            sl = slice(st_i * TRAIN_BS, (st_i + 1) * TRAIN_BS)
            x, b3, lg, y = TR.batch(rows[sl], flips[sl])
            x = E.color_jitter(x, bf[sl], cf[sl], sf[sl])
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                out = om(x, return_dict=True) if om is not None else em(x, b3, lg)
                lf = F.cross_entropy(out['pred'].float(), y)
                ld = F.cross_entropy(out['detail_pred'].float(), y)
                loss = lf + LOSS_W_DETAIL * ld
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=CLIP)
            scaler.step(opt)
            scaler.update()
            b = y.shape[0]
            acc_l += torch.stack([loss.detach(), lf.detach(), ld.detach()]).double() * b
            n_seen += b
            probs.append(torch.softmax(out['pred'].detach().float(), 1)[:, 1])
            labs.append(y)
        lr_now = opt.param_groups[0]['lr']
        sch.step()
        p_tr = torch.cat(probs).cpu().numpy()
        y_tr = torch.cat(labs).cpu().numpy()
        tl, tf, td = (acc_l / n_seen).tolist()
        tl, tf, td = tl * n_seen, tf * n_seen, td * n_seen
        if not np.isfinite(tl) or not np.isfinite(p_tr).all():
            raise RuntimeError(f'epoch {epoch} 학습 손실/확률 비유한')
        vo = predict([em], VA)[0]
        vm = val_metrics(vo, VA.ix)
        row = {'epoch': epoch, 'lr': lr_now, 'train_loss': tl / n_seen, 'train_loss_fused': tf / n_seen,
               'train_loss_detail': td / n_seen, 'train_acc': ((p_tr >= 0.5) == y_tr).mean() * 100,
               'train_auc': roc_auc_score(y_tr, p_tr) * 100, **vm, 'sec': time.time() - t0,
               'grad_scale': scaler.get_scale()}
        curves.append(row)
        if vm['val_loss_bal'] < best['val_loss_bal']:
            best = {'epoch': epoch, 'val_loss_bal': vm['val_loss_bal']}
            torch.save({'model': em.state_dict(), 'epoch': epoch, 'cfg': cfg}, os.path.join(rdir, 'ckpt_best.pt'))
        if epoch == args.epochs:
            torch.save({'model': em.state_dict(), 'epoch': epoch, 'cfg': cfg}, os.path.join(rdir, 'ckpt_last.pt'))
        with open(os.path.join(rdir, 'curves.csv'), 'w', encoding='utf-8') as f:
            keys = list(curves[0].keys())
            f.write(','.join(keys) + '\n')
            for r in curves:
                f.write(','.join(f'{r[k]:.6g}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
        torch.save({'model': em.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'sched': sch.state_dict(), 'epoch': epoch, 'curves': curves, 'best': best,
                    'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                            'py': random.getstate()}}, res_path)
        print(f'[{run_id}] ep {epoch:2d} lr {lr_now:.2e} | train loss {row["train_loss"]:.4f} acc {row["train_acc"]:.2f} '
              f'auc {row["train_auc"]:.2f} | val bal {vm["val_loss_bal"]:.4f} raw {vm["val_loss_raw"]:.4f} '
              f'aucF {vm["val_auc_frame"]:.2f} aucV {vm["val_auc_video"]:.2f} bacc {vm["val_bacc05"]:.2f} | '
              f'{row["sec"]:.0f}s', flush=True)

    train_min = (time.time() - t_start) / 60
    if args.no_test:
        print(f'[{run_id}] 테스트 생략(디버그), {train_min:.1f}분', flush=True)
        return
    del TR
    # ── 테스트: best·last를 한 번의 데이터 순회로 평가 ──
    models, tags = [], ['best', 'last']
    for tag in tags:
        m = E.ExpModel(cfg, seed=args.seed).cuda()
        ck = torch.load(os.path.join(rdir, f'ckpt_{tag}.pt'), map_location='cpu', weights_only=False)
        m.load_state_dict(ck['model'])
        models.append(m)
    TE, CD = Split('test'), Split('cdf')
    o_te = predict(models, TE)
    o_cd = predict(models, CD)
    metrics = {}
    for tag, ot, oc in zip(tags, o_te, o_cd):
        mt, vs_te = C.ffpp_metrics(ot['prob'], TE.ix)
        mc, vs_cd = C.cdf_metrics(oc['prob'], CD.ix)
        metrics[tag] = {**mt, **mc, 'epoch': best['epoch'] if tag == 'best' else args.epochs}
        np.savez(os.path.join(rdir, f'pred_{tag}.npz'),
                 **{f'ffpp_{k}': v for k, v in ot.items() if k not in ('ce', 'ce_d')},
                 **{f'cdf_{k}': v for k, v in oc.items() if k not in ('ce', 'ce_d')},
                 ffpp_video_score=vs_te, cdf_video_score=vs_cd)
    json.dump(metrics, open(os.path.join(rdir, 'metrics.json'), 'w'), indent=1)
    finished = time.strftime('%Y-%m-%d %H:%M:%S')
    if args.kind != 'debug':
        C.registry_append({'run_id': run_id, 'kind': args.kind, 'setting': args.setting, 'data_pct': args.data,
                           'seed': args.seed, 'epochs': args.epochs, 'status': 'done', 'best_epoch': best['epoch'],
                           'best_val_loss': best['val_loss_bal'], 'last_val_loss': curves[-1]['val_loss_bal'],
                           **C.metric_row('best', metrics['best']), **C.metric_row('last', metrics['last']),
                           'train_min': train_min, 'started': started, 'finished': finished,
                           'code_sha': config['code_sha'], 'cfg_sha': config['cfg_sha'],
                           'index_sha': config['index_sha'], 'cache_sha': config['cache_sha'],
                           'torch': torch.__version__, 'notes': f'mode={args.mode} init={args.init}'})
    open(os.path.join(rdir, 'DONE'), 'w').write(finished + '\n')
    print(f'[{run_id}] 완료 {train_min:.1f}분 | best ep {best["epoch"]}: FF++ AUC {metrics["best"]["ffpp_auc"]:.2f} '
          f'CDF {metrics["best"]["cdf_auc"]:.2f} | last: FF++ {metrics["last"]["ffpp_auc"]:.2f} '
          f'CDF {metrics["last"]["cdf_auc"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
