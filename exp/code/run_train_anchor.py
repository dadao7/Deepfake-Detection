"""TASKS4 T4-6: heatmap 고정점(anchor) 학습 — detach를 제거해 heatmap head도 분류 손실로 학습하되,
고정 seg head(SBI 경계 탐지)의 heatmap에서 멀어지지 않게 정규화한다. 목적: P4a의 '중앙 붕괴'를 막으면서 end-to-end 학습.
손실 = CE(결합) + 0.15·CE(detail) + λ·mean((σ(hm) − σ(seg_hm))²)   (seg head는 고정·eval, b_3는 캐시)
나머지는 S4(P4a)와 같다(FF++ 학습, FF++ val 선택). 추가로 epoch마다 DFD-val 영상 AUC를 기록한다(λ 선택용).
실행 이름 P4c_l{λ}_s{seed}, registry kind=anchor.  python code/run_train_anchor.py --lam 10 --seed 0"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import expmodel as E  # noqa: E402
import run_train as R  # noqa: E402
import t34_common as T  # noqa: E402

ND = os.path.join(C.CACHE, 'newdata')


class DFDVal:
    """DFD-val 프레임(신원 분리 절반, 얼굴 8장 이상 영상) — R.predict 인터페이스"""
    def __init__(self):
        split = T.dfd_split()
        z = np.load(os.path.join(ND, 'dfd_meta.npz'))
        vid, keys, vl = z['vid'], z['video_key'], z['video_label']
        m = (np.bincount(vid, minlength=len(vl)) >= 8) & np.isin(keys, split['val'])
        pos = np.full(len(vl), -1)
        pos[np.where(m)[0]] = np.arange(m.sum())
        self.rows = np.where(pos[vid] >= 0)[0]
        self.N = len(self.rows)
        fvid = pos[vid[self.rows]]
        self.ix = {'label': vl[vid[self.rows]].astype(np.int8), 'vid': fvid.astype(np.int32)}
        self.vlabel = vl[m]
        self.img = np.load(os.path.join(ND, 'dfd_img.npy'), mmap_mode='r')
        self.b3 = np.load(os.path.join(ND, 'dfd_b3.npy'), mmap_mode='r')
        self.lg = np.load(os.path.join(ND, 'dfd_logit.npy'), mmap_mode='r')

    def batch(self, rows, flips=None):
        r = self.rows[np.asarray(rows)]
        x = torch.from_numpy(np.ascontiguousarray(self.img[r])).cuda().permute(0, 3, 1, 2).float().div_(255.)
        return (x, torch.from_numpy(np.ascontiguousarray(self.b3[r])).cuda(), torch.from_numpy(np.ascontiguousarray(self.lg[r])).cuda(),
                torch.from_numpy(self.ix['label'][np.asarray(rows)].astype(np.int64)).cuda())

    def video_auc(self, prob):
        vid = self.ix['vid']
        V = len(self.vlabel)
        order = np.argsort(vid, kind='stable')
        cnt = np.bincount(vid, minlength=V)
        st = np.r_[0, np.cumsum(cnt)[:-1]]
        ps = prob[order]
        vs = np.array([C.trimmed_mean(ps[st[v]:st[v] + cnt[v]], int(0.1 * cnt[v])) for v in range(V)])
        return roc_auc_score(self.vlabel, vs) * 100


def code_sha():
    d = os.path.join(C.EXP, 'code')
    return C.sha256_files([os.path.join(d, f) for f in C.CODE_FILES + ('run_train_anchor.py',)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--run_id', default=None)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()
    lam_tag = f'{args.lam:g}'
    run_id = args.run_id or f'P4c_l{lam_tag}_s{args.seed}'
    rdir = os.path.join(C.RUNS, run_id)
    os.makedirs(rdir, exist_ok=True)
    if os.path.exists(os.path.join(rdir, 'DONE')):
        print(f'[{run_id}] 이미 완료 → 건너뜀')
        return
    R.set_determinism()
    t_start, started = time.time(), time.strftime('%Y-%m-%d %H:%M:%S')
    TR, VA, DV = R.Split('train', flip=True), R.Split('val'), DFDVal()
    sched = dict(np.load(os.path.join(C.IDX_DIR, 'schedule_d100.npz')))
    cfg = E.setting_cfg('P4a')
    em, init_msg = R.build_model(cfg, args.seed, cfg['init'])
    seg = E.HeatmapHead(272, 64).cuda().eval()
    seg.load_state_dict(torch.load(os.path.join(C.RUNS, 'sbi_pretrain', 'seg_head.pt'), map_location='cpu',
                                   weights_only=False)['heatmap_head'])
    for p in seg.parameters():
        p.requires_grad = False
    params = em.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=R.LR, weight_decay=R.WD)
    scaler = torch.amp.GradScaler('cuda')
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=R.ETA_MIN)
    curves, best, start_epoch = [], {'epoch': 0, 'val_loss_bal': float('inf')}, 1
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
    else:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    cs = hashlib.sha256(json.dumps({'setting': 'P4c', 'lam': args.lam, 'epochs': args.epochs}, sort_keys=True).encode()).hexdigest()[:12]
    config = {'run_id': run_id, 'kind': 'anchor', 'setting': f'P4c_l{lam_tag}', 'base_cfg': cfg, 'lam': args.lam,
              'anchor': 'λ·mean((σ(hm)−σ(seg_hm))²), seg head = runs/sbi_pretrain/seg_head.pt (고정)', 'seed': args.seed,
              'epochs': args.epochs, 'init_msg': init_msg, 'code_sha': code_sha(), 'cfg_sha': cs, 'started': started,
              'versions': C.env_versions()}
    json.dump(config, open(os.path.join(rdir, 'config.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'[{run_id}] anchor λ={args.lam} | {init_msg}', flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        em.train()
        rows, flips, bf, cf, sf = R.epoch_plan(sched, args.seed, epoch)
        n_steps = len(rows) // R.TRAIN_BS
        if args.max_steps:
            n_steps = min(n_steps, args.max_steps)
        acc_l = torch.zeros(4, device='cuda', dtype=torch.float64)
        n_seen, probs, labs = 0, [], []
        for st_i in range(n_steps):
            sl = slice(st_i * R.TRAIN_BS, (st_i + 1) * R.TRAIN_BS)
            x, b3, lg, y = TR.batch(rows[sl], flips[sl])
            x = E.color_jitter(x, bf[sl], cf[sl], sf[sl])
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                out = em(x, b3, lg)
                with torch.no_grad():
                    tgt = torch.sigmoid(seg(b3).float())
                lf = F.cross_entropy(out['pred'].float(), y)
                ld = F.cross_entropy(out['detail_pred'].float(), y)
                la = ((torch.sigmoid(out['heatmap'].float()) - tgt) ** 2).mean()
                loss = lf + R.LOSS_W_DETAIL * ld + args.lam * la
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=R.CLIP)
            scaler.step(opt)
            scaler.update()
            acc_l += torch.stack([loss.detach(), lf.detach(), ld.detach(), la.detach()]).double() * y.shape[0]
            n_seen += y.shape[0]
            probs.append(torch.softmax(out['pred'].detach().float(), 1)[:, 1])
            labs.append(y)
        lr_now = opt.param_groups[0]['lr']
        sch.step()
        p_tr, y_tr = torch.cat(probs).cpu().numpy(), torch.cat(labs).cpu().numpy()
        tl, tf_, td, ta = (acc_l / n_seen).tolist()
        if not np.isfinite(tl):
            raise RuntimeError('비유한 손실')
        vm = R.val_metrics(R.predict([em], VA)[0], VA.ix)
        dv = R.predict([em], DV)[0]
        row = {'epoch': epoch, 'lr': lr_now, 'train_loss': tl, 'train_loss_fused': tf_, 'train_loss_detail': td,
               'train_loss_anchor': ta, 'train_acc': float(((p_tr >= 0.5) == y_tr).mean() * 100),
               'train_auc': roc_auc_score(y_tr, p_tr) * 100, **vm, 'dfdval_auc_video': DV.video_auc(dv['prob']),
               'sec': time.time() - t0}
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
        torch.save({'model': em.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'sched': sch.state_dict(),
                    'epoch': epoch, 'curves': curves, 'best': best,
                    'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(), 'py': random.getstate()}}, res_path)
        print(f'[{run_id}] ep {epoch:2d} | train {tl:.4f} anchor {ta:.5f} | FFval bal {vm["val_loss_bal"]:.4f} aucV '
              f'{vm["val_auc_video"]:.2f} | DFDval aucV {row["dfdval_auc_video"]:.2f} | {row["sec"]:.0f}s', flush=True)
    if args.debug:
        return
    train_min = (time.time() - t_start) / 60
    del TR
    models = []
    for tag in ('best', 'last'):
        m = E.ExpModel(cfg, seed=args.seed).cuda()
        m.load_state_dict(torch.load(os.path.join(rdir, f'ckpt_{tag}.pt'), map_location='cpu', weights_only=False)['model'])
        models.append(m)
    TE, CD = R.Split('test'), R.Split('cdf')
    o_te, o_cd = R.predict(models, TE), R.predict(models, CD)
    metrics = {}
    for tag, ot, oc in zip(('best', 'last'), o_te, o_cd):
        mt, vs_te = C.ffpp_metrics(ot['prob'], TE.ix)
        mc, vs_cd = C.cdf_metrics(oc['prob'], CD.ix)
        metrics[tag] = {**mt, **mc, 'epoch': best['epoch'] if tag == 'best' else args.epochs}
        np.savez(os.path.join(rdir, f'pred_{tag}.npz'), **{f'ffpp_{k}': v for k, v in ot.items() if k not in ('ce', 'ce_d')},
                 **{f'cdf_{k}': v for k, v in oc.items() if k not in ('ce', 'ce_d')}, ffpp_video_score=vs_te, cdf_video_score=vs_cd)
    json.dump(metrics, open(os.path.join(rdir, 'metrics.json'), 'w'), indent=1)
    finished = time.strftime('%Y-%m-%d %H:%M:%S')
    C.registry_append({'run_id': run_id, 'kind': 'anchor', 'setting': f'P4c_l{lam_tag}', 'data_pct': 100, 'seed': args.seed,
                       'epochs': args.epochs, 'status': 'done', 'best_epoch': best['epoch'], 'best_val_loss': best['val_loss_bal'],
                       'last_val_loss': curves[-1]['val_loss_bal'], **C.metric_row('best', metrics['best']),
                       **C.metric_row('last', metrics['last']), 'train_min': train_min, 'started': started,
                       'finished': finished, 'code_sha': config['code_sha'], 'cfg_sha': cs, 'index_sha': C.index_sha(),
                       'cache_sha': C.cache_sha(), 'torch': torch.__version__, 'notes': f'heatmap anchor λ={args.lam} (TASKS4 T4-6)'})
    open(os.path.join(rdir, 'DONE'), 'w').write(finished + '\n')
    print(f'[{run_id}] 완료 {train_min:.1f}분 | best ep {best["epoch"]}: FF++ {metrics["best"]["ffpp_auc"]:.2f} CDF '
          f'{metrics["best"]["cdf_auc"]:.2f} | last: FF++ {metrics["last"]["ffpp_auc"]:.2f} CDF {metrics["last"]["cdf_auc"]:.2f}',
          flush=True)


if __name__ == '__main__':
    main()
