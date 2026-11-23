"""S9 후속 실험: baseline 일부 풀기 (PROTOCOL §15). 저자 baseline 상단(EffNet f7·f8 + 투영층 + 출력층)을 캐시(CLS + b_3)로 학습.
  U0 = baseline만 (손실 CE),  U3 = P3 + 풀린 baseline 상단 (손실 CE(결합) + 0.15·CE(detail))
  --data ffpp: S4 스케줄(1:1, 4묶음 순환, 반전 캐시), 선택 = FF++ val / --data sbi: S8 스케줄, 선택 = SBI val (다른 val은 기록만)
실행 1회 = runs/{U0|U3}{f|s}_s{seed}/ + registry 1줄(kind=unfreeze). 기존 학습 코드는 고치지 않고 함수를 불러 쓴다.
  python code/run_train_unfreeze.py --cfg U3 --data ffpp --seed 0"""
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
import basetop as BT  # noqa: E402
import common as C  # noqa: E402
import expmodel as E  # noqa: E402
import run_train as R  # noqa: E402
import run_train_sbi as S  # noqa: E402


class FFData(R.Split):
    def __init__(self, split, flip=False):
        super().__init__(split, flip)
        self.cls = np.load(os.path.join(C.BASE_DIR, f'{split}_cls.npy'), mmap_mode='r')
        if flip:
            self.clsf = np.load(os.path.join(C.BASE_DIR, f'{split}_flip_cls.npy'), mmap_mode='r')

    def batch(self, rows, flips=None):
        x, b3, lg, y = super().batch(rows, flips)
        rows = np.asarray(rows)
        if flips is None or not flips.any():
            c = self.cls[rows]
        else:
            c = np.empty((len(rows), 1024), np.float16)
            c[~flips], c[flips] = self.cls[rows[~flips]], self.clsf[rows[flips]]
        return x, b3, lg, y, torch.from_numpy(np.ascontiguousarray(c)).cuda()


class SBIData(S.SBIPairs):
    def __init__(self, split):
        super().__init__(split)
        self.cls = np.load(os.path.join(S.SBI_DIR, f'{split}_cls.npy'), mmap_mode='r')

    def batch(self, rows, flips=None):
        x, b3, lg, y = super().batch(rows)
        rows = np.asarray(rows)
        return x, b3, lg, y, torch.from_numpy(np.ascontiguousarray(self.cls[rows // 2, rows % 2])).cuda()


class UModel(nn.Module):
    def __init__(self, cfg, seed):
        super().__init__()
        self.cfg_name = cfg
        self.top = BT.BaselineTop()
        self.msg = self.top.load_author()
        self.branch, self.k = None, 1
        if cfg == 'U3':
            self.branch = E.ExpModel(E.setting_cfg('P3'), seed=seed)
            self.msg += ' | P3 분기 ' + self.branch.load_init('sbi')
            self.k = self.branch.k

    def forward(self, x, b3, cls):
        bl = self.top(cls, b3)
        if self.branch is None:
            B = bl.shape[0]
            return {'pred': bl, 'detail_pred': bl, 'alpha': torch.ones_like(bl[:, :1]),
                    'peaks': torch.zeros(B, 1, 2, dtype=torch.long, device=bl.device),
                    'peak_vals': torch.zeros(B, 1, device=bl.device)}
        return self.branch(x, b3, bl.float())


@torch.no_grad()
def predict(models, D, rows=None):
    """R.predict와 같은 출력(CLS를 함께 넣는 판)"""
    for m in models:
        m.eval()
    rows = np.arange(D.N) if rows is None else np.asarray(rows)
    N = len(rows)
    outs = [{'prob': np.empty(N, np.float32), 'dprob': np.empty(N, np.float16), 'alpha': np.empty(N, np.float16),
             'peaks': np.empty((N, m.k, 2), np.int8), 'pv': np.empty((N, m.k), np.float16),
             'ce': np.empty(N, np.float32), 'ce_d': np.empty(N, np.float32), 'base_prob': np.empty(N, np.float32)}
            for m in models]
    for s in range(0, N, R.EVAL_BS):
        r = rows[s:s + R.EVAL_BS]
        x, b3, lg, y, cls = D.batch(r)
        sl = slice(s, s + len(r))
        for m, o in zip(models, outs):
            with torch.autocast('cuda', dtype=torch.float16):
                out = m(x, b3, cls)
                bl = m.top(cls, b3)
            pred, det = out['pred'].float(), out['detail_pred'].float()
            o['prob'][sl] = torch.softmax(pred, 1)[:, 1].cpu().numpy()
            o['dprob'][sl] = torch.softmax(det, 1)[:, 1].cpu().numpy()
            o['base_prob'][sl] = torch.softmax(bl.float(), 1)[:, 1].cpu().numpy()
            o['alpha'][sl] = out['alpha'].float().view(-1).cpu().numpy()
            o['peaks'][sl] = out['peaks'].cpu().numpy()
            o['pv'][sl] = out['peak_vals'].float().cpu().numpy()
            o['ce'][sl] = F.cross_entropy(pred, y, reduction='none').cpu().numpy()
            o['ce_d'][sl] = F.cross_entropy(det, y, reduction='none').cpu().numpy()
    return outs


def code_sha():
    d = os.path.join(C.EXP, 'code')
    return C.sha256_files([os.path.join(d, f) for f in C.CODE_FILES + ('run_train_sbi.py', 'basetop.py',
                                                                         'run_train_unfreeze.py')])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True, choices=['U0', 'U3'])
    ap.add_argument('--data', required=True, choices=['ffpp', 'sbi'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--kind', default='unfreeze', choices=['unfreeze', 'debug'])
    ap.add_argument('--run_id', default=None)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--no_test', action='store_true')
    args = ap.parse_args()
    setting = f'{args.cfg}{"f" if args.data == "ffpp" else "s"}'
    run_id = args.run_id or f'{setting}_s{args.seed}'
    rdir = os.path.join(C.RUNS, run_id)
    os.makedirs(rdir, exist_ok=True)
    if os.path.exists(os.path.join(rdir, 'DONE')):
        print(f'[{run_id}] 이미 완료 → 건너뜀')
        return
    R.set_determinism()
    t_start, started = time.time(), time.strftime('%Y-%m-%d %H:%M:%S')
    if args.data == 'ffpp':
        TR = FFData('train', flip=True)
        sched = dict(np.load(os.path.join(C.IDX_DIR, 'schedule_d100.npz')))
    else:
        TR = SBIData('train')
    VF, VS = FFData('val'), SBIData('val')
    VSEL = VF if args.data == 'ffpp' else VS
    model = UModel(args.cfg, args.seed).cuda()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=R.LR, weight_decay=R.WD)
    scaler = torch.amp.GradScaler('cuda')
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=R.ETA_MIN)
    curves, best, start_epoch = [], {'epoch': 0, 'val_loss_bal': float('inf')}, 1
    res_path = os.path.join(rdir, 'resume.pt')
    if os.path.exists(res_path):
        st = torch.load(res_path, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model'])
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
    cs = hashlib.sha256(json.dumps({'cfg': args.cfg, 'data': args.data, 'epochs': args.epochs, 'lr': R.LR, 'wd': R.WD,
                                    'bs': R.TRAIN_BS, 'unfrozen': 'f7,f8,deepfake_proj,vision_proj,alpha_v,output'},
                                   sort_keys=True).encode()).hexdigest()[:12]
    config = {'run_id': run_id, 'kind': args.kind, 'setting': setting, 'cfg': args.cfg, 'train_data': args.data,
              'unfrozen': 'EffNet features[7], features[8], deepfake_proj, vision_proj, clip_vision_alpha, output (eval 모드)',
              'selection': f'{"FF++" if args.data == "ffpp" else "SBI"} val val_loss_bal', 'seed': args.seed,
              'epochs': args.epochs, 'init_msg': model.msg, 'lr': R.LR, 'wd': R.WD, 'eta_min': R.ETA_MIN,
              'train_bs': R.TRAIN_BS, 'loss': 'CE' if args.cfg == 'U0' else 'CE(fused)+0.15*CE(detail)',
              'clip': R.CLIP, 'max_steps': args.max_steps, 'code_sha': code_sha(), 'cfg_sha': cs,
              'index_sha': C.index_sha(), 'cache_sha': C.cache_sha(), 'versions': C.env_versions(),
              'started': started, 'n_trainable': sum(p.numel() for p in params)}
    json.dump(config, open(os.path.join(rdir, 'config.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'[{run_id}] {model.msg} | 학습 파라미터 {config["n_trainable"]:,} | epoch {start_epoch}→{args.epochs}', flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        if args.data == 'ffpp':
            rows, flips, bf, cf, sf = R.epoch_plan(sched, args.seed, epoch)
        else:
            (rows, bf, cf, sf), flips = S.epoch_plan(TR.P, args.seed, epoch), None
        n_steps = len(rows) // R.TRAIN_BS
        if args.max_steps:
            n_steps = min(n_steps, args.max_steps)
        acc_l = torch.zeros(3, device='cuda', dtype=torch.float64)
        n_seen, probs, labs = 0, [], []
        for st_i in range(n_steps):
            sl = slice(st_i * R.TRAIN_BS, (st_i + 1) * R.TRAIN_BS)
            x, b3, lg, y, cls = TR.batch(rows[sl], None if flips is None else flips[sl])
            x = E.color_jitter(x, bf[sl], cf[sl], sf[sl])
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(x, b3, cls)
                lf = F.cross_entropy(out['pred'].float(), y)
                ld = F.cross_entropy(out['detail_pred'].float(), y)
                loss = lf if args.cfg == 'U0' else lf + R.LOSS_W_DETAIL * ld
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=R.CLIP)
            scaler.step(opt)
            scaler.update()
            b = y.shape[0]
            acc_l += torch.stack([loss.detach(), lf.detach(), ld.detach()]).double() * b
            n_seen += b
            probs.append(torch.softmax(out['pred'].detach().float(), 1)[:, 1])
            labs.append(y)
        lr_now = opt.param_groups[0]['lr']
        sch.step()
        p_tr, y_tr = torch.cat(probs).cpu().numpy(), torch.cat(labs).cpu().numpy()
        tl, tf, td = (acc_l / n_seen).tolist()
        if not np.isfinite(tl) or not np.isfinite(p_tr).all():
            raise RuntimeError(f'epoch {epoch} 학습 손실/확률 비유한')
        vf = R.val_metrics(predict([model], VF)[0], VF.ix)
        vs = R.val_metrics(predict([model], VS)[0], VS.ix)
        vm = vf if args.data == 'ffpp' else vs
        other = {f'sbival_{k[4:]}': v for k, v in vs.items()} if args.data == 'ffpp' else \
            {f'ffval_{k[4:]}': v for k, v in vf.items()}
        row = {'epoch': epoch, 'lr': lr_now, 'train_loss': tl, 'train_loss_fused': tf, 'train_loss_detail': td,
               'train_acc': float(((p_tr >= 0.5) == y_tr).mean() * 100), 'train_auc': roc_auc_score(y_tr, p_tr) * 100,
               **vm, **other, 'sec': time.time() - t0, 'grad_scale': scaler.get_scale()}
        curves.append(row)
        if vm['val_loss_bal'] < best['val_loss_bal']:
            best = {'epoch': epoch, 'val_loss_bal': vm['val_loss_bal']}
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'cfg': args.cfg}, os.path.join(rdir, 'ckpt_best.pt'))
        if epoch == args.epochs:
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'cfg': args.cfg}, os.path.join(rdir, 'ckpt_last.pt'))
        with open(os.path.join(rdir, 'curves.csv'), 'w', encoding='utf-8') as f:
            keys = list(curves[0].keys())
            f.write(','.join(keys) + '\n')
            for r in curves:
                f.write(','.join(f'{r[k]:.6g}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'sched': sch.state_dict(), 'epoch': epoch, 'curves': curves, 'best': best,
                    'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(),
                            'py': random.getstate()}}, res_path)
        print(f'[{run_id}] ep {epoch:2d} | train loss {tl:.4f} acc {row["train_acc"]:.2f} | FFval bal {vf["val_loss_bal"]:.4f} '
              f'aucV {vf["val_auc_video"]:.2f} bacc {vf["val_bacc05"]:.2f} | SBIval bal {vs["val_loss_bal"]:.4f} '
              f'auc {vs["val_auc_frame"]:.2f} | {row["sec"]:.0f}s', flush=True)

    train_min = (time.time() - t_start) / 60
    if args.no_test:
        print(f'[{run_id}] 테스트 생략(디버그), {train_min:.1f}분', flush=True)
        return
    del TR
    models, tags = [], ['best', 'last']
    for tag in tags:
        m = UModel(args.cfg, args.seed).cuda()
        m.load_state_dict(torch.load(os.path.join(rdir, f'ckpt_{tag}.pt'), map_location='cpu', weights_only=False)['model'])
        models.append(m)
    TE, CD = FFData('test'), FFData('cdf')
    o_te, o_cd = predict(models, TE), predict(models, CD)
    metrics = {}
    for tag, ot, oc in zip(tags, o_te, o_cd):
        mt, vs_te = C.ffpp_metrics(ot['prob'], TE.ix)
        mc, vs_cd = C.cdf_metrics(oc['prob'], CD.ix)
        mbt, _ = C.ffpp_metrics(ot['base_prob'], TE.ix)
        mbc, _ = C.cdf_metrics(oc['base_prob'], CD.ix)
        metrics[tag] = {**mt, **mc, 'epoch': best['epoch'] if tag == 'best' else args.epochs,
                        'base_only_ffpp_auc': mbt['ffpp_auc'], 'base_only_cdf_auc': mbc['cdf_auc']}
        np.savez(os.path.join(rdir, f'pred_{tag}.npz'),
                 **{f'ffpp_{k}': v for k, v in ot.items() if k not in ('ce', 'ce_d')},
                 **{f'cdf_{k}': v for k, v in oc.items() if k not in ('ce', 'ce_d')},
                 ffpp_video_score=vs_te, cdf_video_score=vs_cd)
    json.dump(metrics, open(os.path.join(rdir, 'metrics.json'), 'w'), indent=1)
    finished = time.strftime('%Y-%m-%d %H:%M:%S')
    if args.kind != 'debug':
        C.registry_append({'run_id': run_id, 'kind': 'unfreeze', 'setting': setting, 'data_pct': 100,
                           'seed': args.seed, 'epochs': args.epochs, 'status': 'done', 'best_epoch': best['epoch'],
                           'best_val_loss': best['val_loss_bal'], 'last_val_loss': curves[-1]['val_loss_bal'],
                           **C.metric_row('best', metrics['best']), **C.metric_row('last', metrics['last']),
                           'train_min': train_min, 'started': started, 'finished': finished,
                           'code_sha': config['code_sha'], 'cfg_sha': cs, 'index_sha': config['index_sha'],
                           'cache_sha': config['cache_sha'], 'torch': torch.__version__,
                           'notes': f'unfreeze top; train={args.data}; base_only best FF++ {metrics["best"]["base_only_ffpp_auc"]:.2f} '
                                    f'CDF {metrics["best"]["base_only_cdf_auc"]:.2f}'})
    open(os.path.join(rdir, 'DONE'), 'w').write(finished + '\n')
    print(f'[{run_id}] 완료 {train_min:.1f}분 | best ep {best["epoch"]}: FF++ AUC {metrics["best"]["ffpp_auc"]:.2f} '
          f'CDF {metrics["best"]["cdf_auc"]:.2f} | last: FF++ {metrics["last"]["ffpp_auc"]:.2f} '
          f'CDF {metrics["last"]["cdf_auc"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
