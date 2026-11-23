"""TASKS6 T1~T3: SBI 전문가(U3 구조) 개선 학습. 데이터·스케줄·손실·선택은 S9 U3s와 같다(SBI 쌍 S8 스케줄, CE(결합) + 0.15·CE(detail),
20 epoch, AdamW wd 1e-3, cosine, 배치 16, SBI val 균형 CE 최저 = best + last). 바꾸는 것만:
  --init theta0 | p3last | p3best : 시작점(θ0 = 저자 상단 + SBI 사전학습 분기, p3* = θ0 상단 + P3 분기)
  --ema D : 0이면 끔. D>0이면 매 스텝 EMA 가중치를 유지하고, 검증·저장은 EMA 가중치로 한다.
  --lr    : 학습률(기본 5e-5)
registry kind=sbi2, 실행 이름 기본 X{tag}_s{seed} (tag는 --tag). 기존 학습 코드는 고치지 않고 함수를 불러 쓴다.
  python code/run_train_sbi2.py --tag p3 --init p3last --seed 0"""
import argparse
import copy
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
import run_train_sbi as S  # noqa: E402
import run_train_unfreeze as U  # noqa: E402


def code_sha():
    d = os.path.join(C.EXP, 'code')
    return C.sha256_files([os.path.join(d, f) for f in C.CODE_FILES + ('run_train_sbi.py', 'basetop.py', 'run_train_unfreeze.py',
                                                                         'run_train_sbi2.py')])


def init_model(init, seed):
    m = U.UModel('U3', seed)
    msg = m.msg
    if init != 'theta0':
        sd = torch.load(os.path.join(C.RUNS, f'P3_d100_s{seed}', f'ckpt_{init[2:]}.pt'), map_location='cpu', weights_only=False)['model']
        r = m.branch.load_state_dict(sd, strict=True)
        msg += f' | P3_d100_s{seed} {init[2:]} 분기에서 시작({r})'
    return m, msg


@torch.no_grad()
def ema_update(ema, model, d):
    for (k, v), (_, w) in zip(ema.state_dict().items(), model.state_dict().items()):
        if torch.is_floating_point(v):
            v.mul_(d).add_(w.detach(), alpha=1 - d)
        else:
            v.copy_(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--init', default='theta0', choices=['theta0', 'p3last', 'p3best'])
    ap.add_argument('--ema', type=float, default=0.0)
    ap.add_argument('--lr', type=float, default=R.LR)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--run_id', default=None)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()
    run_id = args.run_id or f'X{args.tag}_s{args.seed}'
    rdir = os.path.join(C.RUNS, run_id)
    os.makedirs(rdir, exist_ok=True)
    if os.path.exists(os.path.join(rdir, 'DONE')):
        print(f'[{run_id}] 이미 완료 → 건너뜀')
        return
    R.set_determinism()
    t_start, started = time.time(), time.strftime('%Y-%m-%d %H:%M:%S')
    TR = U.SBIData('train')
    VF, VS = U.FFData('val'), U.SBIData('val')
    model, init_msg = init_model(args.init, args.seed)
    model = model.cuda()
    ema = copy.deepcopy(model).eval() if args.ema > 0 else None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=R.WD)
    scaler = torch.amp.GradScaler('cuda')
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=min(R.ETA_MIN, args.lr / 10))
    curves, best, start_epoch = [], {'epoch': 0, 'val_loss_bal': float('inf')}, 1
    res_path = os.path.join(rdir, 'resume.pt')
    if os.path.exists(res_path):
        st = torch.load(res_path, map_location='cpu', weights_only=False)
        model.load_state_dict(st['model'])
        if ema is not None:
            ema.load_state_dict(st['ema'])
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
    cs = hashlib.sha256(json.dumps({'tag': args.tag, 'init': args.init, 'ema': args.ema, 'lr': args.lr, 'epochs': args.epochs},
                                   sort_keys=True).encode()).hexdigest()[:12]
    config = {'run_id': run_id, 'kind': 'sbi2', 'setting': f'X{args.tag}', 'train_data': 'sbi', 'init': args.init,
              'ema': args.ema, 'lr': args.lr, 'wd': R.WD, 'epochs': args.epochs, 'seed': args.seed, 'init_msg': init_msg,
              'selection': 'SBI val val_loss_bal' + (' (EMA 가중치)' if ema is not None else ''),
              'code_sha': code_sha(), 'cfg_sha': cs, 'index_sha': C.index_sha(), 'cache_sha': C.cache_sha(),
              'versions': C.env_versions(), 'started': started, 'n_trainable': sum(p.numel() for p in params)}
    json.dump(config, open(os.path.join(rdir, 'config.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'[{run_id}] SBI 전문가 학습 | {init_msg} | ema {args.ema} lr {args.lr} | epoch {start_epoch}→{args.epochs}', flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        rows, bf, cf, sf = S.epoch_plan(TR.P, args.seed, epoch)
        n_steps = len(rows) // R.TRAIN_BS
        if args.max_steps:
            n_steps = min(n_steps, args.max_steps)
        acc_l = torch.zeros(3, device='cuda', dtype=torch.float64)
        n_seen, probs, labs = 0, [], []
        for st_i in range(n_steps):
            sl = slice(st_i * R.TRAIN_BS, (st_i + 1) * R.TRAIN_BS)
            x, b3, lg, y, cls = TR.batch(rows[sl])
            x = E.color_jitter(x, bf[sl], cf[sl], sf[sl])
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(x, b3, cls)
                lf = F.cross_entropy(out['pred'].float(), y)
                ld = F.cross_entropy(out['detail_pred'].float(), y)
                loss = lf + R.LOSS_W_DETAIL * ld
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=R.CLIP)
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema_update(ema, model, args.ema)
            acc_l += torch.stack([loss.detach(), lf.detach(), ld.detach()]).double() * y.shape[0]
            n_seen += y.shape[0]
            probs.append(torch.softmax(out['pred'].detach().float(), 1)[:, 1])
            labs.append(y)
        lr_now = opt.param_groups[0]['lr']
        sch.step()
        p_tr, y_tr = torch.cat(probs).cpu().numpy(), torch.cat(labs).cpu().numpy()
        tl, tf_, td = (acc_l / n_seen).tolist()
        if not np.isfinite(tl):
            raise RuntimeError('비유한 손실')
        evm = ema if ema is not None else model
        vf = R.val_metrics(U.predict([evm], VF)[0], VF.ix)
        vs = R.val_metrics(U.predict([evm], VS)[0], VS.ix)
        row = {'epoch': epoch, 'lr': lr_now, 'train_loss': tl, 'train_loss_fused': tf_, 'train_loss_detail': td,
               'train_acc': float(((p_tr >= 0.5) == y_tr).mean() * 100), 'train_auc': roc_auc_score(y_tr, p_tr) * 100,
               **vs, **{f'ffval_{k[4:]}': v for k, v in vf.items()}, 'sec': time.time() - t0}
        curves.append(row)
        if vs['val_loss_bal'] < best['val_loss_bal']:
            best = {'epoch': epoch, 'val_loss_bal': vs['val_loss_bal']}
            torch.save({'model': evm.state_dict(), 'epoch': epoch}, os.path.join(rdir, 'ckpt_best.pt'))
        if epoch == args.epochs:
            torch.save({'model': evm.state_dict(), 'epoch': epoch}, os.path.join(rdir, 'ckpt_last.pt'))
        with open(os.path.join(rdir, 'curves.csv'), 'w', encoding='utf-8') as f:
            keys = list(curves[0].keys())
            f.write(','.join(keys) + '\n')
            for r in curves:
                f.write(','.join(f'{r[k]:.6g}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
        torch.save({'model': model.state_dict(), 'ema': ema.state_dict() if ema is not None else None, 'opt': opt.state_dict(),
                    'scaler': scaler.state_dict(), 'sched': sch.state_dict(), 'epoch': epoch, 'curves': curves, 'best': best,
                    'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(), 'py': random.getstate()}},
                   res_path)
        print(f'[{run_id}] ep {epoch:2d} | train {tl:.4f} | FFval bal {vf["val_loss_bal"]:.4f} aucV {vf["val_auc_video"]:.2f} | '
              f'SBIval bal {vs["val_loss_bal"]:.4f} auc {vs["val_auc_frame"]:.2f} | {row["sec"]:.0f}s', flush=True)
    if args.debug:
        return
    train_min = (time.time() - t_start) / 60
    del TR
    models = []
    for tag in ('best', 'last'):
        m = U.UModel('U3', args.seed).cuda()
        m.load_state_dict(torch.load(os.path.join(rdir, f'ckpt_{tag}.pt'), map_location='cpu', weights_only=False)['model'])
        models.append(m)
    TE, CD = U.FFData('test'), U.FFData('cdf')
    o_te, o_cd = U.predict(models, TE), U.predict(models, CD)
    metrics = {}
    for tag, ot, oc in zip(('best', 'last'), o_te, o_cd):
        mt, vs_te = C.ffpp_metrics(ot['prob'], TE.ix)
        mc, vs_cd = C.cdf_metrics(oc['prob'], CD.ix)
        metrics[tag] = {**mt, **mc, 'epoch': best['epoch'] if tag == 'best' else args.epochs}
        np.savez(os.path.join(rdir, f'pred_{tag}.npz'), **{f'ffpp_{k}': v for k, v in ot.items() if k not in ('ce', 'ce_d')},
                 **{f'cdf_{k}': v for k, v in oc.items() if k not in ('ce', 'ce_d')}, ffpp_video_score=vs_te, cdf_video_score=vs_cd)
    json.dump(metrics, open(os.path.join(rdir, 'metrics.json'), 'w'), indent=1)
    finished = time.strftime('%Y-%m-%d %H:%M:%S')
    C.registry_append({'run_id': run_id, 'kind': 'sbi2', 'setting': f'X{args.tag}', 'data_pct': 100, 'seed': args.seed,
                       'epochs': args.epochs, 'status': 'done', 'best_epoch': best['epoch'], 'best_val_loss': best['val_loss_bal'],
                       'last_val_loss': curves[-1]['val_loss_bal'], **C.metric_row('best', metrics['best']),
                       **C.metric_row('last', metrics['last']), 'train_min': train_min, 'started': started,
                       'finished': finished, 'code_sha': config['code_sha'], 'cfg_sha': cs, 'index_sha': C.index_sha(),
                       'cache_sha': C.cache_sha(), 'torch': torch.__version__,
                       'notes': f'TASKS6 SBI expert init={args.init} ema={args.ema} lr={args.lr}'})
    open(os.path.join(rdir, 'DONE'), 'w').write(finished + '\n')
    print(f'[{run_id}] 완료 {train_min:.1f}분 | best ep {best["epoch"]}: FF++ {metrics["best"]["ffpp_auc"]:.2f} CDF '
          f'{metrics["best"]["cdf_auc"]:.2f} | last: FF++ {metrics["last"]["ffpp_auc"]:.2f} CDF {metrics["last"]["cdf_auc"]:.2f}',
          flush=True)


if __name__ == '__main__':
    main()
