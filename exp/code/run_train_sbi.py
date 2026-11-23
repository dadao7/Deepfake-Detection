"""S8 후속 실험: SBI 학습 프로토콜 (PROTOCOL §14). 덧붙인 분기를 baseline이 보지 않은 데이터(SBI 쌍)로 학습.
학습 데이터만 S4와 다르고(진짜+SBI 가짜, epoch당 7,200쌍 = 900스텝), 모델·초기화·학습법·epoch·seed·테스트는 S4와 같다.
선택: SBI val 손실(best) + last. FF++ val 지표는 기록만(선택에 쓰지 않음) → FF++ 가짜는 학습·선택에 전혀 안 쓰임.
기존 run_train.py는 고치지 않고 함수만 불러 쓴다. 실행 1회 = runs/{설정}_sbi_s{seed}/ + registry 1줄(kind=sbi).
  python code/run_train_sbi.py --setting P3 --seed 0"""
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

SBI_DIR = os.path.join(C.CACHE, 'sbi')
PAIRS_PER_EPOCH = 7200          # 진짜 7,200 + SBI 7,200 = 14,400장 = 900스텝 (S4 100%와 같음)
SCHEDULE_SEED = 2027            # seed와 무관한 고정 쌍 순열


class SBIPairs:
    """cache/sbi/{split}_*: 쌍 p의 진짜(행 2p)·SBI 가짜(행 2p+1). R.predict·R.val_metrics와 같은 인터페이스"""
    def __init__(self, split):
        self.img = np.load(os.path.join(SBI_DIR, f'{split}_img.npy'), mmap_mode='r')
        self.b3 = np.load(os.path.join(SBI_DIR, f'{split}_b3.npy'), mmap_mode='r')
        self.lg = np.load(os.path.join(SBI_DIR, f'{split}_logit.npy'), mmap_mode='r')
        self.P = self.img.shape[0]
        self.N = 2 * self.P
        self.ix = {'label': np.tile(np.array([0, 1], np.int8), self.P), 'vid': np.arange(self.N, dtype=np.int32)}

    def batch(self, rows, flips=None):
        rows = np.asarray(rows)
        p, w = rows // 2, rows % 2
        x = torch.from_numpy(self.img[p, w]).cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255.)
        return (x, torch.from_numpy(self.b3[p, w]).cuda(), torch.from_numpy(self.lg[p, w]).cuda(),
                torch.from_numpy(w.astype(np.int64)).cuda())


def epoch_plan(P, seed, epoch):
    """PROTOCOL §14: 고정 쌍 순열에서 epoch마다 7,200쌍을 차례로, 순서·색 계수는 (seed, epoch)로만"""
    perm = np.random.default_rng(SCHEDULE_SEED).permutation(P)
    chunk = perm[((epoch - 1) * PAIRS_PER_EPOCH + np.arange(PAIRS_PER_EPOCH)) % P]
    rows = np.concatenate([chunk * 2, chunk * 2 + 1])
    rows = rows[np.random.default_rng([seed, epoch, 0]).permutation(len(rows))]
    r2 = np.random.default_rng([seed, epoch, 2])
    n = len(rows)
    return rows, r2.uniform(0.8, 1.2, n), r2.uniform(0.8, 1.2, n), r2.uniform(0.9, 1.1, n)


def s8_code_sha():
    d = os.path.join(C.EXP, 'code')
    return C.sha256_files([os.path.join(d, f) for f in C.CODE_FILES + ('run_train_sbi.py',)])


def sbi_cache_sha():
    return C.sha256_files([os.path.join(SBI_DIR, f) for f in ('train_meta.npy', 'val_meta.npy',
                                                               'train_meta_rows.npz', 'val_meta_rows.npz')])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', required=True, choices=['P2', 'P3', 'P4a', 'P2a', 'P3a', 'P3b', 'P1p', 'P2b'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--kind', default='sbi', choices=['sbi', 'debug'])
    ap.add_argument('--run_id', default=None)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--no_test', action='store_true')
    args = ap.parse_args()
    cfg = E.setting_cfg(args.setting)
    run_id = args.run_id or f'{args.setting}_sbi_s{args.seed}'
    rdir = os.path.join(C.RUNS, run_id)
    os.makedirs(rdir, exist_ok=True)
    if os.path.exists(os.path.join(rdir, 'DONE')):
        print(f'[{run_id}] 이미 완료 → 건너뜀')
        return
    R.set_determinism()
    t_start = time.time()
    started = time.strftime('%Y-%m-%d %H:%M:%S')
    TR, VS, VF = SBIPairs('train'), SBIPairs('val'), R.Split('val')
    em, init_msg = R.build_model(cfg, args.seed, cfg['init'])
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
        print(f'[{run_id}] epoch {st["epoch"]}까지 완료된 상태에서 이어받기', flush=True)
    else:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    cs = hashlib.sha256(json.dumps({**cfg, 'train': 'sbi', 'epochs': args.epochs, 'pairs_per_epoch': PAIRS_PER_EPOCH,
                                    'lr': R.LR, 'wd': R.WD, 'bs': R.TRAIN_BS}, sort_keys=True, default=str)
                        .encode()).hexdigest()[:12]
    config = {'run_id': run_id, 'kind': args.kind, 'setting': args.setting, 'cfg': cfg, 'train_data': 'sbi',
              'pairs_per_epoch': PAIRS_PER_EPOCH, 'schedule_seed': SCHEDULE_SEED, 'n_train_pairs': TR.P,
              'n_val_pairs': VS.P, 'selection': 'SBI val val_loss_bal (FF++ val 기록만)', 'seed': args.seed,
              'epochs': args.epochs, 'init': cfg['init'], 'init_msg': init_msg, 'lr': R.LR, 'wd': R.WD,
              'eta_min': R.ETA_MIN, 'train_bs': R.TRAIN_BS, 'eval_bs': R.EVAL_BS, 'loss_w_detail': R.LOSS_W_DETAIL,
              'clip': R.CLIP, 'steps_per_epoch': 2 * PAIRS_PER_EPOCH // R.TRAIN_BS, 'max_steps': args.max_steps,
              'code_sha': s8_code_sha(), 'cfg_sha': cs, 'index_sha': C.index_sha(), 'cache_sha': C.cache_sha(),
              'sbi_cache_sha': sbi_cache_sha(), 'versions': C.env_versions(), 'started': started,
              'n_trainable': sum(p.numel() for p in params)}
    json.dump(config, open(os.path.join(rdir, 'config.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'[{run_id}] SBI 학습 | {init_msg} | epoch {start_epoch}→{args.epochs}', flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        em.train()
        rows, bf, cf, sf = epoch_plan(TR.P, args.seed, epoch)
        n_steps = len(rows) // R.TRAIN_BS
        if args.max_steps:
            n_steps = min(n_steps, args.max_steps)
        acc_l = torch.zeros(3, device='cuda', dtype=torch.float64)
        n_seen, probs, labs = 0, [], []
        for st_i in range(n_steps):
            sl = slice(st_i * R.TRAIN_BS, (st_i + 1) * R.TRAIN_BS)
            x, b3, lg, y = TR.batch(rows[sl])
            x = E.color_jitter(x, bf[sl], cf[sl], sf[sl])
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                out = em(x, b3, lg)
                lf = F.cross_entropy(out['pred'].float(), y)
                ld = F.cross_entropy(out['detail_pred'].float(), y)
                loss = lf + R.LOSS_W_DETAIL * ld
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
        vm = R.val_metrics(R.predict([em], VS)[0], VS.ix)                         # 선택용 (SBI val)
        vf = {f'ffval_{k[4:]}': v for k, v in R.val_metrics(R.predict([em], VF)[0], VF.ix).items()}   # 기록용
        row = {'epoch': epoch, 'lr': lr_now, 'train_loss': tl, 'train_loss_fused': tf, 'train_loss_detail': td,
               'train_acc': float(((p_tr >= 0.5) == y_tr).mean() * 100), 'train_auc': roc_auc_score(y_tr, p_tr) * 100,
               **vm, **vf, 'sec': time.time() - t0, 'grad_scale': scaler.get_scale()}
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
        print(f'[{run_id}] ep {epoch:2d} | train loss {tl:.4f} acc {row["train_acc"]:.2f} | SBIval bal {vm["val_loss_bal"]:.4f} '
              f'auc {vm["val_auc_frame"]:.2f} | FFval bal {vf["ffval_loss_bal"]:.4f} aucV {vf["ffval_auc_video"]:.2f} '
              f'bacc {vf["ffval_bacc05"]:.2f} | {row["sec"]:.0f}s', flush=True)

    train_min = (time.time() - t_start) / 60
    if args.no_test:
        print(f'[{run_id}] 테스트 생략(디버그), {train_min:.1f}분', flush=True)
        return
    del TR
    models, tags = [], ['best', 'last']
    for tag in tags:
        m = E.ExpModel(cfg, seed=args.seed).cuda()
        m.load_state_dict(torch.load(os.path.join(rdir, f'ckpt_{tag}.pt'), map_location='cpu',
                                     weights_only=False)['model'])
        models.append(m)
    TE, CD = R.Split('test'), R.Split('cdf')
    o_te, o_cd = R.predict(models, TE), R.predict(models, CD)
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
        C.registry_append({'run_id': run_id, 'kind': 'sbi', 'setting': args.setting, 'data_pct': 100,
                           'seed': args.seed, 'epochs': args.epochs, 'status': 'done', 'best_epoch': best['epoch'],
                           'best_val_loss': best['val_loss_bal'], 'last_val_loss': curves[-1]['val_loss_bal'],
                           **C.metric_row('best', metrics['best']), **C.metric_row('last', metrics['last']),
                           'train_min': train_min, 'started': started, 'finished': finished,
                           'code_sha': config['code_sha'], 'cfg_sha': cs, 'index_sha': config['index_sha'],
                           'cache_sha': config['cache_sha'], 'torch': torch.__version__,
                           'notes': f'train=SBI pairs (sbi_cache={config["sbi_cache_sha"]}); select=SBI val'})
    open(os.path.join(rdir, 'DONE'), 'w').write(finished + '\n')
    print(f'[{run_id}] 완료 {train_min:.1f}분 | best ep {best["epoch"]}: FF++ AUC {metrics["best"]["ffpp_auc"]:.2f} '
          f'CDF {metrics["best"]["cdf_auc"]:.2f} | last: FF++ {metrics["last"]["ffpp_auc"]:.2f} '
          f'CDF {metrics["last"]["cdf_auc"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
