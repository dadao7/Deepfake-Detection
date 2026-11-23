"""TASKS5 M3: 상보 전문가 증류. 두 데이터 전문가(같은 seed의 P3 last = FF++ 전문가, U3s best = SBI 전문가)의
결합 logit z_T = (1−w)·z_P3 + w·z_U3s (w = 0.7, TASKS4에서 선택 세트로 고른 값)를 단일 학생 모델(U3 구조)에 증류한다.
근거: 혼합 학습(U3m, hard label)은 상보성을 잃었고(FaceShifter↓), 따로 학습한 전문가 결합은 이득(T4 ④).
데이터·스케줄은 혼합 학습과 같다: 배치 16 = FF++ 8(S4 스케줄) + SBI 8(S8 스케줄), epoch당 900스텝, 20 epoch, AdamW lr 5e-5.
손실 = CE(결합) + 0.15·CE(detail) + β·T²·KL(softmax(z_T/T) ‖ softmax(z_S/T)), 기본 β=1, T=2. 교사는 eval 모드·no_grad.
--init: scratch(U3 초기값 = 저자 상단 + SBI 분기) | merge(TASKS5 M1에서 고른 병합 모델에서 시작)
선택: (FF++ val 균형 CE + SBI val 균형 CE)/2 최저(best) + last. registry kind=distill, 실행 이름 D{init}_b{β}_s{seed}.
  python code/run_train_distill.py --seed 0 [--init merge] [--beta 1 --T 2]"""
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
import run_train_sbi as S  # noqa: E402
import run_train_unfreeze as U  # noqa: E402
import t5_merge as M  # noqa: E402

W_ENS = 0.7


def code_sha():
    d = os.path.join(C.EXP, 'code')
    return C.sha256_files([os.path.join(d, f) for f in C.CODE_FILES + ('run_train_sbi.py', 'basetop.py', 'run_train_unfreeze.py',
                                                                         't5_merge.py', 'run_train_distill.py')])


def teachers(seed):
    s0 = M.init_state(seed)
    out = []
    for key in (('P3', 'last'), ('U3s', 'best')):
        m = M.model_from(M.expert_state(*key, seed, s0), seed)
        for p in m.parameters():
            p.requires_grad = False
        out.append(m.eval())
    return out


def merged_init(seed):
    ch = json.load(open(os.path.join(C.REPORTS, 't5_merge_choice.json')))
    name = ch['best_lin'][1]                     # lin|{쌍}|t{t}
    pname, t = name.split('|')[1], float(name.split('|')[2][1:])
    A, B = M.PAIRS[pname]
    s0 = M.init_state(seed)
    return M.combine(s0, M.expert_state(*A, seed, s0), M.expert_state(*B, seed, s0), 1 - t, t), name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--init', default='scratch', choices=['scratch', 'merge'])
    ap.add_argument('--beta', type=float, default=1.0)
    ap.add_argument('--T', type=float, default=2.0)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--run_id', default=None)
    ap.add_argument('--max_steps', type=int, default=0)
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()
    setting = f'D{args.init[0]}_b{args.beta:g}'
    run_id = args.run_id or f'{setting}_s{args.seed}'
    rdir = os.path.join(C.RUNS, run_id)
    os.makedirs(rdir, exist_ok=True)
    if os.path.exists(os.path.join(rdir, 'DONE')):
        print(f'[{run_id}] 이미 완료 → 건너뜀')
        return
    R.set_determinism()
    t_start, started = time.time(), time.strftime('%Y-%m-%d %H:%M:%S')
    TF, TS = U.FFData('train', flip=True), U.SBIData('train')
    sched = dict(np.load(os.path.join(C.IDX_DIR, 'schedule_d100.npz')))
    VF, VS = U.FFData('val'), U.SBIData('val')
    tA, tB = teachers(args.seed)
    model = U.UModel('U3', args.seed)
    init_msg = model.msg
    if args.init == 'merge':
        sd, name = merged_init(args.seed)
        model.load_state_dict(sd)
        init_msg += f' | 병합 초기화 {name}'
    model = model.cuda()
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
    else:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    cs = hashlib.sha256(json.dumps({'init': args.init, 'beta': args.beta, 'T': args.T, 'w': W_ENS, 'epochs': args.epochs},
                                   sort_keys=True).encode()).hexdigest()[:12]
    config = {'run_id': run_id, 'kind': 'distill', 'setting': setting, 'train_data': 'FF++ 8 + SBI 8 per batch',
              'teachers': f'P3_d100_s{args.seed} last ×{1 - W_ENS:.1f} + U3s_s{args.seed} best ×{W_ENS:.1f} (logit)',
              'loss': f'CE + {R.LOSS_W_DETAIL}·CE(detail) + {args.beta}·T²·KL(T={args.T})',
              'selection': '(FF++ val bal CE + SBI val bal CE)/2', 'seed': args.seed, 'epochs': args.epochs,
              'init': args.init, 'init_msg': init_msg, 'lr': R.LR, 'wd': R.WD, 'code_sha': code_sha(), 'cfg_sha': cs,
              'started': started, 'versions': C.env_versions(), 'n_trainable': sum(p.numel() for p in params)}
    json.dump(config, open(os.path.join(rdir, 'config.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    print(f'[{run_id}] 증류 학습 | {init_msg} | epoch {start_epoch}→{args.epochs}', flush=True)
    half, Tt = R.TRAIN_BS // 2, args.T
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        rf, ff, bf1, cf1, sf1 = R.epoch_plan(sched, args.seed, epoch)
        rs, bf2, cf2, sf2 = S.epoch_plan(TS.P, args.seed, epoch)
        n_steps = min(len(rf), len(rs)) // R.TRAIN_BS
        if args.max_steps:
            n_steps = min(n_steps, args.max_steps)
        acc_l = torch.zeros(4, device='cuda', dtype=torch.float64)
        n_seen, probs, labs = 0, [], []
        for st_i in range(n_steps):
            a = slice(st_i * half, (st_i + 1) * half)
            x1, b1, l1, y1, c1 = TF.batch(rf[a], ff[a])
            x1 = E.color_jitter(x1, bf1[a], cf1[a], sf1[a])
            x2, b2, l2, y2, c2 = TS.batch(rs[a])
            x2 = E.color_jitter(x2, bf2[a], cf2[a], sf2[a])
            x, b3, cls, y = torch.cat([x1, x2]), torch.cat([b1, b2]), torch.cat([c1, c2]), torch.cat([y1, y2])
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                zt = (1 - W_ENS) * tA(x, b3, cls)['pred'].float() + W_ENS * tB(x, b3, cls)['pred'].float()
            opt.zero_grad()
            with torch.autocast('cuda', dtype=torch.float16):
                out = model(x, b3, cls)
                zs = out['pred'].float()
                lf = F.cross_entropy(zs, y)
                ld = F.cross_entropy(out['detail_pred'].float(), y)
                lk = F.kl_div(F.log_softmax(zs / Tt, 1), F.softmax(zt / Tt, 1), reduction='batchmean') * Tt * Tt
                loss = lf + R.LOSS_W_DETAIL * ld + args.beta * lk
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, max_norm=R.CLIP)
            scaler.step(opt)
            scaler.update()
            acc_l += torch.stack([loss.detach(), lf.detach(), ld.detach(), lk.detach()]).double() * y.shape[0]
            n_seen += y.shape[0]
            probs.append(torch.softmax(zs.detach(), 1)[:, 1])
            labs.append(y)
        lr_now = opt.param_groups[0]['lr']
        sch.step()
        p_tr, y_tr = torch.cat(probs).cpu().numpy(), torch.cat(labs).cpu().numpy()
        tl, tf_, td, tk = (acc_l / n_seen).tolist()
        if not np.isfinite(tl):
            raise RuntimeError('비유한 손실')
        vf = R.val_metrics(U.predict([model], VF)[0], VF.ix)
        vs = R.val_metrics(U.predict([model], VS)[0], VS.ix)
        sel = (vf['val_loss_bal'] + vs['val_loss_bal']) / 2
        row = {'epoch': epoch, 'lr': lr_now, 'train_loss': tl, 'train_loss_fused': tf_, 'train_loss_detail': td,
               'train_loss_kd': tk, 'train_acc': float(((p_tr >= 0.5) == y_tr).mean() * 100),
               'train_auc': roc_auc_score(y_tr, p_tr) * 100, 'val_loss_bal': sel,
               **{f'ffval_{k[4:]}': v for k, v in vf.items()}, **{f'sbival_{k[4:]}': v for k, v in vs.items()},
               'sec': time.time() - t0}
        curves.append(row)
        if sel < best['val_loss_bal']:
            best = {'epoch': epoch, 'val_loss_bal': sel}
            torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(rdir, 'ckpt_best.pt'))
        if epoch == args.epochs:
            torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(rdir, 'ckpt_last.pt'))
        with open(os.path.join(rdir, 'curves.csv'), 'w', encoding='utf-8') as f:
            keys = list(curves[0].keys())
            f.write(','.join(keys) + '\n')
            for r in curves:
                f.write(','.join(f'{r[k]:.6g}' if isinstance(r[k], float) else str(r[k]) for k in keys) + '\n')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'sched': sch.state_dict(), 'epoch': epoch, 'curves': curves, 'best': best,
                    'rng': {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all(), 'py': random.getstate()}},
                   res_path)
        print(f'[{run_id}] ep {epoch:2d} | train {tl:.4f} (kd {tk:.4f}) | FFval bal {vf["val_loss_bal"]:.4f} aucV '
              f'{vf["val_auc_video"]:.2f} | SBIval bal {vs["val_loss_bal"]:.4f} auc {vs["val_auc_frame"]:.2f} | {row["sec"]:.0f}s',
              flush=True)
    if args.debug:
        return
    train_min = (time.time() - t_start) / 60
    del TF, TS
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
    C.registry_append({'run_id': run_id, 'kind': 'distill', 'setting': setting, 'data_pct': 100, 'seed': args.seed,
                       'epochs': args.epochs, 'status': 'done', 'best_epoch': best['epoch'], 'best_val_loss': best['val_loss_bal'],
                       'last_val_loss': curves[-1]['val_loss_bal'], **C.metric_row('best', metrics['best']),
                       **C.metric_row('last', metrics['last']), 'train_min': train_min, 'started': started,
                       'finished': finished, 'code_sha': config['code_sha'], 'cfg_sha': cs, 'index_sha': C.index_sha(),
                       'cache_sha': C.cache_sha(), 'torch': torch.__version__,
                       'notes': f'complementary expert distillation init={args.init} beta={args.beta} T={args.T} (TASKS5 M3)'})
    open(os.path.join(rdir, 'DONE'), 'w').write(finished + '\n')
    print(f'[{run_id}] 완료 {train_min:.1f}분 | best ep {best["epoch"]}: FF++ {metrics["best"]["ffpp_auc"]:.2f} CDF '
          f'{metrics["best"]["cdf_auc"]:.2f} | last: FF++ {metrics["last"]["ffpp_auc"]:.2f} CDF {metrics["last"]["cdf_auc"]:.2f}',
          flush=True)


if __name__ == '__main__':
    main()
