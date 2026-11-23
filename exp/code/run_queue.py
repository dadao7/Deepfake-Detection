"""학습 큐 러너 (분리 실행용). registry.csv + runs/{id}/DONE을 보고 끝난 실행은 건너뛴다.
  python code/run_queue.py queues/<이름>.json
큐 파일: [{"setting":"P3","data":100,"seed":0,"kind":"main","mode":"cache","init":null,"epochs":20,"run_id":null}, ...]
실패하면 한 번 이어받기 재시도, 또 실패하면 FAILED 표시 후 큐를 멈춘다(체계적 오류가 다른 실행에 번지지 않게).
진행 기록: logs/queue_<이름>.log (한 실행당 시작·끝 두 줄)"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def run_id_of(j):
    if j.get('script') == 'run_train_sbi2.py':
        return j.get('run_id') or f"X{j['tag']}_s{j.get('seed', 0)}"
    if j.get('script') == 'run_train_distill.py':
        return j.get('run_id') or f"D{j.get('init', 'scratch')[0]}_b{j.get('beta', 1.0):g}_s{j.get('seed', 0)}"
    if j.get('script') == 'run_train_anchor.py':
        return j.get('run_id') or f"P4c_l{j['lam']:g}_s{j.get('seed', 0)}"
    if j.get('script') == 'run_train_mix.py':
        return j.get('run_id') or f"{j['cfg']}m_s{j.get('seed', 0)}"
    if j.get('script') == 'run_train_unfreeze.py':
        return j.get('run_id') or f"{j['cfg']}{'f' if j['data'] == 'ffpp' else 's'}_s{j.get('seed', 0)}"
    if j.get('script') == 'run_train_sbi.py':
        return j.get('run_id') or f"{j['setting']}_sbi_s{j.get('seed', 0)}"
    return j.get('run_id') or f"{j['setting']}_d{j.get('data', 100)}_s{j.get('seed', 0)}"


def main():
    qpath = sys.argv[1]
    name = os.path.splitext(os.path.basename(qpath))[0]
    jobs = json.load(open(qpath, encoding='utf-8'))
    qlog = os.path.join(C.LOGS, f'queue_{name}.log')

    def log(msg):
        with open(qlog, 'a', encoding='utf-8') as f:
            f.write(time.strftime('%m-%d %H:%M:%S ') + msg + '\n')
    log(f'큐 시작: {len(jobs)}개')
    for j in jobs:
        rid = run_id_of(j)
        rdir = os.path.join(C.RUNS, rid)
        if os.path.exists(os.path.join(rdir, 'DONE')) and (j.get('kind') == 'debug' or rid in C.registry_ids()):
            log(f'건너뜀(완료): {rid}')
            continue
        script = j.get('script', 'run_train.py')
        if script == 'run_train_sbi2.py':         # TASKS6 SBI 전문가 개선
            cmd = [C.PY, '-X', 'utf8', os.path.join(C.EXP, 'code', script), '--tag', j['tag'], '--init', j.get('init', 'theta0'),
                   '--ema', str(j.get('ema', 0.0)), '--lr', str(j.get('lr', 5e-5)), '--seed', str(j.get('seed', 0)),
                   '--epochs', str(j.get('epochs', 20)), '--run_id', rid]
        elif script == 'run_train_distill.py':      # TASKS5 M3 상보 전문가 증류
            cmd = [C.PY, '-X', 'utf8', os.path.join(C.EXP, 'code', script), '--seed', str(j.get('seed', 0)),
                   '--init', j.get('init', 'scratch'), '--beta', str(j.get('beta', 1.0)), '--T', str(j.get('T', 2.0)),
                   '--epochs', str(j.get('epochs', 20)), '--run_id', rid]
        elif script == 'run_train_anchor.py':       # TASKS4 T4-6 heatmap anchor
            cmd = [C.PY, '-X', 'utf8', os.path.join(C.EXP, 'code', script), '--lam', str(j['lam']),
                   '--seed', str(j.get('seed', 0)), '--epochs', str(j.get('epochs', 20)), '--run_id', rid]
        elif script == 'run_train_mix.py':          # TASKS4 T4-5 혼합 학습
            cmd = [C.PY, '-X', 'utf8', os.path.join(C.EXP, 'code', script), '--cfg', j['cfg'],
                   '--seed', str(j.get('seed', 0)), '--epochs', str(j.get('epochs', 20)), '--run_id', rid]
        elif script == 'run_train_unfreeze.py':     # S9 baseline 일부 풀기 (PROTOCOL §15)
            cmd = [C.PY, '-X', 'utf8', os.path.join(C.EXP, 'code', script), '--cfg', j['cfg'], '--data', j['data'],
                   '--seed', str(j.get('seed', 0)), '--epochs', str(j.get('epochs', 20)), '--run_id', rid]
        elif script == 'run_train_sbi.py':          # S8 SBI 학습 프로토콜 (PROTOCOL §14)
            cmd = [C.PY, '-X', 'utf8', os.path.join(C.EXP, 'code', script), '--setting', j['setting'],
                   '--seed', str(j.get('seed', 0)), '--epochs', str(j.get('epochs', 20)), '--run_id', rid]
        else:
            cmd = [C.PY, '-X', 'utf8', os.path.join(C.EXP, 'code', 'run_train.py'), '--setting', j['setting'],
                   '--data', str(j.get('data', 100)), '--seed', str(j.get('seed', 0)),
                   '--epochs', str(j.get('epochs', 20)), '--kind', j.get('kind', 'main'),
                   '--mode', j.get('mode', 'cache'), '--run_id', rid]
            if j.get('init'):
                cmd += ['--init', j['init']]
        os.makedirs(rdir, exist_ok=True)
        ok = False
        for attempt in (1, 2):
            t0 = time.time()
            log(f'시작: {rid} (시도 {attempt})')
            with open(os.path.join(rdir, 'train.log'), 'a', encoding='utf-8') as f:
                p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=C.EXP)
            if p.returncode == 0 and os.path.exists(os.path.join(rdir, 'DONE')):
                log(f'끝: {rid} {(time.time() - t0) / 60:.1f}분')
                ok = True
                break
            log(f'실패: {rid} 종료코드 {p.returncode} (train.log tail 확인)')
        if not ok:
            open(os.path.join(rdir, 'FAILED'), 'w').write(time.strftime('%Y-%m-%d %H:%M:%S\n'))
            log(f'큐 멈춤: {rid} 두 번 실패')
            sys.exit(1)
    log('큐 끝')


if __name__ == '__main__':
    main()
