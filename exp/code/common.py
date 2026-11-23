"""공통 경로·상수·저자 baseline 로드·지표·해시 (DeepSight 통제 비교 실험, PROTOCOL.md 기준)
원본(C:\\m2f2\\code, M2F2_Det, weights)은 읽기만 한다."""
import hashlib
import json
import os
import platform
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# ── 경로 ─────────────────────────────────────────────────────────────
EXP = r'C:\m2f2\exp'
M2F2 = r'C:\m2f2\M2F2_Det'
ORIG_CODE = r'C:\m2f2\code'
WEIGHTS = r'C:\m2f2\weights'
DATA = r'C:\m2f2\data'
FFPP_DIR = os.path.join(DATA, 'FF++')
TEST_DIR = os.path.join(DATA, 'test_only')
CDF_DIR = os.path.join(DATA, 'Celeb-DF-v2')
SPLIT_DIR = os.path.join(M2F2, 'utils', 'FFPP_split')
CACHE = os.path.join(EXP, 'cache')
IDX_DIR = os.path.join(CACHE, 'index')
FRAME_DIR = os.path.join(CACHE, 'frames')
BASE_DIR = os.path.join(CACHE, 'base')
RUNS = os.path.join(EXP, 'runs')
REPORTS = os.path.join(EXP, 'reports')
LOGS = os.path.join(EXP, 'logs')
REGISTRY = os.path.join(EXP, 'registry.csv')
PY = r'C:\Users\COM\miniconda3\envs\m2f2\python.exe'

# ── 데이터 상수 (P0과 같은 종류 순서) ────────────────────────────────
CATS = ['original', 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Face2Face']
METHODS = CATS[1:]
METHOD_SHORT = {'Deepfakes': 'DF', 'Face2Face': 'F2F', 'FaceSwap': 'FS', 'NeuralTextures': 'NT'}
N_TRAIN_FR, N_VAL_FR, N_TEST_FR = 10, 8, 100
PAIR_PERM_SEED = 2026          # nested 부분집합용 쌍 순열 (PROTOCOL §3.3)
DATA_PCTS = (25, 50, 100)


def h5_path(cat, split):
    if split == 'test':
        return os.path.join(TEST_DIR, f'FF++_{cat}_c40_test_only.h5')
    return os.path.join(FFPP_DIR, f'FF++_{cat}_c40.h5')


def load_pairs(split):
    with open(os.path.join(SPLIT_DIR, f'{split}.json')) as f:
        return json.load(f)


def videos_of_pairs(pairs, cat):
    """P0과 같은 순서: 진짜는 쌍마다 a, b / 가짜는 쌍마다 a_b, b_a"""
    if cat == 'original':
        return [x for p in pairs for x in p]
    return [k for p in pairs for k in (f'{p[0]}_{p[1]}', f'{p[1]}_{p[0]}')]


def frame_idx(n, k):
    """k등분 구간 가운데 (PROTOCOL §3.2)"""
    return [int((j + 0.5) * n / k) for j in range(k)]


# ── 저자 baseline (ref/p0_eval.py와 같은 로드) ───────────────────────
def _patch_torch_load():
    if getattr(torch.load, '_exp_patched', False):
        return
    orig = torch.load

    def patched(*a, **k):
        k.pop('weights_only', None)
        return orig(*a, **k, weights_only=False)
    patched._exp_patched = True
    torch.load = patched


def remap(k):
    """transformers 5.x 키 이름 맞추기 (p0_eval.py / train_improved.load_model 규칙)"""
    for a in ('module.', 'baseline.', 'vision_tower.'):
        if k.startswith(a):
            k = k.replace(a, '', 1)
    if k.startswith('vision_model.'):
        k = k.replace('vision_model.', '', 1)
    return k.replace('.text_model.', '.').replace('.vision_model.', '.')


def load_baseline(dev='cuda', verbose=True):
    """저자 M2F2Det + current_model_180.pth, eval 모드, 파라미터 고정. 반환: M2F2Det 모듈(DataParallel 아님)"""
    _patch_torch_load()
    if M2F2 not in sys.path:
        sys.path.insert(0, M2F2)
    cwd = os.getcwd()
    os.chdir(M2F2)
    try:
        from sequence.models.M2F2_Det.models.model import M2F2Det
        model = M2F2Det(clip_text_encoder_name="openai/clip-vit-large-patch14-336",
                        clip_vision_encoder_name="openai/clip-vit-large-patch14-336",
                        deepfake_encoder_name='efficientnet_b4', hidden_size=1792)
        vt = torch.load(os.path.join(WEIGHTS, 'vision_tower.pth'), map_location='cpu')
        r1 = model.clip_vision_encoder.model.load_state_dict({remap(k): v for k, v in vt.items()}, strict=False)
        ck = torch.load(os.path.join(WEIGHTS, 'current_model_180.pth'), map_location='cpu')
        r2 = model.load_state_dict({remap(k): v for k, v in ck['model_state_dict'].items()}, strict=False)
        if verbose:
            print(f'[baseline] vision_tower missing {len(r1.missing_keys)} unexpected {len(r1.unexpected_keys)} / '
                  f'저자 가중치 missing {len(r2.missing_keys)} unexpected {len(r2.unexpected_keys)}', flush=True)
        del vt, ck
    finally:
        os.chdir(cwd)
    model = model.to(dev).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def baseline_forward(model, x, amp=True):
    """x: (B,3,224,224) float [0,1] → (logit fp32 (B,2), b_3 fp16 (B,272,11,11)). 저자 forward 그대로 호출"""
    with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
        out = model(x, return_dict=True)
    logit = out['pred'].float()
    b3 = model.block_outputs['b_3']
    return logit, b3.half() if amp else b3.float()


# ── 지표 (PROTOCOL §8) ───────────────────────────────────────────────
def trimmed_mean(p, t):
    s = np.sort(np.asarray(p, dtype=np.float64))
    return s[t:len(s) - t].mean() if len(s) > 2 * t else s.mean()


def best_acc(scores, labels):
    """테스트에서 고른 최적 기준값 ACC (논문 비교용, 저자 방식 0~1 0.0001 간격)"""
    s = np.asarray(scores)[:, None]
    l = np.asarray(labels)[:, None].astype(bool)
    ths = np.arange(0, 1.0001, 0.0001)[None, :]
    return float(((s >= ths) == l).mean(0).max())


def summarize(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    pred = scores >= 0.5
    tpr = pred[labels == 1].mean()
    tnr = (~pred[labels == 0]).mean()
    return {'auc': roc_auc_score(labels, scores) * 100, 'acc05': (pred == labels).mean() * 100,
            'bacc05': (tpr + tnr) / 2 * 100, 'fpr': (1 - tnr) * 100, 'fnr': (1 - tpr) * 100,
            'accbest': best_acc(scores, labels) * 100}


def video_scores(frame_p, vid_ids, trim):
    """frame_p: 프레임 확률, vid_ids: 같은 길이의 영상 번호(0..V-1). trim: 'ffpp'(10개) 또는 'frac'(floor(0.1n))"""
    frame_p = np.asarray(frame_p, dtype=np.float64)
    vid_ids = np.asarray(vid_ids)
    V = int(vid_ids.max()) + 1
    out = np.zeros(V)
    for v in range(V):
        p = frame_p[vid_ids == v]
        t = 10 if trim == 'ffpp' else int(0.1 * len(p))
        out[v] = trimmed_mean(p, t)
    return out


def ffpp_metrics(frame_p, index):
    """index: test 인덱스 dict(arrays: vid, label, cat). 반환: PROTOCOL §8.1 지표"""
    vid, lab, cat = index['vid'], index['label'], index['cat']
    vs = video_scores(frame_p, vid, 'ffpp')
    V = len(vs)
    v_lab = np.zeros(V, int)
    v_cat = np.empty(V, object)
    v_lab[vid] = lab
    v_cat[vid] = cat
    m = {f'ffpp_{k}': v for k, v in summarize(vs, v_lab).items()}
    m['ffpp_frame_auc'] = roc_auc_score(lab, frame_p) * 100
    for meth in METHODS:
        sel = (v_cat == 'original') | (v_cat == meth)
        m[f'auc_{METHOD_SHORT[meth]}'] = roc_auc_score(v_lab[sel], vs[sel]) * 100
    return m, vs


def cdf_metrics(frame_p, index):
    vid, lab = index['vid'], index['label']
    vs = video_scores(frame_p, vid, 'frac')
    V = len(vs)
    v_lab = np.zeros(V, int)
    v_lab[vid] = lab
    return {f'cdf_{k}': v for k, v in summarize(vs, v_lab).items()}, vs


# ── 해시·버전 ─────────────────────────────────────────────────────────
def sha256_file(path, n=None):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()[:n] if n else h.hexdigest()


def sha256_files(paths, n=12):
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(os.path.basename(p).encode())
        with open(p, 'rb') as f:
            h.update(f.read())
    return h.hexdigest()[:n]


CODE_FILES = ('common.py', 'expmodel.py', 'run_train.py')   # 학습·평가 결과에 영향을 주는 파일만 (분석 스크립트 추가로 바뀌지 않게)


def code_sha():
    d = os.path.join(EXP, 'code')
    return sha256_files([os.path.join(d, f) for f in CODE_FILES])


def index_sha():
    return sha256_files([os.path.join(IDX_DIR, f) for f in os.listdir(IDX_DIR)
                         if f.endswith(('.json', '.npz', '.csv')) and f != 'index_summary.json'])


def cache_sha():
    p = os.path.join(CACHE, 'MANIFEST.json')
    return sha256_file(p, 12) if os.path.exists(p) else ''


def env_versions():
    import h5py
    import sklearn
    import torchvision
    import transformers
    return {'python': platform.python_version(), 'torch': torch.__version__, 'torchvision': torchvision.__version__,
            'transformers': transformers.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__,
            'h5py': h5py.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}


def load_index(split):
    """cache/index/{split}.npz → dict of arrays"""
    z = np.load(os.path.join(IDX_DIR, f'{split}.npz'), allow_pickle=False)
    return {k: z[k] for k in z.files}


# ── registry.csv (PROTOCOL §9.2, 실행 1회 1줄) ───────────────────────
def registry_header():
    with open(REGISTRY, encoding='utf-8') as f:
        return f.readline().strip().split(',')


def registry_ids():
    import csv
    with open(REGISTRY, encoding='utf-8') as f:
        return {r['run_id'] for r in csv.DictReader(f)}


def registry_append(row):
    cols = registry_header()
    unknown = set(row) - set(cols)
    assert not unknown, f'registry에 없는 열: {unknown}'
    assert row['run_id'] not in registry_ids(), f"이미 있는 run_id: {row['run_id']}"

    def fmt(v):
        if v is None:
            return ''
        if isinstance(v, (float, np.floating)):
            return f'{float(v):.4f}'
        s = str(v)
        return '"' + s.replace('"', "'") + '"' if (',' in s or '"' in s) else s
    with open(REGISTRY, 'a', encoding='utf-8', newline='') as f:
        f.write(','.join(fmt(row.get(c)) for c in cols) + '\n')


def metric_row(prefix, m):
    """평가 지표 dict → registry 열 이름(prefix_*)"""
    keys = ['ffpp_auc', 'ffpp_acc05', 'ffpp_bacc05', 'ffpp_fpr', 'ffpp_fnr', 'ffpp_accbest', 'ffpp_frame_auc',
            'auc_DF', 'auc_F2F', 'auc_FS', 'auc_NT', 'cdf_auc', 'cdf_acc05', 'cdf_bacc05', 'cdf_fpr', 'cdf_fnr']
    return {f'{prefix}_{k}': m.get(k) for k in keys}
