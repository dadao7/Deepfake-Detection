"""TASKS4 공통 (3·4번): DFD 신원 분할, 점수 은행(모든 모델의 영상 점수), 지표·bootstrap.
선택 전용: FF++ val, DFD-val / 보고 전용: FF++ test, Celeb-DF(cdf), FaceShifter(fsh), DFD-test."""
import json
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SPLIT_PATH = os.path.join(C.IDX_DIR, 'dfd_split.json')
BANK_PATH = os.path.join(C.CACHE, 'scorebank.npz')
SELECT_SETS = ('ffval', 'dfdval')
REPORT_SETS = ('fftest', 'cdf', 'fsh', 'dfdtest')


def dfd_split(seed=2026):
    """배우 28명을 14/14로. 진짜: 배우 쪽, 가짜: 두 배우가 같은 쪽일 때만(섞인 가짜는 제외)"""
    if os.path.exists(SPLIT_PATH):
        return json.load(open(SPLIT_PATH))
    d = json.load(open(os.path.join(C.DATA, 'dl_meta', 'deepfake_detection_filenames.json')))
    actors = sorted({n.split('__')[0] for n in d['actors']})
    perm = list(np.random.default_rng(seed).permutation(actors))
    A, B = set(perm[:14]), set(perm[14:])
    out = {'seed': seed, 'val_actors': sorted(A), 'test_actors': sorted(B), 'val': [], 'test': [], 'dropped': []}
    for n in d['actors']:
        out['val' if n.split('__')[0] in A else 'test'].append(n)
    for n in d['DeepFakesDetection']:
        ids = set(n.split('__')[0].split('_'))
        if ids <= A:
            out['val'].append(n)
        elif ids <= B:
            out['test'].append(n)
        else:
            out['dropped'].append(n)
    json.dump(out, open(SPLIT_PATH, 'w'), indent=0)
    return out


def summarize(scores, labels):
    return C.summarize(scores, labels)


def auc(labels, scores):
    return roc_auc_score(labels, scores) * 100


def boot_delta(labels, s_new_list, s_base, B=2000, seed=0):
    """영상 단위 짝 bootstrap: AUC(새 점수, seed 평균) − AUC(기준). → (점 추정, lo, hi)"""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n = len(labels)
    s_new_list = [np.asarray(s) for s in s_new_list]
    point = np.mean([auc(labels, s) for s in s_new_list]) - auc(labels, s_base)
    ds = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if labels[i].min() == labels[i].max():
            continue
        ds.append(np.mean([auc(labels[i], s[i]) for s in s_new_list]) - auc(labels[i], s_base[i]))
    lo, hi = np.quantile(ds, [0.025, 0.975])
    return point, lo, hi


def load_bank():
    z = np.load(BANK_PATH, allow_pickle=True)
    return {k: z[k] for k in z.files}


def model_keys(bank):
    return [k[len('score|fftest|'):] for k in bank if k.startswith('score|fftest|')]
