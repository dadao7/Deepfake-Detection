"""S1 게이트: 캐시(autocast fp16)로 P0 재평가 → 영상 AUC가 95.22 ± 0.3, 프레임 확률이 ref/p0/p0_frames.npy(fp32)와 근접.
통과하면 registry.csv에 P0(kind=ref) 한 줄. 결과: reports/p0_cache_metrics.json"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def pearson(a, b):
    """np.corrcoef는 이 PC의 numpy BLAS가 깨져 프로세스가 죽음(0xC06D007F) → 원소별 연산으로 계산"""
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def softmax_fake(logit):
    z = logit - logit.max(1, keepdims=True)
    e = np.exp(z)
    return (e[:, 1] / e.sum(1)).astype(np.float64)


def main():
    te = C.load_index('test')
    cdf = C.load_index('cdf')
    p_te = softmax_fake(np.load(os.path.join(C.BASE_DIR, 'test_logit.npy')))
    p_cdf = softmax_fake(np.load(os.path.join(C.BASE_DIR, 'cdf_logit.npy')))
    p0 = np.load(os.path.join(C.EXP, 'ref', 'p0', 'p0_frames.npy')).astype(np.float64)
    ref = json.load(open(os.path.join(C.EXP, 'ref', 'p0', 'p0_metrics.json'), encoding='utf-8'))

    m_te, vs = C.ffpp_metrics(p_te, te)
    m_cdf, _ = C.cdf_metrics(p_cdf, cdf)
    d = np.abs(p_te - p0)
    m_p0ref, _ = C.ffpp_metrics(p0, te)        # 같은 지표 코드로 P0 fp32 값 재계산(대조용)
    rep = {'cache_fp16': {**m_te, **m_cdf},
           'p0_fp32_same_code': m_p0ref,
           'frame_prob_vs_p0': {'mean_abs_diff': float(d.mean()), 'max_abs_diff': float(d.max()),
                                'p99_abs_diff': float(np.quantile(d, 0.99)), 'pearson': pearson(p_te, p0),
                                'frame_auc_fp32': roc_auc_score(te['label'], p0) * 100},
           'ref_p0_metrics_json_AUC': ref['영상 단위(수정)']['AUC']}
    auc_diff = m_te['ffpp_auc'] - ref['영상 단위(수정)']['AUC']
    rep['gate'] = {'video_auc_diff': auc_diff, 'pass_auc': abs(auc_diff) <= 0.3,
                   'pass_frames': bool(d.mean() < 0.005 and d.max() < 0.05)}
    os.makedirs(C.REPORTS, exist_ok=True)
    json.dump(rep, open(os.path.join(C.REPORTS, 'p0_cache_metrics.json'), 'w', encoding='utf-8'), indent=1,
              ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))

    if rep['gate']['pass_auc'] and rep['gate']['pass_frames'] and 'P0_ref' not in C.registry_ids():
        m = {**m_te, **m_cdf}
        row = {'run_id': 'P0_ref', 'kind': 'ref', 'setting': 'P0', 'data_pct': '', 'seed': '', 'epochs': 0,
               'status': 'done', 'best_epoch': '', **C.metric_row('best', m), **C.metric_row('last', m),
               'finished': time.strftime('%Y-%m-%d %H:%M'), 'code_sha': C.code_sha(), 'index_sha': C.index_sha(),
               'cache_sha': C.cache_sha(), 'torch': C.env_versions()['torch'],
               'notes': 'baseline only; cache autocast fp16; best=last'}
        C.registry_append(row)
        print('registry에 P0_ref 기록')


if __name__ == '__main__':
    main()
