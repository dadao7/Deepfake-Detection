"""TASKS5 D: 병합·수프·증류 모델의 heatmap 위치 지표 (PROTOCOL §11과 같은 규칙, loc_eval.py 함수 재사용).
결과: reports/t5_loc.json  python code/t5_loc.py"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common as C  # noqa: E402
import expmodel as E  # noqa: E402
import loc_eval as L  # noqa: E402
import t5_soup as SP  # noqa: E402

SEEDS = (0, 1, 2)


def head_from(sd):
    h = E.HeatmapHead(272, 64).cuda()
    h.load_state_dict({k[len('branch.heatmap_head.'):]: v for k, v in sd.items() if k.startswith('branch.heatmap_head.')})
    return h


def main():
    val, test = L.load_split('val'), L.load_split('test')
    ch = json.load(open(os.path.join(C.REPORTS, 't5_merge_choice.json')))
    rows = []
    for tag, key in (('병합', 'best_lin'), ('병합TV', 'best_tv')):
        for sd in SEEDS:
            h = head_from(SP.merged_state(ch[key][1], sd))
            L.eval_model(tag, '-', sd, 'merge', L.heatmaps(h, val), L.heatmaps(h, test), val, test, rows)
        h = head_from(SP.soup(ch[key][1]))
        L.eval_model(f'수프({tag})', '-', 0, 'soup', L.heatmaps(h, val), L.heatmaps(h, test), val, test, rows)
    for rd in sorted(glob.glob(os.path.join(C.RUNS, 'D*_s*'))):
        if os.path.exists(os.path.join(rd, 'DONE')):
            st, sd = os.path.basename(rd).rsplit('_s', 1)
            for ck in ('best', 'last'):
                m = torch.load(os.path.join(rd, f'ckpt_{ck}.pt'), map_location='cpu', weights_only=False)['model']
                h = head_from(m)
                L.eval_model(f'증류:{st}', ck, int(sd), 'distill', L.heatmaps(h, val), L.heatmaps(h, test), val, test, rows)
    out = {}
    for r in rows:
        if r['method'] == 'all':
            out.setdefault(f"{r['model']}|{r['ckpt']}", []).append(r)
    summ = {k: {m: [float(np.mean([r[m] for r in rs])), float(np.std([r[m] for r in rs])), len(rs)]
                for m in ('cell_auc', 'bnd_auc', 'iou', 'max_hit', 'boundary_hit', 'crop_hit_k1', 'crop_hit_k3', 'crop_cov_k3')}
            for k, rs in out.items()}
    json.dump(summ, open(os.path.join(C.REPORTS, 't5_loc.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    for k, v in summ.items():
        print(k, ' '.join(f'{m}={v[m][0]:.2f}±{v[m][1]:.2f}' for m in ('boundary_hit', 'bnd_auc', 'crop_hit_k3', 'cell_auc')))


if __name__ == '__main__':
    main()
