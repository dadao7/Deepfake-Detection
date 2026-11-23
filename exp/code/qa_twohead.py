"""질문 방어 ②: 두 head 구조 (2026-10-21, 사용자 위임으로 진행). 기존 코드는 고치지 않고, run_train의 모델 생성만 이 파일 안에서 바꿔 끼운다.
설계: crop 위치 = 고정 seg head(SBI 사전학습 heatmap head 복사본, 가중치·BN 고정) → 위치가 무너질 수 없다.
      학습 head = 두 번째 heatmap head(SBI 초기화, detach 없음) → seg head가 고른 위치에서의 값만 내서
      crop 가중치(peak 가중 평균)와 동적 결합 MLP 입력으로 쓰인다. 나머지(데이터·손실·스케줄·epoch·seed)는 P3와 같다.
P3와의 차이 = 가중치·결합 입력이 고정 head 대신 학습 head에서 온다. P4a와의 차이 = 학습 head가 crop 위치를 바꾸지 못한다.
  python code/qa_twohead.py train --seed 0     (runs/TH_s{seed}, --no_test라 registry 미기록)
  python code/qa_twohead.py eval               (cache/qa_twohead_scores.npz)
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import common as C  # noqa: E402
import expmodel as E  # noqa: E402
import run_train as R  # noqa: E402
import t5_common as T5  # noqa: E402
from topk_crop import nms_heatmap  # noqa: E402

OUT = os.path.join(C.CACHE, 'qa_twohead_scores.npz')


class TwoHead(E.ExpModel):
    def load_init(self, which):
        msg = super().load_init(which)
        self.seg_head = copy.deepcopy(self.heatmap_head)
        for p in self.seg_head.parameters():
            p.requires_grad = False
        self.seg_head.eval()
        return msg + ' | seg head = SBI heatmap head 고정 복사본'

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'seg_head'):
            self.seg_head.eval()
        return self

    def forward(self, images, b3, base_logit):
        with torch.no_grad():
            seg_prob = torch.sigmoid(self.seg_head(b3).float())
        peaks_yx, _ = nms_heatmap(seg_prob, k=self.k, min_distance=self.cfg['min_distance'])
        hm_logit = self.heatmap_head(b3)                              # 학습 head (detach 없음)
        w = torch.sigmoid(hm_logit.float())[:, 0]
        bi = torch.arange(w.shape[0], device=w.device).unsqueeze(1)
        peak_vals = w[bi, peaks_yx[:, :, 0], peaks_yx[:, :, 1]]       # 고정 위치에서의 학습 head 값
        y1, x1 = E.peaks_to_boxes(peaks_yx, crop=self.cfg['crop'])
        crops = E.crop_boxes(images, y1, x1, crop=self.cfg['crop'])
        if self.training and self.cfg['m4']:
            crops = E.method4(crops)
        B, k = crops.shape[:2]
        k_preds = self.detail_branch(crops.view(B * k, *crops.shape[2:])).view(B, k, 2)
        weights = peak_vals.unsqueeze(-1)
        detail = (k_preds * weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-6)).sum(dim=1)
        bl = base_logit.half() if torch.is_autocast_enabled('cuda') else base_logit
        alpha = self.gating_network(torch.cat([bl, peak_vals.to(bl.dtype)], dim=1))
        fused = alpha * bl + (1.0 - alpha) * detail
        return {'pred': fused, 'detail_pred': detail, 'heatmap': hm_logit, 'alpha': alpha,
                'peaks': peaks_yx, 'peak_vals': peak_vals, 'boxes': torch.stack([y1, x1], -1)}


def build_twohead(cfg, seed, init):
    m = TwoHead(cfg, seed=seed)
    msg = m.load_init(init)
    return m.cuda(), msg


def load(seed, ck):
    m = TwoHead(E.setting_cfg('P3'), seed=seed)
    m.load_init('sbi')
    m.load_state_dict(torch.load(os.path.join(C.RUNS, f'TH_s{seed}', f'ckpt_{ck}.pt'), map_location='cpu', weights_only=False)['model'])
    return m.cuda().eval()


def main():
    if sys.argv[1] == 'train':
        seed = sys.argv[sys.argv.index('--seed') + 1]
        R.build_model = build_twohead
        sys.argv = ['run_train.py', '--setting', 'P3', '--data', '100', '--seed', seed, '--no_test', '--run_id', f'TH_s{seed}']
        R.main()
        open(os.path.join(C.RUNS, f'TH_s{seed}', 'QA_TWOHEAD.txt'), 'w', encoding='utf-8').write(
            'code/qa_twohead.py의 TwoHead 모델(config.json의 setting P3는 학습 설정만 같다는 뜻)\n')
    else:
        ms, names = [], []
        for seed in (0, 1, 2):
            for ck in ('best', 'last'):
                ms.append(load(seed, ck))
                names.append(f'TH|{seed}|{ck}')
        sc = T5.eval_models(ms, sets=T5.ALL)
        np.savez(OUT, names=np.array(names), **{s: sc[s] for s in T5.ALL})
        print('TWOHEAD_EVAL_OK', flush=True)


if __name__ == '__main__':
    main()
