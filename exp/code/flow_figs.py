"""프로젝트 흐름 문서용 파이프라인 그림(matplotlib). 구조는 code/expmodel.py·basetop.py·run_train_unfreeze.py·t5_merge.py 기준.
범례: 회색 = 고정, 파랑 = 학습, 주황 굵은 테두리 = 이번 단계에서 새로 생기거나 바뀐 부분, 빨간 점선 = gradient 경로.
  python code/flow_figs.py → PROJECT_FLOW/figs/*.png"""
import os
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.axes  # noqa: E402
_orig_text = matplotlib.axes.Axes.text


def _clean(s):
    """맑은 고딕에 없는 글리프 치환: ✔ → *, − → -, ⓪ → 0."""
    return s.replace(' ✔', '*').replace('✔', '*').replace('−', '-').replace('⓪', '0.') if isinstance(s, str) else s


matplotlib.axes.Axes.text = lambda self, x, y, s, *a, **k: _orig_text(self, x, y, _clean(s), *a, **k)
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'PROJECT_FLOW', 'figs')
COL = {'frozen': ('#e9ecef', '#495057'), 'train': ('#d0e4ff', '#1c4f9c'), 'data': ('#ffffff', '#343a40'),
       'op': ('#f8f9fa', '#343a40'), 'out': ('#fff4e0', '#8a5300'), 'note': ('#fffbe6', '#7a6400'), 'off': ('#ffffff', '#adb5bd')}
NEW = '#e8590c'


def fig(w=16, h=7, title=None):
    f, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis('off')
    if title:
        ax.text(0.2, h - 0.35, title, fontsize=14, fontweight='bold', va='top')
    return f, ax


def box(ax, x, y, w, h, text, kind='op', new=False, fs=9.5, dashed=False, bold=False):
    fc, ec = COL[kind]
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc,
                       ec=NEW if new else ec, lw=3.0 if new else 1.2, ls='--' if dashed else '-')
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color='#212529',
            fontweight='bold' if bold else 'normal', wrap=True)
    if new:
        ax.text(x + w - 0.05, y + h + 0.02, '변경' if new is True else new, ha='right', va='bottom', fontsize=8.5,
                color='white', fontweight='bold', bbox=dict(boxstyle='round,pad=0.15', fc=NEW, ec=NEW))
    return (x, y, w, h)


def mid(b, side):
    x, y, w, h = b
    return {'l': (x, y + h / 2), 'r': (x + w, y + h / 2), 't': (x + w / 2, y + h), 'b': (x + w / 2, y)}[side]


def arrow(ax, p, q, color='#343a40', ls='-', lw=1.4, text=None, tpos=0.5, toff=(0, 0.12), fs=8.5, rad=0.0):
    ax.annotate('', xy=q, xytext=p, arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, ls=ls, shrinkA=2, shrinkB=2,
                                                     connectionstyle=f'arc3,rad={rad}'))
    if text:
        ax.text(p[0] + (q[0] - p[0]) * tpos + toff[0], p[1] + (q[1] - p[1]) * tpos + toff[1], text, fontsize=fs,
                ha='center', va='bottom', color=color)


def legend(ax, x, y, extra=(), sp=2.3):
    items = [('frozen', '고정(학습 안 함)'), ('train', '학습')] + list(extra)
    for i, (k, t) in enumerate(items):
        box(ax, x + i * sp, y, 0.45, 0.3, '', k)
        ax.text(x + i * sp + 0.55, y + 0.15, t, va='center', fontsize=8.5)
    j = len(items)
    p = FancyBboxPatch((x + j * sp, y), 0.45, 0.3, boxstyle='round,pad=0.02', fc='white', ec=NEW, lw=3)
    ax.add_patch(p)
    ax.text(x + j * sp + 0.55, y + 0.15, '이번 단계에서 바뀐 부분', va='center', fontsize=8.5)


def save(f, name):
    os.makedirs(OUT, exist_ok=True)
    f.text(0.99, 0.005, '* = 확실한 차이(영상 단위 95% 신뢰구간이 0 밖)', ha='right', va='bottom', fontsize=8, color='#868e96')
    f.savefig(os.path.join(OUT, name), dpi=130, bbox_inches='tight')
    plt.close(f)
    print('저장', name)


def baseline_block(ax, x, y, new_top=False, top_kind='frozen', label='P0 저자 baseline M2F2-Det (고정)'):
    """P0 내부를 축약해 그림: 이미지 → CLIP·EffNet 앞단(고정) → b_3 / 상단 → baseline logit. 반환: (b3 박스, logit 박스)"""
    ax.add_patch(FancyBboxPatch((x - 0.15, y - 0.2), 7.0, 3.0, boxstyle='round,pad=0.02', fc='none', ec='#868e96', lw=1, ls=':'))
    ax.text(x - 0.05, y + 2.62, label, fontsize=9, color='#495057')
    b_clip = box(ax, x, y + 1.6, 2.0, 0.7, 'CLIP ViT-L/14\n비전 인코더', 'frozen')
    b_eff = box(ax, x, y, 2.0, 0.7, 'EfficientNet-B4\nfeatures[0..6]', 'frozen')
    b_b3 = box(ax, x + 2.4, y, 1.3, 0.7, 'b_3\n272×11×11', 'data', fs=9)
    b_top = box(ax, x + 4.05, y + 0.55, 1.45, 1.5, '상단\nf7·f8·투영층\n·출력층', top_kind, new=new_top, fs=9)
    b_lg = box(ax, x + 5.75, y + 0.95, 0.95, 0.7, 'baseline\nlogit', 'out', fs=9)
    arrow(ax, mid(b_eff, 'r'), mid(b_b3, 'l'))
    arrow(ax, mid(b_clip, 'r'), (x + 4.05, y + 1.75), text='CLS', tpos=0.2)
    arrow(ax, mid(b_b3, 'r'), (x + 4.05, y + 0.85))
    arrow(ax, mid(b_top, 'r'), mid(b_lg, 'l'))
    return b_b3, b_lg


def f_p0():
    f, ax = fig(16, 6.4, '① P0: 저자 모델 M2F2-Det (전부 고정, S1에서 재현) — 우리가 바꾸지 않은 출발점')
    b_in = box(ax, 0.3, 2.6, 1.5, 0.9, '얼굴 프레임\n224×224', 'data')
    b_clip = box(ax, 2.4, 4.1, 2.4, 0.9, 'CLIP ViT-L/14-336\n비전 인코더', 'frozen')
    b_eff = box(ax, 2.4, 1.1, 2.4, 0.9, 'EfficientNet-B4\nfeatures[0..6]', 'frozen')
    b_cls = box(ax, 5.3, 4.2, 1.2, 0.7, 'CLS\n1024', 'data')
    b_b3 = box(ax, 5.3, 1.2, 1.3, 0.7, 'b_3\n272×11×11', 'data')
    b_vp = box(ax, 7.0, 4.1, 2.3, 0.9, 'vision_proj\n1024→1792 + LN, ×α_v', 'frozen', fs=9)
    b_f78 = box(ax, 7.0, 1.1, 2.3, 0.9, 'features[7]·[8]\n+ avgpool', 'frozen', fs=9)
    b_dp = box(ax, 9.7, 1.1, 1.9, 0.9, 'deepfake_proj\n1792 + LN', 'frozen', fs=9)
    b_ad = box(ax, 7.0, 2.65, 2.3, 0.8, 'bridge adapter\n= 입력 무관 상수 128', 'frozen', fs=9, dashed=True)
    b_cat = box(ax, 11.95, 2.55, 1.3, 1.0, 'concat\n3712', 'op')
    b_o = box(ax, 13.6, 2.55, 1.15, 1.0, 'output\nLinear→2', 'frozen', fs=9)
    b_p = box(ax, 13.45, 0.6, 2.3, 1.0, '프레임 확률 →\n영상 점수(위·아래\n10% 제외 평균)', 'out', fs=8.5)
    arrow(ax, mid(b_in, 'r'), mid(b_clip, 'l'))
    arrow(ax, mid(b_in, 'r'), mid(b_eff, 'l'))
    arrow(ax, mid(b_clip, 'r'), mid(b_cls, 'l'))
    arrow(ax, mid(b_eff, 'r'), mid(b_b3, 'l'))
    arrow(ax, mid(b_cls, 'r'), mid(b_vp, 'l'))
    arrow(ax, mid(b_b3, 'r'), mid(b_f78, 'l'))
    arrow(ax, mid(b_f78, 'r'), mid(b_dp, 'l'))
    arrow(ax, mid(b_vp, 'r'), (11.95, 3.35))
    arrow(ax, mid(b_ad, 'r'), mid(b_cat, 'l'))
    arrow(ax, mid(b_dp, 'r'), (11.95, 2.75))
    arrow(ax, mid(b_cat, 'r'), mid(b_o, 'l'))
    arrow(ax, mid(b_o, 'b'), mid(b_p, 't'))
    ax.text(5.95, 0.55, '★ S1 캐시 지점: b_3·baseline logit·CLS를 한 번만 계산해 저장\n(원본 실시간 계산과 비트 단위로 같음) → 이후 모든 실험은 이 뒤만 학습',
            ha='center', fontsize=9, color='#7a6400', bbox=dict(boxstyle='round,pad=0.3', fc='#fffbe6', ec='#e0c200'))
    ax.text(10.4, 5.35, '공개 가중치 재현: FF++ c40 AUC 95.22 (논문 보고 96.58은 재현 안 됨), Celeb-DF 76.15', fontsize=9.5, color='#343a40')
    legend(ax, 0.3, 5.3)
    save(f, 'F1_P0_baseline.png')


def branch_common(ax, b_b3, b_lg, x0, stage):
    """P1~P4 분기 그림의 공통 부분. stage: P1 | P2 | P3 | P4a."""
    hm_new = stage == 'P2'
    if stage == 'P1':
        b_map = box(ax, x0, 4.6, 2.2, 0.85, '특징 평균 맵\nσ(채널 평균 b_3) 11×11', 'op', new='새로', fs=9)
        b_pk = box(ax, x0 + 2.9, 4.6, 1.7, 0.85, '최댓값 위치\nk = 1', 'op', new='새로', fs=9)
    else:
        b_map = box(ax, x0, 4.6, 2.2, 0.85, 'heatmap head\n(272→64→1, SBI 사전학습)', 'train' if stage == 'P4a' else 'frozen',
                    new=hm_new or (stage == 'P4a' and '변경'), fs=9)
        b_pk = box(ax, x0 + 2.9, 4.6, 1.7, 0.85, 'Top-3 peak\n(NMS, 거리 ≥ 2)', 'op', new=hm_new, fs=9)
    arrow(ax, (b_b3[0] + b_b3[2] / 2, b_b3[1] + b_b3[3]), (x0, 5.0), rad=-0.15)
    lab = {'P2': 'σ · detach', 'P3': 'σ · detach', 'P4a': 'σ (detach 없음)'}.get(stage)
    arrow(ax, mid(b_map, 'r'), mid(b_pk, 'l'), color=NEW if stage == 'P4a' else '#343a40')
    if lab:
        ax.text(x0 + 2.55, 4.3, lab, ha='center', va='top', fontsize=8.5, color=NEW if stage == 'P4a' else '#343a40')
    k = '1' if stage == 'P1' else '3'
    b_crop = box(ax, x0 + 5.1, 4.6, 1.8, 0.85, f'원본 이미지에서\n64×64 crop × {k}', 'op',
                 new='새로' if stage == 'P1' else (stage == 'P2'), fs=9)
    arrow(ax, mid(b_pk, 'r'), mid(b_crop, 'l'))
    if stage in ('P3', 'P4a'):
        b_m4 = box(ax, x0 + 5.1, 3.1, 1.8, 1.0, '패치 정규화(M4)\n학습 때만: blur·noise\n·dropout', 'op',
                   new=(stage == 'P3') and '새로', fs=8.5)
        arrow(ax, mid(b_crop, 'b'), mid(b_m4, 't'))
        src = mid(b_m4, 'r')
    else:
        src = mid(b_crop, 'r')
    b_db = box(ax, x0 + 7.4, 4.0, 1.9, 1.0, 'detail 분기\nDenseNet121\n(crop 공유)', 'train', new='새로' if stage == 'P1' else False, fs=9)
    arrow(ax, src, mid(b_db, 'l'))
    if stage == 'P1':
        b_agg = box(ax, x0 + 9.7, 4.15, 1.5, 0.7, 'detail logit', 'out', fs=9)
    else:
        b_agg = box(ax, x0 + 9.7, 4.05, 1.5, 0.9, 'peak 확률\n가중 평균', 'op', new=(stage == 'P2'), fs=9)
        arrow(ax, (x0 + 3.75, 5.45), (x0 + 10.45, 4.95), color='#868e96', ls=':', rad=-0.22)
        ax.text(x0 + 7.1, 6.05, 'peak 확률(가중치)', fontsize=8, color='#868e96', ha='center')
    arrow(ax, mid(b_db, 'r'), mid(b_agg, 'l'))
    alpha_txt = 'α·baseline\n+ (1−α)·detail' if stage in ('P3', 'P4a') else '0.5·baseline\n+ 0.5·detail (고정 α)'
    box(ax, x0 + 11.7, 3.05, 1.8, 1.1, '결합 logit\n' + alpha_txt, 'out', new='새로' if stage == 'P1' else False, fs=8.5)
    arrow(ax, mid(b_agg, 'r'), (x0 + 11.7, 3.95))
    arrow(ax, mid(b_lg, 'r'), (x0 + 11.7, 3.25))
    if stage in ('P3', 'P4a'):
        b_g = box(ax, x0 + 9.4, 1.15, 1.9, 1.15, '동적 결합 MLP\n[baseline logit 2\n+ peak 3]→16→1→σ', 'train',
                  new=(stage == 'P3') and '새로', fs=8.5)
        arrow(ax, (b_lg[0] + b_lg[2], b_lg[1] + 0.15), (x0 + 9.4, 1.75), color='#868e96')
        arrow(ax, mid(b_g, 'r'), (x0 + 11.95, 3.05), text='α', toff=(0.1, 0.0))
    if stage == 'P4a':
        arrow(ax, (x0 + 12.6, 4.15), (x0 + 1.1, 5.45), color='#c92a2a', ls='--', lw=1.6, rad=0.25)
        ax.text(x0 + 6.6, 6.55, '분류 손실의 gradient가 heatmap head까지 흐름 → heatmap이 얼굴 중앙으로 붕괴', fontsize=9.5,
                color='#c92a2a', ha='center')
    return b_map


def f_p1():
    f, ax = fig(17.6, 6.9, '② P1·P1′ (S4): P0 위에 detail crop 분기 추가')
    b_b3, b_lg = baseline_block(ax, 0.3, 0.9)
    branch_common(ax, b_b3, b_lg, 3.6, 'P1')
    ax.text(0.3, 0.25, '학습: baseline은 고정, 손실 = CE(결합) + 0.15·CE(detail).  P1: detail 분기 ImageNet 초기화 / P1′: 같은 구조, SBI 사전학습(S3) 초기화.\n'
            '결과: P0 대비 AUC 소폭 저하(FF++ −0.27 ✔, CDF −0.58 ✔)', fontsize=9.5)
    legend(ax, 11.3, 6.3, sp=1.75)
    save(f, 'F2_P1_detail_branch.png')


def f_p2():
    f, ax = fig(17.6, 6.9, '③ P2 (S4): 위치 지도를 SBI heatmap으로, crop을 Top-3로, 결합을 peak 가중 평균으로')
    b_b3, b_lg = baseline_block(ax, 0.3, 0.9)
    branch_common(ax, b_b3, b_lg, 3.6, 'P2')
    ax.text(0.3, 0.25, '중간 변형: P2a = heatmap crop 1개 / P2b = 특징 평균 맵 Top-3(단순 평균).  heatmap은 detach → 분류 손실이 heatmap head를 바꾸지 않음.\n'
            '결과: P2a 균형정확도 +2.6 ✔, P2는 best에서 P0와 대등, 오래 학습하면(last) Celeb-DF 69.2로 붕괴', fontsize=9.5)
    legend(ax, 11.3, 6.3, sp=1.75)
    save(f, 'F3_P2_heatmap_top3.png')


def f_p3():
    f, ax = fig(17.6, 6.9, '④ P3 = 현재 앱 모델 (S4): P2 + 패치 정규화(M4) + 동적 결합')
    b_b3, b_lg = baseline_block(ax, 0.3, 0.9)
    branch_common(ax, b_b3, b_lg, 3.6, 'P3')
    ax.text(0.3, 0.25, '결과: P0와 대등(FF++ 95.11 vs 95.22, CDF 76.86 vs 76.15, 차이 불분명). 오래 학습해도 무너지지 않음(last CDF +7.1 ✔ vs P2) = 과적합 안전장치.\n'
            '중간 변형: P3a = M4만, P3b = 동적 결합만.  α는 학습할수록 0.68 → 0.90(baseline 쪽)으로 커짐', fontsize=9.5)
    legend(ax, 11.3, 6.3, sp=1.75)
    save(f, 'F4_P3_current_model.png')


def f_p4():
    f, ax = fig(17.6, 8.0, '⑤ P4a·P4b·P4 (S4·S6): detach 제거와 위치 설명용 seg head')
    b_b3, b_lg = baseline_block(ax, 0.3, 0.9)
    branch_common(ax, b_b3, b_lg, 3.6, 'P4a')
    b_seg = box(ax, 0.3, 4.55, 2.4, 0.75, 'seg head (고정 복사본)\n위치 표시 전용', 'frozen', new='P4b·P4', fs=9)
    arrow(ax, (b_b3[0] + 0.3, b_b3[1] + b_b3[3]), (1.5, 4.55), color='#868e96', rad=0.2)
    b_hmout = box(ax, 0.3, 5.75, 2.4, 0.75, '설명용 heatmap\n(분류에 안 쓰임)', 'out', fs=9)
    arrow(ax, mid(b_seg, 't'), mid(b_hmout, 'b'))
    ax.text(0.3, 6.75, 'P4b = P3 + seg head, P4 = P4a + seg head\n→ 분류는 P3/P4a와 같고 위치 설명만 다름', fontsize=9)
    ax.text(0.3, 0.25, '결과: P4a는 Celeb-DF −6.4(best)·−7.2(last) ✔, DFD −4.6~−6.8 ✔, 경계 적중 35% → 5%.  → 위치 붕괴가 일반화 손해의 원인(TASKS4 anchor 실험으로 확인)',
            fontsize=9.5)
    legend(ax, 9.0, 7.35, extra=[('out', '출력')], sp=1.75)
    save(f, 'F5_P4_detach_seghead.png')


def f_s8_s9():
    f, ax = fig(17, 7.8, '⑥ S8·S9: 학습 데이터 바꾸기(S8)와 baseline 상단 풀기(S9)')
    ax.text(0.3, 6.6, 'S8: 구조는 P2·P3·P4a 그대로, 학습 데이터만 SBI 쌍으로 교체', fontsize=11, fontweight='bold')
    box(ax, 0.3, 5.3, 2.6, 0.9, 'FF++ 진짜/가짜\n(S4 학습 데이터)', 'data', dashed=True)
    box(ax, 3.5, 5.3, 3.0, 0.9, 'SBI 쌍(진짜 영상 + 그 영상으로\n합성한 가짜, 56,308쌍)', 'data', new='데이터 변경')
    b_m = box(ax, 7.2, 5.3, 2.8, 0.9, 'P2 / P3 / P4a\n(구조 동일, baseline 고정)', 'train')
    arrow(ax, (6.5, 5.75), mid(b_m, 'l'))
    ax.text(10.3, 5.55, '결과: FF++ −2~−4, Celeb-DF −7~−14.5 ✔(가설 반증)\n단, P2→P3 안전장치 효과는 재현(CDF +7.3 ✔)', fontsize=9.5)
    ax.text(0.3, 4.45, 'S9: baseline 상단(약 10.3M)을 풀어 함께 학습 — U0 = 상단만, U3 = P3 분기 + 상단', fontsize=11, fontweight='bold')
    b_b3, b_lg = baseline_block(ax, 0.3, 0.75, new_top='풀어서 학습', top_kind='train',
                                label='baseline: CLIP·EffNet 앞단은 고정, 상단(EffNet f7·f8, deepfake_proj, vision_proj, α_v, output)은 학습')
    b_br = box(ax, 7.8, 1.6, 2.6, 1.0, 'P3 분기\n(heatmap·Top-3·DenseNet\n·M4·동적 결합)', 'train', fs=9)
    b_out = box(ax, 10.9, 1.75, 1.6, 0.7, '결합 logit', 'out', fs=9)
    arrow(ax, mid(b_lg, 'r'), mid(b_br, 'l'))
    arrow(ax, mid(b_br, 'r'), mid(b_out, 'l'))
    ax.text(12.8, 1.2, 'U0f/U3f = FF++로 학습\nU0s/U3s = SBI로 학습\n\n결과: P0를 못 넘음\n(U0f CDF −2.1, U0s CDF −10.6)\n예외: U3s는 baseline 부분이\nCDF 76.5 유지 → "SBI 전문가"',
            fontsize=9.5, va='bottom')
    legend(ax, 0.3, 0.1)
    save(f, 'F6_S8_S9_data_unfreeze.png')


def f_t4():
    f, ax = fig(17, 6.2, '⑦ TASKS4: 두 데이터 전문가의 출력 결합(모델 2개) + anchor 학습')
    b_in = box(ax, 0.3, 3.2, 1.5, 0.8, '얼굴 프레임', 'data')
    b_a = box(ax, 2.5, 4.3, 3.2, 1.0, '전문가 A = P3\n(FF++로 학습, baseline 고정)', 'frozen', fs=9.5)
    b_b = box(ax, 2.5, 1.9, 3.2, 1.0, '전문가 B = U3s\n(SBI로 학습, 상단 + 분기)', 'frozen', fs=9.5)
    b_c = box(ax, 6.6, 3.05, 2.6, 1.1, 'logit 결합\n0.3·A + 0.7·B', 'op', new='새로')
    b_o = box(ax, 9.9, 3.2, 1.7, 0.8, '최종 점수', 'out')
    for b in (b_a, b_b):
        arrow(ax, mid(b_in, 'r'), mid(b, 'l'))
        arrow(ax, mid(b, 'r'), mid(b_c, 'l'))
    arrow(ax, mid(b_c, 'r'), mid(b_o, 'l'))
    ax.text(2.5, 0.95, '가중치 0.3:0.7은 FF++ val + DFD-val(배우 신원 분리)로만 선택.  결과(P0 대비): DFD +4.8 ✔, FaceShifter +5.8 ✔, CDF +1.4, FF++ −0.3\n'
            '한계: 추론 때 모델 2개를 돌려야 함.  같은 데이터를 한 모델에 섞어 학습(U3m)하면 이득이 사라짐(FaceShifter 63~65)', fontsize=9.5)
    ax.text(12.1, 5.0, 'anchor 학습(원인 확인 실험)', fontsize=11, fontweight='bold')
    ax.text(12.1, 2.2, 'P4a(detach 없음) + 손실에\nλ·MSE(σ(heatmap), σ(seg head))\n\nλ = 1 → 10 → 100 일 때\n경계 적중 5 → 15 → 37%\nCeleb-DF 69 → 75 → 76.5\n\n→ 위치를 지키면 일반화도\n지켜진다(P3를 넘지는 못함)',
            fontsize=9.5, bbox=dict(boxstyle='round,pad=0.4', fc='#fffbe6', ec='#e0c200'))
    save(f, 'F7_T4_ensemble_anchor.png')


def f_t5():
    f, ax = fig(17, 8.8, '⑧ TASKS5 최종 모델: 학습 없는 데이터 전문가 가중치 병합 (모델 1개)')
    ax.text(0.3, 7.45, '가중치 공간(왜 섞을 수 있나): 같은 seed의 P3와 U3s는 출발점 θ0가 같다', fontsize=11, fontweight='bold')
    b0 = box(ax, 0.4, 5.3, 3.2, 1.3, 'θ0 (출발점)\n상단 = 저자 가중치\n분기 = SBI 사전학습 + seed 규칙 초기화', 'frozen', fs=9)
    bp = box(ax, 5.0, 6.2, 3.3, 0.95, 'P3 = θ0 + τ_P3\nFF++ 학습: 분기만 변함(상단 τ = 0)', 'frozen', fs=9)
    bu = box(ax, 5.0, 4.75, 3.3, 0.95, 'U3s = θ0 + τ_U3s\nSBI 학습: 상단 + 분기 변함', 'frozen', fs=9)
    arrow(ax, mid(b0, 'r'), mid(bp, 'l'), text='FF++로 학습', tpos=0.5)
    arrow(ax, mid(b0, 'r'), mid(bu, 'l'), text='SBI로 학습', tpos=0.5, toff=(0, -0.32))
    bm = box(ax, 9.4, 5.25, 4.2, 1.4, '병합(학습 없음)\nθ = θ0 + 0.5·τ_P3 + 0.75·τ_U3s\nBN 통계는 볼록 결합', 'op', new='새로', fs=10)
    arrow(ax, mid(bp, 'r'), (9.4, 6.2))
    arrow(ax, mid(bu, 'r'), (9.4, 5.7))
    ax.text(13.9, 5.3, '비율 a = 0.5, b = 0.75는\n선택 세트(FF++ val +\nDFD-val)로만 고름\n(243개 변형 중)', fontsize=9)
    ax.text(0.3, 4.05, '병합된 모델의 구조(= U3 구조, 추론은 모델 1개)', fontsize=11, fontweight='bold')
    b_b3, b_lg = baseline_block(ax, 0.3, 0.75, new_top='병합', top_kind='frozen',
                                label='baseline: CLIP·EffNet 앞단 = 저자 가중치 그대로, 상단 = θ0 + 0.75·τ_U3s(SBI 학습분의 75%)')
    b_br = box(ax, 7.8, 1.45, 3.0, 1.25, 'P3 분기(heatmap·Top-3·DenseNet\n·M4·동적 결합)\n= θ0 + 0.5·τ_P3 + 0.75·τ_U3s', 'frozen', new='병합', fs=9)
    b_out = box(ax, 11.3, 1.7, 1.5, 0.75, '결합 logit', 'out', fs=9)
    arrow(ax, mid(b_lg, 'r'), mid(b_br, 'l'))
    arrow(ax, mid(b_br, 'r'), mid(b_out, 'l'))
    ax.text(13.1, 0.75, 'P3 대비 AUC\nCeleb-DF +2.6 ✔\nFaceShifter +4.1 ✔\nDFD +5.3 ✔\nFF++ −0.2(유지)\n\n위치 설명 유지(35%)\n추론 비용 = P3',
            fontsize=10, bbox=dict(boxstyle='round,pad=0.4', fc='#fff4e0', ec='#8a5300'))
    save(f, 'F8_T5_final_merge.png')


def f_protocol():
    f, ax = fig(16, 5.6, '⓪ 평가 설계: 무엇으로 학습하고, 무엇으로 고르고, 무엇으로 보고하나')
    b1 = box(ax, 0.3, 2.3, 3.2, 1.6, '학습\nFF++ c40 공식 train\n(360쌍, 영상당 10장)\n또는 SBI 쌍', 'train', fs=10)
    b2 = box(ax, 4.3, 2.3, 3.6, 1.6, '선택(고르기) 전용\nFF++ val(70쌍)\n+ DFD-val(배우 14명)\n→ epoch·비율·가중치', 'note', fs=10)
    b3 = box(ax, 8.7, 2.3, 4.0, 1.6, '보고(판정) 전용 — 끝까지 고르는 데 안 씀\nFF++ test 700 · Celeb-DF 518\nFaceShifter 280 · DFD-test(배우 14명)', 'out', fs=10)
    arrow(ax, mid(b1, 'r'), mid(b2, 'l'))
    arrow(ax, mid(b2, 'r'), mid(b3, 'l'))
    ax.text(13.0, 2.2, '• seed 3개(0·1·2)\n• 같은 seed = 같은 배치·증강\n  (공통 난수 → 짝 비교)\n• 영상 단위 짝 bootstrap\n  95% CI, 0 밖이면 ✔\n• 원본 코드 수정 없음\n  (해시 23/23)', fontsize=9.5, va='bottom')
    ax.text(0.3, 1.2, '1학기 평가의 문제: 영상 이름순 80/20 분할 → 공식 test 140개 중 115개가 학습 쪽에 들어감(누수) → 2학기는 공식 분할로 전부 다시 평가', fontsize=10, color='#c92a2a')
    ax.text(0.3, 0.6, 'DFD는 배우 28명을 14/14로 나눠 val/test로 씀(두 배우가 섞인 가짜 1,357개 제외). 교차 데이터셋 얼굴 crop은 dlib 정렬로 재현하고 FF++에서 검증(AUC 차 −0.07)', fontsize=9.5)
    save(f, 'F0_eval_protocol.png')


def f_timeline():
    f, ax = fig(17, 7.0, '프로젝트 전체 흐름 (2026-10-14 ~ 10-17)')
    steps = [('S0~S2', '실험 기반', '공식 분할·캐시·\n원본과 비트 일치\nP0 재현 95.22'),
             ('S3', 'SBI 사전학습', 'heatmap head·\ndetail 분기 초기값'),
             ('S4~S7', '기능 추가 ablation', 'P1→P2→P3→P4\n9설정×seed 3\n단독 향상 없음'),
             ('S8', 'SBI로 분기 학습', '가설 반증\n(오히려 저하)'),
             ('S9', 'baseline 상단 풀기', 'P0 못 넘음\nU3s = SBI 전문가'),
             ('S10', '새 데이터셋 평가', 'DFD·FaceShifter\n효과가 데이터셋\n마다 반대'),
             ('TASKS4', '약점 기반 개선', '출력 결합(2개)\nDFD +4.8 FSH +5.8\nanchor로 원인 확인'),
             ('TASKS5', '가중치 병합', '★ 최종 모델\n단일 모델로\nCDF +2.6 FSH +4.1\nDFD +5.3 (vs P3)'),
             ('TASKS6', '추가 탐색 8가지', '선택 과적합 발견\n이득은 상단에서\n(최종 모델 유지)')]
    w, gap = 1.62, 0.21
    for i, (s, t, r) in enumerate(steps):
        x = 0.3 + i * (w + gap)
        final = s == 'TASKS5'
        box(ax, x, 4.3, w, 1.1, f'{s}\n{t}', 'train' if s.startswith('TASKS') else 'frozen', new='최종' if final else False, fs=9.5, bold=True)
        box(ax, x, 1.9, w, 1.9, r, 'note' if final else 'op', fs=8.8)
        arrow(ax, (x + w / 2, 4.3), (x + w / 2, 3.8), color='#868e96')
        if i < len(steps) - 1:
            arrow(ax, (x + w, 4.85), (x + w + gap, 4.85))
    ax.text(0.3, 5.9, '구간 A: 논문 모델 + 우리 기능 검증', fontsize=10.5, color='#495057')
    ax.text(5.8, 5.9, '구간 B: 원인 검증', fontsize=10.5, color='#495057')
    ax.text(11.2, 5.9, '구간 C·D·E: 자체 개선', fontsize=10.5, color='#1c4f9c')
    ax.text(0.3, 1.2, '질문의 흐름: 기능을 붙이면 좋아지나?(아니오) → 왜 안 되나?(baseline이 학습 데이터를 이미 맞춤, 데이터셋마다 효과가 반대) → 그 반대 효과를 합치면?(결합 → 병합으로 단일 모델 향상)',
            fontsize=10)
    ax.text(0.3, 0.6, '학습 총 95회 + SBI 사전학습 1회, 모든 결과는 seed 3개·신뢰구간으로 판정, 원본 코드 무수정', fontsize=9.5, color='#495057')
    save(f, 'F00_project_timeline.png')


if __name__ == '__main__':
    for fn in (f_timeline, f_protocol, f_p0, f_p1, f_p2, f_p3, f_p4, f_s8_s9, f_t4, f_t5):
        fn()
