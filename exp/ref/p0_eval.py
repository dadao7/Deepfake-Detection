"""P0: 논문 원래 모델(저자 공개 가중치 current_model_180.pth)을 FF++ c40 공식 테스트셋으로 평가 (코드 수정 없음)
- 저자 inference 와 같은 데이터: test.json 의 원본 140 + 위조 4종 × 140, 영상당 100프레임, 정규화 없음(0~1), fp32
- 지표: 저자 방식(영상별로 높은·낮은 10프레임 제외 평균 → AUC, 최적 기준값 ACC) + 라벨 버그 수정값 + 기준 0.5 ACC 등"""
import json, os, sys, time
import h5py, numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

M2F2 = r'C:\m2f2\M2F2_Det'
sys.path.insert(0, M2F2)
sys.path.insert(0, r'C:\m2f2\code')
os.chdir(M2F2)
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, 'weights_only': False})
from sequence.models.M2F2_Det.models.model import M2F2Det  # noqa: E402

OUT = r'C:\Users\COM\AppData\Local\Temp\claude\c--m2f2\0e1dfcd2-bc72-4433-a053-953ab69c855d\scratchpad\p0'
os.makedirs(OUT, exist_ok=True)
dev = torch.device('cuda')

# ── 1) 저자 스크립트와 같은 방식으로 모델 생성·가중치 로드 ──
model = M2F2Det(clip_text_encoder_name="openai/clip-vit-large-patch14-336",
                clip_vision_encoder_name="openai/clip-vit-large-patch14-336",
                deepfake_encoder_name='efficientnet_b4', hidden_size=1792)
def remap(k):
    """현재 transformers 버전의 키 이름에 맞추기 (train_improved.load_model 과 같은 규칙)"""
    for a in ('module.', 'baseline.', 'vision_tower.'):
        if k.startswith(a):
            k = k.replace(a, '', 1)
    if k.startswith('vision_model.'):
        k = k.replace('vision_model.', '', 1)
    return k.replace('.text_model.', '.').replace('.vision_model.', '.')


vt = torch.load(r'C:\m2f2\weights\vision_tower.pth')
r1 = model.clip_vision_encoder.model.load_state_dict({remap(k): v for k, v in vt.items()}, strict=False)
print('LLaVA vision tower 로드: missing', len(r1.missing_keys), 'unexpected', len(r1.unexpected_keys))
ck = torch.load(r'C:\m2f2\weights\current_model_180.pth', map_location='cpu')
r2 = model.load_state_dict({remap(k): v for k, v in ck['model_state_dict'].items()}, strict=False)
print('저자 가중치 로드: missing', len(r2.missing_keys), 'unexpected', len(r2.unexpected_keys), r2.missing_keys[:3], r2.unexpected_keys[:3])
model = nn.DataParallel(model).to(dev)
model.eval()

# ── 2) 우리 학습 코드(train_improved.load_model) 방식과 가중치가 같은지 ──
import train_improved as T  # noqa: E402
ours = T.load_model('dynamic').baseline
sd_a = {k: v for k, v in model.state_dict().items()}
sd_b = {k: v for k, v in ours.state_dict().items()}
common = [k for k in sd_a if k in sd_b]
diff = [k for k in common if not torch.equal(sd_a[k].cpu(), sd_b[k].cpu())]
print(f'가중치 비교: 공통 {len(common)}개 / 다른 것 {len(diff)}개 / 한쪽에만 {len(set(sd_a) ^ set(sd_b))}개', diff[:5])
del ours
torch.cuda.empty_cache()

# ── 3) 공식 테스트셋 추론 ──
splits = json.load(open(os.path.join(M2F2, 'utils', 'FFPP_split', 'test.json')))
cats = ['original', 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Face2Face']
items = []  # (cat, key, label)
for ci, c in enumerate(cats):
    keys = [x for p in splits for x in p] if ci == 0 else [k for p in splits for k in ('_'.join([p[0], p[1]]), '_'.join([p[1], p[0]]))]
    items += [(c, k, 0 if ci == 0 else 1) for k in keys]
print('테스트 영상', len(items))

rows = []  # cat, key, label, frame_idx, p_fake
t0 = time.time()
with torch.no_grad():
    for vi, (c, k, lab) in enumerate(items):
        with h5py.File(os.path.join(r'C:\m2f2\data\test_only', f'FF++_{c}_c40_test_only.h5'), 'r') as f:
            frames = f[k][:100]
        x = torch.from_numpy(frames.astype(np.float32) / 255.).permute(0, 3, 1, 2)
        probs = []
        for s in range(0, len(x), 50):
            out = model(x[s:s + 50].to(dev), return_dict=True)['pred']
            probs.append(torch.softmax(out.float(), 1)[:, 1].cpu())
        p = torch.cat(probs).numpy()
        rows += [(c, k, lab, i, float(v)) for i, v in enumerate(p)]
        if vi % 70 == 0:
            print(f'{vi}/{len(items)} 영상, {time.time() - t0:.0f}초', flush=True)
np.save(os.path.join(OUT, 'p0_frames.npy'), np.array([r[4] for r in rows], dtype=np.float32))
json.dump([[r[0], r[1], r[2], r[3]] for r in rows], open(os.path.join(OUT, 'p0_index.json'), 'w'))

# ── 4) 지표 ──
by_vid = {}
for c, k, lab, i, p in rows:
    by_vid.setdefault((c, k), [lab, []])[1].append(p)
vids = list(by_vid.items())
v_lab = np.array([v[0] for _, v in vids])
v_trim = np.array([np.mean(sorted(v[1])[10:-10]) for _, v in vids])  # 저자 방식 점수
f_lab = np.array([r[2] for r in rows]); f_p = np.array([r[4] for r in rows])


def best_acc(scores, labels):
    return max(((scores >= t) == labels).mean() for t in np.arange(0, 1.0001, 0.0001))


def summarize(scores, labels):
    pred = scores >= 0.5
    tpr = pred[labels == 1].mean(); tnr = (~pred[labels == 0]).mean()
    return {'AUC': roc_auc_score(labels, scores) * 100, 'ACC_best': best_acc(scores, labels) * 100,
            'ACC@0.5': (pred == labels).mean() * 100, 'balanced_ACC@0.5': (tpr + tnr) / 2 * 100,
            '오탐률(진짜→가짜)': (1 - tnr) * 100, '미탐률(가짜→진짜)': (1 - tpr) * 100}


rep = {'영상 단위(수정)': summarize(v_trim, v_lab), '프레임 단위': summarize(f_p, f_lab)}
# 저자 코드의 라벨 귀속 방식 재현(영상이 바뀔 때 새 영상 라벨을 붙임, 마지막 영상 누락)
seq = [(f'{cats.index(c)}_{k}', lab, p) for c, k, lab, i, p in rows]
pl, gl, cur, last = [], [], [], ''
for vid, lab, p in seq:
    if vid != last:
        last = vid
        if cur:
            pl.append(np.mean(sorted(cur)[10:-10])); gl.append(lab)
        cur = []
    cur.append(p)
pl, gl = np.array(pl), np.array(gl)
rep['영상 단위(저자 코드 그대로)'] = {'AUC': roc_auc_score(gl, pl) * 100, 'ACC_best': best_acc(pl, gl) * 100}
for ci, c in enumerate(cats[1:], 1):
    m = np.array([cc in ('original', c) for (cc, _), _ in vids])
    rep[f'방식별 AUC: {c}'] = {'AUC': roc_auc_score(v_lab[m], v_trim[m]) * 100}
json.dump(rep, open(os.path.join(OUT, 'p0_metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for k, v in rep.items():
    print(k, {a: round(b, 2) for a, b in v.items()})
print(f'총 {time.time() - t0:.0f}초')
