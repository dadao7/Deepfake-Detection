"""논문 모델 평가에서 fp32(저자 방식)와 혼합정밀도(fp16)의 결과 차이 확인
p0_eval.py 가 저장한 fp32 프레임 확률(p0_frames.npy)과, 같은 영상 앞 140개(원본 140개)·각 방식 35개를 fp16 으로 다시 계산해 비교"""
import json, os, sys
import h5py, numpy as np, torch, torch.nn as nn
from torch.amp import autocast
from sklearn.metrics import roc_auc_score

M2F2 = r'C:\m2f2\M2F2_Det'
sys.path.insert(0, M2F2); sys.path.insert(0, r'C:\m2f2\code'); os.chdir(M2F2)
_orig = torch.load
torch.load = lambda *a, **k: _orig(*a, **{**k, 'weights_only': False})
import train_improved as T  # noqa: E402

OUT = r'C:\Users\COM\AppData\Local\Temp\claude\c--m2f2\0e1dfcd2-bc72-4433-a053-953ab69c855d\scratchpad\p0'
idx = json.load(open(os.path.join(OUT, 'p0_index.json')))
p32 = np.load(os.path.join(OUT, 'p0_frames.npy'))
model = T.load_model('dynamic').baseline.eval()

# 원본 70개 + 방식별 35개 = 210개 영상
vids = []
for c in ('original', 'Deepfakes', 'FaceSwap', 'NeuralTextures', 'Face2Face'):
    ks = []
    for i, (cc, k, lab, fi) in enumerate(idx):
        if cc == c and fi == 0:
            ks.append((c, k, lab, i))
    vids += ks[:70 if c == 'original' else 35]

a32, a16, labs, v32, v16 = [], [], [], [], []
with torch.no_grad():
    for c, k, lab, start in vids:
        with h5py.File(os.path.join(r'C:\m2f2\data\test_only', f'FF++_{c}_c40_test_only.h5'), 'r') as f:
            x = torch.from_numpy(f[k][:100].astype(np.float32) / 255.).permute(0, 3, 1, 2)
        ps = []
        for s in range(0, 100, 50):
            with autocast('cuda'):
                out = model(x[s:s + 50].cuda(), return_dict=True)['pred']
            ps.append(torch.softmax(out.float(), 1)[:, 1].cpu())
        q = torch.cat(ps).numpy(); r = p32[start:start + 100]
        a16 += list(q); a32 += list(r); labs.append(lab)
        v16.append(np.mean(sorted(q)[10:-10])); v32.append(np.mean(sorted(r)[10:-10]))
a16, a32 = np.array(a16), np.array(a32)
print(f'영상 {len(vids)}개 / 프레임 {len(a16)}장')
print(f'프레임 확률 차이: 평균 {np.abs(a16 - a32).mean():.5f}, 최대 {np.abs(a16 - a32).max():.5f}')
print(f'판정(0.5 기준)이 달라진 프레임 {int(((a16 >= .5) != (a32 >= .5)).sum())}장')
print(f'영상 AUC: fp32 {roc_auc_score(labs, v32) * 100:.3f} / fp16 {roc_auc_score(labs, v16) * 100:.3f}')
