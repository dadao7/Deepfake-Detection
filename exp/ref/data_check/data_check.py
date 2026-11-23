"""지금 있는 파일로 실험이 되는지 확인 (읽기만, GPU 안 씀)
(a) lm_json 랜드마크가 h5 프레임과 같은 이미지·좌표인지
(b) FF++ 원본·위조 h5 프레임이 정렬되어 픽셀 차이로 위조 영역 마스크를 만들 수 있는지
(c) Celeb-DF 테스트 목록 영상이 다 있는지"""
import json, os, glob, re, collections
import h5py, numpy as np, cv2

OUT = r'C:\Users\COM\AppData\Local\Temp\claude\c--m2f2\0e1dfcd2-bc72-4433-a053-953ab69c855d\scratchpad\data_check'
os.makedirs(OUT, exist_ok=True)
FF = r'C:\m2f2\data\FF++'
split = {n: json.load(open(rf'C:\m2f2\M2F2_Det\utils\FFPP_split\{n}.json')) for n in ('train', 'val', 'test')}
ids = {n: {i for p in v for i in p} for n, v in split.items()}

# ── (a) 랜드마크 ──
lm = json.load(open(r'C:\m2f2\data\lm_json\train_81_FF++_processed_720.json'))['data']
per_vid = collections.defaultdict(list)
for e in lm:
    m = re.match(r'frames/(\d+)/(\d+)\.png', e['image_path'])
    per_vid[m.group(1)].append((int(m.group(2)), e['aligned_lms']))
vids = sorted(per_vid)
print('(a) 랜드마크 영상 수', len(vids), '| 공식 train과 일치', len(set(vids) & ids['train']), '| val', len(set(vids) & ids['val']), '| test', len(set(vids) & ids['test']))
counts = [len(v) for v in per_vid.values()]
print('    영상당 프레임 수 min/mean/max', min(counts), round(np.mean(counts), 1), max(counts), '| 랜드마크 점 수', len(lm[0]['aligned_lms']))
lm_keys = set(lm[0].keys())
print('    항목 키', lm_keys)
with h5py.File(os.path.join(FF, 'FF++_original_c40.h5'), 'r') as f:
    canvas = []
    for v in vids[:4]:
        fr_idx, pts = sorted(per_vid[v])[0]
        n = f[v].shape[0]
        idxs = sorted(i for i, _ in per_vid[v])
        img = f[v][min(fr_idx, n - 1)].copy()
        for x, y in pts:
            cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)
        cv2.putText(img, f'{v} f{fr_idx} h5n={n}', (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        canvas.append(img)
        print(f'    영상 {v}: h5 프레임 {n}개, 랜드마크 프레임 번호 {idxs[:6]}... (최대 {max(idxs)})')
    cv2.imwrite(os.path.join(OUT, 'a_landmarks.png'), cv2.cvtColor(np.hstack(canvas), cv2.COLOR_RGB2BGR))

# ── (b) 원본 vs 위조 픽셀 차이 ──
rows = []
for meth in ('Deepfakes', 'FaceSwap', 'Face2Face', 'NeuralTextures'):
    with h5py.File(os.path.join(FF, f'FF++_{meth}_c40.h5'), 'r') as ff, h5py.File(os.path.join(FF, 'FF++_original_c40.h5'), 'r') as fo:
        tgt, src = split['test'][0]
        key = f'{tgt}_{src}'
        nf, no = ff[key].shape[0], fo[tgt].shape[0]
        a = fo[tgt][10].astype(np.int16); b = ff[key][10].astype(np.int16)
        d = np.abs(a - b).mean(2)
        # 정렬 정도: 가장자리 띠(배경)와 가운데(얼굴)의 평균 차이
        edge = np.concatenate([d[:20].ravel(), d[-20:].ravel(), d[:, :20].ravel(), d[:, -20:].ravel()]).mean()
        center = d[60:170, 60:170].mean()
        print(f'(b) {meth:15s} {key}: 프레임 수 원본 {no} / 위조 {nf} | 차이 평균 가장자리 {edge:.1f} vs 얼굴 가운데 {center:.1f}')
        dm = np.clip(d * 4, 0, 255).astype(np.uint8)
        rows.append(np.hstack([a.astype(np.uint8), b.astype(np.uint8), cv2.cvtColor(dm, cv2.COLOR_GRAY2RGB)]))
cv2.imwrite(os.path.join(OUT, 'b_diff.png'), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR))

# ── (c) Celeb-DF ──
cd = r'C:\m2f2\data\Celeb-DF-v2'
lst = [l.split() for l in open(os.path.join(cd, 'List_of_testing_videos.txt'), encoding='utf-8') if l.strip()]
found, nfr = 0, []
for lab, rel in lst:
    sub, name = rel.split('/')
    d = os.path.join(cd, sub, 'frames', os.path.splitext(name)[0])
    fs = glob.glob(os.path.join(d, '*.png')) + glob.glob(os.path.join(d, '*.jpg'))
    if fs:
        found += 1; nfr.append(len(fs))
lab_cnt = collections.Counter(l for l, _ in lst)
print(f'(c) Celeb-DF 테스트 목록 {len(lst)}개 (라벨 {dict(lab_cnt)}) | 프레임 있는 영상 {found}개 | 영상당 프레임 min/mean/max {min(nfr)}/{np.mean(nfr):.1f}/{max(nfr)}')
img = cv2.imread(glob.glob(os.path.join(cd, 'Celeb-synthesis', 'frames', '*', '*.png'))[0])
print('    Celeb-DF 프레임 크기', img.shape)
