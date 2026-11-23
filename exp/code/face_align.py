"""S10: h5와 같은 방식의 얼굴 crop 재현 (PROTOCOL §16)
h5 얼굴 = 고정 템플릿으로의 닮음변환 정렬(정렬 랜드마크의 눈 위치 표준편차 약 2px). 템플릿 = train 정렬 랜드마크 68점 평균.
절차: dlib HOG 얼굴 검출(긴 변 640으로 축소해 검출, 가장 큰 얼굴) → dlib 68점(원 해상도) → 템플릿으로 닮음변환 → 224×224."""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

MODEL_DIR = os.path.join(C.CACHE, 'models')
PRED_PATH = os.path.join(MODEL_DIR, 'shape_predictor_68_face_landmarks.dat')
TEMPLATE_PATH = os.path.join(MODEL_DIR, 'template68.npy')
DET_LONG = 640
SUBSETS = {'all68': np.arange(68), 'stable': np.r_[27:48, 48:68],          # 코·눈·입 (턱선·눈썹 제외)
           'eyes_mouthout': np.r_[36:48, 48:60],                         # 확정: 눈 12점 + 바깥 입술 12점 (val NCC 중앙값 0.982, test 0.994)
           'five': None}                                                 # 눈 중심 2 + 코끝 + 입꼬리 2
_DET = _PRED = None


def template():
    if not os.path.exists(TEMPLATE_PATH):
        d = json.load(open(os.path.join(C.DATA, 'lm_json', 'train_81_FF++_processed_720.json')))['data']
        t = np.mean([np.asarray(e['aligned_lms'], np.float32)[:68] for e in d], 0)
        np.save(TEMPLATE_PATH, t)
    return np.load(TEMPLATE_PATH)


def five(p):
    return np.stack([p[36:42].mean(0), p[42:48].mean(0), p[30], p[48], p[54]]).astype(np.float32)


def _models():
    global _DET, _PRED
    if _DET is None:
        import dlib
        _DET = dlib.get_frontal_face_detector()
        _PRED = dlib.shape_predictor(PRED_PATH)
    return _DET, _PRED


def landmarks(frame_bgr):
    """→ (68,2) float32 원 해상도 좌표, 또는 None (얼굴 없음). 가장 큰 얼굴"""
    import dlib
    det, pred = _models()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    s = DET_LONG / max(H, W) if max(H, W) > DET_LONG else 1.0
    small = cv2.resize(gray, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else gray
    rects = det(small, 1)
    if len(rects) == 0:
        return None
    r = max(rects, key=lambda q: q.width() * q.height())
    rr = dlib.rectangle(int(r.left() / s), int(r.top() / s), int(r.right() / s), int(r.bottom() / s))
    shp = pred(gray, rr)
    return np.array([[shp.part(i).x, shp.part(i).y] for i in range(68)], np.float32)


def similarity(pts, subset='eyes_mouthout'):
    """68점 → 템플릿 닮음변환 (2×3), 최소제곱"""
    t = template()
    if subset == 'five':
        src, dst = five(pts), five(t)
    else:
        idx = SUBSETS[subset]
        src, dst = pts[idx], t[idx]
    # 2차원 닮음변환 최소제곱의 닫힌 해(반사 없음): R = [[a, −b], [b, a]], t = μ_dst − R μ_src (SVD 불필요 — BLAS 회피)
    ms, md = src.mean(0), dst.mean(0)
    x, y = src - ms, dst - md
    den = (x * x).sum()
    a = (x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]).sum() / den
    b = (x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]).sum() / den
    R = np.array([[a, -b], [b, a]], np.float64)
    t = md - np.array([R[0, 0] * ms[0] + R[0, 1] * ms[1], R[1, 0] * ms[0] + R[1, 1] * ms[1]])
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]]], np.float64)


def align(frame_bgr, subset='eyes_mouthout', pts=None, border=cv2.BORDER_CONSTANT):
    """→ (crop RGB uint8 224×224, M) 또는 (None, None)"""
    if pts is None:
        pts = landmarks(frame_bgr)
    if pts is None:
        return None, None
    M = similarity(pts, subset)
    if M is None:
        return None, None
    crop = cv2.warpAffine(frame_bgr, M, (224, 224), flags=cv2.INTER_LINEAR, borderMode=border, borderValue=0)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), M


def read_frames(path, want):
    """want: 프레임 번호 집합 → {번호: BGR}. 순차 디코딩(탐색 오차 없음)"""
    cap = cv2.VideoCapture(path)
    T = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out, t, mx = {}, 0, max(want) if want else -1
    while t <= mx:
        ok, fr = cap.read()
        if not ok:
            break
        if t in want:
            out[t] = fr
        t += 1
    cap.release()
    return out, T
