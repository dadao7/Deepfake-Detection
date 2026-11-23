"""
================================================================================
[M2F2-Det Improved] server.py - 모바일 앱(dadao7/Deepfake-Detection, app 브랜치) 연동 API 서버
================================================================================
[실행]
   conda activate m2f2
   cd C:\\m2f2\\code
   python server.py                                   # 기본: improved_finetune_full_last_100ep
   python server.py --exp exp8_complete_20ep --port 8000

[API] (앱의 services/api.ts, app/loading.tsx 와 동일한 규격)
   POST /predict          multipart 'file'  →  {"job_id": "...", "status": "pending"}
   GET  /result/{job_id}  →  {"status": "pending"|"processing"|"done"|"failed", ...결과 필드}
                             결과 필드: label, fake_probability, ..., model_label, verdict_basis(model|c2pa|metadata),
                                        metadata(C2PA·AI 표시·EXIF), evidence(판정 근거 태그), source
                             ?images=1 이면 preview_image, face_image, heatmap_image(data URI) 포함
   POST /predict_url      JSON {"url", "single"} → 서버가 링크에서 이미지·영상을 받아 분석
                             (웹페이지면 대표 이미지/영상, 인스타그램 게시물은 전용 처리)
                             →  {"kind": "image", "job_id"} / {"kind": "video", "video_id"}
                                / {"kind": "batch", "batch_id", "thumbs"} (인스타그램 여러 장 게시물, single 이면 한 장)
   POST /predict_batch    multipart 'files' + 'urls' (합쳐서 최대 10개) →  {"batch_id": "...", "total": N, "status": "pending"}
   GET  /batch/{batch_id} →  {"status", "total", "done", "items": [장별 결과 (/result 와 같은 필드 + index, filename)]}
   GET  /batch/{batch_id}/items/{index}  →  한 장 결과 (?images=1 이면 이미지 포함)
   POST /predict_video    multipart 'file' (30초 이하 영상) →  {"video_id": "...", "total": 장면 수, "status": "pending"}
   GET  /video/{video_id} →  {"status", "total", "done", "frames": [장면별 결과], 종합 결과 필드...}
                             ?images=1 이면 장면 썸네일·가장 의심되는 장면 이미지(data URI) 포함
   GET  /health           →  서버/모델 상태 확인
================================================================================
"""
import os, sys, uuid, argparse, threading, traceback, tempfile, shutil, base64
from concurrent.futures import ThreadPoolExecutor

import cv2, numpy as np, torch, torch.nn.functional as F

import config as C
from evaluate_improved import load_improved_model  # torch.load weights_only 패치 포함

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import server_features as SF  # 메타데이터(C2PA·EXIF), 링크 받기, 히트맵 그림, 판정 근거 태그

IMG_SIZE = 224
# 얼굴 박스 주변 여유 비율. FF++ 학습 프레임에서 Haar 얼굴 폭/이미지 폭 ≈ 0.78 → margin ≈ 0.14
FACE_MARGIN = 0.14
# TTA: 여러 crop 비율 × 좌우반전으로 추론 후 평균 (한 번의 crop 에 결과가 흔들리지 않도록)
TTA_MARGINS = (0.10, 0.14, 0.20)
TTA_FLIP = True
DETECT_MAX_SIDE = 800  # 얼굴 검출 시 축소 크기 (큰 사진에서 검출률·속도 개선)
BATCH_MAX_FILES = 10   # 여러 장 분석 시 한 번에 받을 최대 장수
# 영상 분석: 영상 전체에서 장면을 고르게 뽑아 장면마다 판정한 뒤 종합
VIDEO_FRAMES = 16          # 뽑을 장면 수
VIDEO_MAX_SEC = 30         # 받을 수 있는 최대 영상 길이 (초)
URL_VIDEO_MAX_SEC = 180    # 링크 영상(인스타그램 등)은 길어도 거절하지 않고 앞 3분 안에서 장면을 뽑음
VIDEO_TRIM = 2             # 종합 확률: 가장 높은/낮은 장면을 2개씩 빼고 평균 (튀는 장면의 영향 줄이기)
VIDEO_MIN_FACE_FRAMES = 3  # 얼굴이 잡힌 장면이 이보다 적으면 전체 장면으로 판정
FACE_VIEW_SIZE = 448       # 앱에 보낼 얼굴·히트맵 비교 이미지 크기

app = FastAPI(title='M2F2-Det Improved API')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

MODEL = None
JOBS = {}                                     # job_id -> dict
BATCHES = {}                                  # batch_id -> {'status', 'total', 'done', 'items'}
VIDEOS = {}                                   # video_id -> {'status', 'total', 'done', 'frames', 종합 결과...}
# 결과 이미지(미리보기·얼굴·히트맵 data URI)는 용량이 커서 따로 보관 → ?images=1 로 요청할 때만 보냄
JOB_IMAGES = {}                               # job_id -> {'preview_image', 'face_image', 'heatmap_image'}
BATCH_IMAGES = {}                             # (batch_id, index) -> 위와 같음
JOBS_LOCK = threading.Lock()
EXECUTOR = ThreadPoolExecutor(max_workers=1)  # GPU 1개 → 순차 처리
FACE_CASCADE = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'))
FACE_CASCADE_ALT = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_alt2.xml'))


# ── 전처리 ────────────────────────────────────────────────────────────────
def decode_image(data: bytes):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('이미지를 디코딩할 수 없습니다.')
    return img


def detect_face(img_bgr):
    """가장 큰 얼굴 박스 (x, y, w, h) 반환. 기본 검출기 실패 시 alt2 / 완화된 조건으로 재시도."""
    H, W = img_bgr.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(H, W))
    small = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else img_bgr
    gray = cv2.equalizeHist(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    min_size = (max(40, int(min(small.shape[:2]) * 0.08)),) * 2
    for cascade, neighbors in ((FACE_CASCADE, 5), (FACE_CASCADE_ALT, 3), (FACE_CASCADE, 3)):
        faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=neighbors, minSize=min_size)
        if len(faces):
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            return tuple(v / scale for v in (x, y, w, h))
    return None


def has_face(data):
    """링크에서 받은 후보 이미지에 얼굴이 보이는지 (인스타그램 미리보기 이미지 고르기용)"""
    try:
        return detect_face(decode_image(data)) is not None
    except ValueError:
        return False


def crop_square(img_bgr, box, margin):
    """얼굴 박스 중심으로 정사각형 crop. 이미지 경계를 넘으면 잘라내지 않고 창을 안쪽으로 밀어 정사각형 유지."""
    H, W = img_bgr.shape[:2]
    x, y, w, h = box
    side = min(max(w, h) * (1 + 2 * margin), H, W)
    x0 = int(np.clip(x + w / 2 - side / 2, 0, W - side))
    y0 = int(np.clip(y + h / 2 - side / 2, 0, H - side))
    side = int(side)
    return img_bgr[y0:y0 + side, x0:x0 + side]


def crop_face(img_bgr, margin=FACE_MARGIN):
    """가장 큰 얼굴을 여유 영역과 함께 정사각형으로 crop. 실패 시 원본 반환."""
    box = detect_face(img_bgr)
    if box is None:
        return img_bgr, False
    return crop_square(img_bgr, box, margin), True


def to_tensor(img_bgr):
    # evaluate_improved.py 와 동일한 전처리: RGB, 224x224, [0,1], CHW
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(C.device)


# ── 결과 해석 ─────────────────────────────────────────────────────────────
def region_name(y, x):
    v = '상단(이마·눈썹)' if y < 1 / 3 else ('중앙(눈·코)' if y < 2 / 3 else '하단(입·턱)')
    h = '왼쪽' if x < 1 / 3 else ('가운데' if x < 2 / 3 else '오른쪽')
    return f'{v} {h}'


def heatmap_regions(hm_logit, k=C.K):
    hm = torch.sigmoid(hm_logit[0, 0]).float()
    Hh, Wh = hm.shape
    vals, idx = hm.flatten().topk(min(k, hm.numel()))
    regions = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        name = region_name((i // Wh + 0.5) / Hh, (i % Wh + 0.5) / Wh)
        if name not in [r[0] for r in regions]:
            regions.append((name, v))
    return regions


def build_response(fused_p, base_p, detail_p, alpha, regions, face_found):
    is_fake = fused_p >= 50.0
    label = 'fake' if is_fake else 'real'
    conf = fused_p if is_fake else 100 - fused_p
    strength = '매우 높은' if conf >= 90 else ('높은' if conf >= 70 else '다소 낮은')

    if is_fake:
        title, color = '딥페이크 의심', '#FF5A5F'
        summary = f'이 이미지는 딥페이크일 가능성이 {fused_p:.1f}%로 판단되었습니다. 얼굴 영역에서 합성 흔적으로 의심되는 패턴이 발견되었습니다.'
    else:
        title, color = '실제 이미지로 판단', '#4CD964'
        summary = f'이 이미지는 실제 이미지일 가능성이 {100 - fused_p:.1f}%로 판단되었습니다. 뚜렷한 위조 흔적이 발견되지 않았습니다.'

    reasons = [
        f'전역 분석(Baseline, CLIP+EfficientNet) 딥페이크 확률: {base_p:.1f}%',
        f'국소 분석(Detail Region) 딥페이크 확률: {detail_p:.1f}%',
        f'최종 판단 가중치: 전역 {alpha * 100:.0f}% / 국소 {(1 - alpha) * 100:.0f}%',
    ]
    if regions:
        reasons.append('모델이 집중한 의심 영역: ' + ', '.join(r[0] for r in regions))
    if not face_found:
        reasons.append('얼굴을 자동 검출하지 못해 이미지 전체로 분석했습니다. 정확도가 낮을 수 있습니다.')

    agree = (base_p >= 50) == (detail_p >= 50)
    explanation = (
        f'M2F2-Det 개선 모델은 이미지 전체의 맥락을 보는 전역 분석과, 히트맵으로 찾아낸 의심 영역 {len(regions)}곳을 '
        f'확대해 보는 국소 분석을 결합해 판단합니다. 전역 분석은 {base_p:.1f}%, 국소 분석은 {detail_p:.1f}%의 딥페이크 확률을 '
        f'보였고, 두 결과를 {alpha:.2f} : {1 - alpha:.2f} 비율로 융합한 최종 확률은 {fused_p:.1f}%입니다. '
        + ('두 분석 결과가 일치하여 ' if agree else '두 분석 결과가 서로 달라 판단에 주의가 필요하며, ')
        + f'최종적으로 {"딥페이크" if is_fake else "실제 이미지"}로 {strength} 신뢰도로 판단했습니다.'
    )

    return {
        'status': 'done',
        'label': label,
        'result_title': title,
        'result_color': color,
        'fake_probability': round(fused_p, 2),          # 앱이 toFixed(1)+'%' 로 표시 → 0~100 단위
        'baseline_fake_probability': round(base_p, 2),
        'detail_fake_probability': round(detail_p, 2),
        'summary': summary,
        'reasons': reasons,
        'llm_explanation': explanation,
        'face_detected': face_found,
    }


def tta_views(img_bgr):
    """TTA 입력 목록. 첫 번째는 기준 view(FACE_MARGIN, 반전 없음) → 히트맵 영역 표시에 사용."""
    box = detect_face(img_bgr)
    if box is None:
        crops = [img_bgr]
    else:
        margins = [FACE_MARGIN] + [m for m in TTA_MARGINS if m != FACE_MARGIN]
        crops = [crop_square(img_bgr, box, m) for m in margins]
    views = crops + ([cv2.flip(c, 1) for c in crops] if TTA_FLIP else [])
    return views, box is not None


@torch.no_grad()
def infer(img_bgr):
    """TTA 6회 추론의 원본 값 (view 별 확률, 융합 가중치, 첫 view 의 의심 영역 지도)"""
    views, found = tta_views(img_bgr)
    out = MODEL(torch.cat([to_tensor(v) for v in views]), return_dict=True)
    prob = lambda logit: (torch.softmax(logit.float(), dim=1)[:, 1] * 100).tolist()
    fused = prob(out['pred'])
    print(f'[TTA] fused per view: {[round(v, 1) for v in fused]}')
    return {
        'views': views,
        'found': found,
        'fused': fused,
        'base': prob(out['baseline_pred']),
        'detail': prob(out['detail_pred']),
        'alpha': float(out['dynamic_alpha'].float().mean().item()),
        'heatmap_logit': out['heatmap'][:1],
        'heat': torch.sigmoid(out['heatmap'][0, 0]).float().cpu().numpy(),  # 0~1 의심 영역 지도
    }


def run_inference(img_bgr):
    """기존 응답 형식 (check_test_images.py 등 테스트 스크립트도 사용)"""
    r = infer(img_bgr)
    return build_response(float(np.mean(r['fused'])), float(np.mean(r['base'])), float(np.mean(r['detail'])),
                          r['alpha'], heatmap_regions(r['heatmap_logit']), r['found'])


def apply_provenance(result, metadata, noun='이미지'):
    """출처 정보에 AI 생성·편집이 기록되어 있으면 최종 판정을 가짜로 (모델 확률은 그대로 둠)
    1순위 C2PA(서명 있음 → verdict_basis 'c2pa'), 2순위 XMP·PNG·EXIF 의 AI 표시(서명 없음 → 'metadata')"""
    c2 = (metadata or {}).get('c2pa') or {}
    marks = (metadata or {}).get('ai_marks') or {}
    result['verdict_basis'] = 'model'
    if c2.get('ai_generated') or c2.get('ai_edited'):
        mark, basis, where, title = c2, 'c2pa', 'C2PA 출처 정보', '출처 정보 확인'
    elif marks.get('ai_generated') or marks.get('ai_edited'):
        mark, basis, where, title = marks, 'metadata', f'메타데이터({marks["where"]}, 서명 없음)', '메타데이터 확인'
    else:
        return
    result.update({
        'label': 'fake',
        'verdict_basis': basis,
        'result_color': '#FF5A5F',
        'result_title': f'AI {"생성" if mark.get("ai_generated") else "편집"} {noun} ({title})',
        'summary': (f'파일에 담긴 {where}에 "{SF.tool_text(mark)}"(으)로 만든 '
                    f'{mark["source_label"].replace("이미지", noun)}로 기록되어 있어 가짜로 판정했습니다. '
                    f'(모델 분석 확률은 {result["fake_probability"]:.1f}%)'),
    })


def face_images(face_bgr, heat):
    """비교 슬라이더용: 얼굴 사진, 같은 사진 위에 의심 영역 지도를 겹친 사진"""
    face = cv2.resize(face_bgr, (FACE_VIEW_SIZE, FACE_VIEW_SIZE), interpolation=cv2.INTER_AREA)
    return {'face_image': to_data_uri(face, FACE_VIEW_SIZE, 85),
            'heatmap_image': to_data_uri(SF.heatmap_overlay(face, heat), FACE_VIEW_SIZE, 85)}


def analyze_image(img_bgr, metadata=None, source=None):
    """사진 한 장 분석 → (결과, 이미지). 결과에 메타데이터·C2PA·판정 근거 태그 포함"""
    r = infer(img_bgr)
    fused_p, base_p, detail_p = (float(np.mean(r[k])) for k in ('fused', 'base', 'detail'))
    result = build_response(fused_p, base_p, detail_p, r['alpha'], heatmap_regions(r['heatmap_logit']), r['found'])
    result['model_label'] = 'fake' if fused_p >= 50 else 'real'  # 모델만의 판정 (히트맵 표시 기준)
    apply_provenance(result, metadata)
    peaks = SF.heatmap_peaks(r['heat'], FACE_MARGIN) if r['found'] else []
    result['evidence'] = SF.build_evidence(base_p, detail_p, peaks, r['found'],
                                           result['model_label'] == 'fake', metadata)
    result['metadata'] = metadata or {}
    result['source'] = source or {'type': 'upload'}

    images = {'preview_image': to_data_uri(img_bgr, 720, 82)}
    if r['found']:
        images.update(face_images(r['views'][0], r['heat']))
    return result, images


def analyze_upload(data, source=None, mime_hint=None):
    """업로드·링크로 받은 원본 바이트 → (결과, 이미지). 메타데이터는 디코딩 전 원본에서 읽음"""
    metadata = SF.inspect_metadata(data=data, mime_hint=mime_hint)
    return analyze_image(decode_image(data), metadata, source)


def process_job(job_id, data, source=None, mime_hint=None):
    with JOBS_LOCK:
        JOBS[job_id]['status'] = 'processing'
    images = None
    try:
        result, images = analyze_upload(data, source, mime_hint)
    except Exception as e:
        traceback.print_exc()
        result = {'status': 'failed', 'error': str(e)}
    with JOBS_LOCK:
        JOBS[job_id] = result
        if images:
            JOB_IMAGES[job_id] = images
    print(f'[job {job_id}] {result.get("status")} {result.get("label", "")} {result.get("fake_probability", "")}')


def url_source(url, media):
    """링크로 받은 경우의 출처 정보 (앱의 메타데이터 카드·안내 문구에 표시)"""
    return {'type': 'url', 'url': url, 'media_url': media.get('media_url'), 'page_url': media.get('page_url'),
            'page_title': media.get('page_title'), 'site': media.get('site'), 'notice': media.get('notice')}


def process_batch(batch_id, uploads):
    """여러 장을 순서대로 판정. 한 장이 끝날 때마다 결과를 바로 반영해 앱이 진행 상황을 볼 수 있게 함.
    uploads[i] = (이름, 바이트, 링크, 출처 정보) — 링크 항목은 여기서 받아옴.
    출처 정보는 인스타그램 여러 장 게시물처럼 링크 하나에서 나온 사진들에 게시물 정보를 붙일 때 사용"""
    batch = BATCHES[batch_id]
    with JOBS_LOCK:
        batch['status'] = 'processing'
    for i, (name, data, url, given_source) in enumerate(uploads):
        with JOBS_LOCK:
            if batch['items'][i]['status'] != 'pending':  # 접수 단계에서 이미 실패 처리된 항목 (빈 파일)
                continue
            batch['items'][i] = {'index': i, 'filename': name, 'status': 'processing'}
        images = None
        try:
            source, mime_hint = given_source or {'type': 'upload'}, None
            if url:
                media = SF.fetch_media(url, face_check=has_face, single=True)
                if media['kind'] != 'image':
                    raise ValueError('이미지가 아닌 주소입니다. 영상은 영상 분석에서 확인해 주세요.')
                data, mime_hint = media['data'], media['mime']
                source = {**url_source(url, media), **(given_source or {})}
            result, images = analyze_upload(data, source, mime_hint)
        except Exception as e:
            traceback.print_exc()
            result = {'status': 'failed', 'error': str(e)}
        uploads[i] = None  # 처리가 끝난 이미지는 메모리에서 해제
        with JOBS_LOCK:
            batch['items'][i] = {'index': i, 'filename': name, **result}
            if images:
                BATCH_IMAGES[(batch_id, i)] = images
            batch['done'] += 1
        print(f'[batch {batch_id}] {i + 1}/{batch["total"]} {result.get("status")} '
              f'{result.get("label", "")} {result.get("fake_probability", "")}')
    with JOBS_LOCK:
        batch['status'] = 'done'


# ── 영상 분석 ─────────────────────────────────────────────────────────────
def read_video_info(path):
    """영상 기본 정보 (fps, 프레임 수, 길이, 크기). 읽을 수 없는 파일이면 None."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)  # 휴대폰 세로 영상의 회전 정보 반영
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok or fps <= 0 or count <= 0:
            return None
        h, w = frame.shape[:2]
        return {'fps': round(fps, 2), 'frame_count': count, 'duration_sec': round(count / fps, 2),
                'width': w, 'height': h}
    finally:
        cap.release()


def sample_frame_numbers(frame_count, n=VIDEO_FRAMES):
    """영상을 n 구간으로 나눠 각 구간 가운데 프레임 번호 (첫·마지막 프레임은 제외됨)."""
    n = min(n, frame_count)
    return sorted({int((i + 0.5) * frame_count / n) for i in range(n)})


def iter_video_frames(path, frame_numbers):
    """지정한 프레임만 순서대로 꺼냄 → (프레임 번호, BGR 이미지). 나머지 프레임은 grab 으로 건너뜀."""
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    pos = 0
    try:
        for target in frame_numbers:
            while pos < target and cap.grab():
                pos += 1
            ok, frame = cap.read()
            if not ok:
                return
            pos += 1
            yield target, frame
    finally:
        cap.release()


def to_data_uri(img_bgr, max_side, quality=80):
    """앱에 보낼 작은 JPEG (data URI)."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1:
        img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.tobytes()).decode('ascii')


def trimmed_mean(values, trim=VIDEO_TRIM):
    """양 끝 trim 개씩 빼고 평균. 값이 적으면 전체 평균."""
    values = sorted(values)
    if len(values) > 2 * trim + 2:
        values = values[trim:len(values) - trim]
    return float(np.mean(values))


def build_video_response(frames, info):
    """장면별 결과를 종합해 영상 전체 판정 (/result 와 같은 필드 + 영상 정보)."""
    analyzed = [f for f in frames if f['status'] == 'done']
    if not analyzed:
        raise ValueError('분석할 수 있는 장면이 없습니다.')
    with_face = [f for f in analyzed if f['face_detected']]
    face_ok = len(with_face) >= VIDEO_MIN_FACE_FRAMES
    used = with_face if face_ok else analyzed

    probs = [f['fake_probability'] for f in used]
    fused_p = trimmed_mean(probs)
    base_p = trimmed_mean([f['baseline_fake_probability'] for f in used])
    detail_p = trimmed_mean([f['detail_fake_probability'] for f in used])
    fake_frames = sum(p >= 50 for p in probs)
    key = max(used, key=lambda f: f['fake_probability'])
    trimmed = len(used) > 2 * VIDEO_TRIM + 2
    spread = max(probs) - min(probs)

    is_fake = fused_p >= 50.0
    conf = fused_p if is_fake else 100 - fused_p
    strength = '매우 높은' if conf >= 90 else ('높은' if conf >= 70 else '다소 낮은')

    if is_fake:
        title, color = '딥페이크 의심', '#FF5A5F'
        summary = (f'이 영상은 딥페이크일 가능성이 {fused_p:.1f}%로 판단되었습니다. '
                   f'분석한 장면 {len(used)}개 중 {fake_frames}개에서 합성 흔적이 의심됩니다.')
    else:
        title, color = '실제 영상으로 판단', '#4CD964'
        summary = (f'이 영상은 실제 영상일 가능성이 {100 - fused_p:.1f}%로 판단되었습니다. '
                   + (f'분석한 장면 {len(used)}개 중 {fake_frames}개에서만 의심 패턴이 보였습니다.' if fake_frames
                      else '의심 패턴이 보인 장면이 없습니다.'))
    if not face_ok:
        summary = '영상에서 얼굴이 잡힌 장면이 거의 없어 판정을 믿기 어렵습니다. ' + summary

    analyzed_sec = info.get('analyzed_sec', info['duration_sec'])
    if analyzed_sec < info['duration_sec'] - 1:  # 긴 링크 영상: 앞부분만 분석
        reasons = [f'영상 {info["duration_sec"]:.0f}초 중 앞 {analyzed_sec:.0f}초에서 장면 {len(analyzed)}개를 고르게 뽑아 분석']
    else:
        reasons = [f'영상 {info["duration_sec"]:.1f}초에서 장면 {len(analyzed)}개를 고르게 뽑아 분석']
    if face_ok:
        reasons.append(f'얼굴이 잡힌 장면 {len(with_face)}개로 종합 판정')
    else:
        reasons.append(f'얼굴이 잡힌 장면이 {len(with_face)}개뿐이라 전체 장면으로 판정했습니다. 정확도가 낮을 수 있습니다.')
    reasons += [
        f'장면별 딥페이크 확률: 최저 {min(probs):.1f}% · 중간값 {float(np.median(probs)):.1f}% · 최고 {max(probs):.1f}%',
        f'FAKE 로 판정된 장면: {fake_frames} / {len(used)}개',
        f'가장 의심되는 장면: {key["time_sec"]:.1f}초 ({key["fake_probability"]:.1f}%)',
        f'전역 분석(Baseline) {base_p:.1f}% · 국소 분석(Detail) {detail_p:.1f}%',
    ]
    if trimmed:
        reasons.append(f'종합 확률은 확률이 가장 높은·낮은 장면을 {VIDEO_TRIM}개씩 빼고 평균한 값입니다')

    explanation = (
        f'영상은 사진 한 장보다 많은 장면을 볼 수 있어 판단이 더 안정적입니다. 이 영상에서는 장면 {len(analyzed)}개를 '
        f'고르게 뽑아, 장면마다 얼굴을 찾고 자르기 3가지와 좌우 반전으로 6번씩 분석했습니다. '
        f'장면별 딥페이크 확률은 {min(probs):.1f}%~{max(probs):.1f}% 범위였고, '
        + ('장면에 따라 결과 차이가 커서 판단에 주의가 필요하며, ' if spread > 50 else '장면별 결과가 비교적 일관되며, ')
        + f'{"극단값을 뺀 " if trimmed else ""}평균 {fused_p:.1f}%를 최종 확률로 사용해 '
        + (f'{"딥페이크" if is_fake else "실제 영상"}로 {strength} 신뢰도로 판단했습니다.' if face_ok
           else f'{"딥페이크" if is_fake else "실제 영상"} 쪽으로 판단했지만, 얼굴이 잡힌 장면이 거의 없어 '
                f'이 결과는 참고용으로만 봐야 합니다.')
    )

    return {
        'label': 'fake' if is_fake else 'real',
        'model_label': 'fake' if is_fake else 'real',  # 모델만의 판정 (히트맵 표시 기준)
        'result_title': title,
        'result_color': color,
        'fake_probability': round(fused_p, 2),
        'baseline_fake_probability': round(base_p, 2),
        'detail_fake_probability': round(detail_p, 2),
        'summary': summary,
        'reasons': reasons,
        'llm_explanation': explanation,
        'face_detected': face_ok,
        'analyzed_frames': len(analyzed),
        'used_frames': len(used),
        'fake_frames': fake_frames,
        'key_frame_index': key['index'],
    }


def frame_result(r):
    """장면 하나의 추론 값 → 장면 결과 필드"""
    fused_p = float(np.mean(r['fused']))
    return {'status': 'done', 'label': 'fake' if fused_p >= 50 else 'real', 'fake_probability': round(fused_p, 2),
            'baseline_fake_probability': round(float(np.mean(r['base'])), 2),
            'detail_fake_probability': round(float(np.mean(r['detail'])), 2), 'face_detected': r['found']}


def process_video(video_id, path):
    """영상에서 장면을 뽑아 하나씩 판정 (끝날 때마다 바로 반영) → 마지막에 종합."""
    video = VIDEOS[video_id]
    previews = {}  # 장면 번호 -> 640px 이미지 (가장 의심되는 장면 표시용)
    faces = {}     # 장면 번호 -> (448px 얼굴 사진, 의심 영역 지도) — 가장 의심되는 장면의 히트맵 비교용
    try:
        with JOBS_LOCK:
            video['status'] = 'processing'
        numbers = [f['frame_no'] for f in video['frames']]
        read = 0
        for i, (frame_no, img) in enumerate(iter_video_frames(path, numbers)):
            read = i + 1
            with JOBS_LOCK:
                video['frames'][i]['status'] = 'processing'
            try:
                r = infer(img)
                result = frame_result(r)
                if r['found']:
                    faces[i] = (cv2.resize(r['views'][0], (FACE_VIEW_SIZE, FACE_VIEW_SIZE),
                                           interpolation=cv2.INTER_AREA), r['heat'])
            except Exception as e:
                traceback.print_exc()
                result = {'status': 'failed', 'error': str(e)}
            thumb = to_data_uri(img, 240, 75)
            previews[i] = to_data_uri(img, 640, 85)
            with JOBS_LOCK:
                video['frames'][i] = {**video['frames'][i], **result}
                video['thumbnails'][i] = thumb
                video['done'] += 1
            print(f'[video {video_id}] {i + 1}/{video["total"]} {result.get("label", "")} '
                  f'{result.get("fake_probability", "")}')

        with JOBS_LOCK:
            for f in video['frames'][read:]:  # 영상 끝까지 읽지 못한 장면
                f.update({'status': 'failed', 'error': '장면을 읽을 수 없습니다.'})
                video['done'] += 1
            frames = [dict(f) for f in video['frames']]
        summary = build_video_response(frames, video)

        key = summary['key_frame_index']
        key_peaks = SF.heatmap_peaks(faces[key][1], FACE_MARGIN) if key in faces else []
        apply_provenance(summary, video.get('metadata'), '영상')
        summary['evidence'] = SF.build_video_evidence(
            summary['baseline_fake_probability'], summary['detail_fake_probability'],
            summary['fake_frames'], summary['used_frames'], frames[key]['time_sec'], key_peaks,
            summary['face_detected'], summary['model_label'] == 'fake', video.get('metadata'))
        key_images = face_images(*faces[key]) if key in faces else {}

        with JOBS_LOCK:
            video.update(summary)
            video['key_frame_image'] = previews.get(key)
            video['key_face_image'] = key_images.get('face_image')
            video['key_heatmap_image'] = key_images.get('heatmap_image')
            video['status'] = 'done'
    except Exception as e:
        traceback.print_exc()
        with JOBS_LOCK:
            video['status'] = 'failed'
            video['error'] = str(e)
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


# ── API ───────────────────────────────────────────────────────────────────
@app.get('/health')
def health():
    return {'status': 'ok', 'model_loaded': MODEL is not None, 'device': str(C.device)}


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, '빈 파일입니다.')
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {'status': 'pending'}
    EXECUTOR.submit(process_job, job_id, data, {'type': 'upload'}, file.content_type)
    return {'job_id': job_id, 'status': 'pending'}


@app.get('/result/{job_id}')
def result(job_id: str, images: bool = False):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, '존재하지 않는 job_id 입니다.')
        job = dict(job)
        if images:  # 미리보기·얼굴·히트맵 이미지는 요청할 때만 포함 (진행 상황 확인 때는 제외)
            job.update(JOB_IMAGES.get(job_id, {}))
    return job


@app.post('/predict_batch')
async def predict_batch(files: list[UploadFile] | None = File(default=None),
                        urls: list[str] | None = Form(default=None)):
    """사진 파일(files)과 링크(urls)를 함께 받아 최대 10개를 순서대로 분석 (파일 먼저, 링크 다음)"""
    files, urls = files or [], [u.strip() for u in (urls or []) if u.strip()]
    if not files and not urls:
        raise HTTPException(400, '분석할 사진이나 링크가 없습니다.')
    if len(files) + len(urls) > BATCH_MAX_FILES:
        raise HTTPException(400, f'한 번에 최대 {BATCH_MAX_FILES}장까지 분석할 수 있습니다.')
    uploads = [(f.filename or f'image_{i + 1}', await f.read(), None, None) for i, f in enumerate(files)]
    uploads += [(url, None, url, None) for url in urls]  # 링크는 처리할 때 서버가 받아옴
    return create_batch(uploads)


def create_batch(uploads, **extra):
    """여러 장 분석 작업 등록 (사진·링크 업로드, 인스타그램 여러 장 게시물 공용).
    extra(게시물 제목·안내 문구 등)는 /batch/{batch_id} 응답에 그대로 포함"""
    items = [{'index': i, 'filename': name, 'status': 'pending'} if (data or url) else
             {'index': i, 'filename': name, 'status': 'failed', 'error': '빈 파일입니다.'}
             for i, (name, data, url, _) in enumerate(uploads)]
    batch_id = uuid.uuid4().hex
    with JOBS_LOCK:
        BATCHES[batch_id] = {'batch_id': batch_id, 'status': 'pending', 'total': len(items),
                             'done': sum(it['status'] == 'failed' for it in items), 'items': items, **extra}
    EXECUTOR.submit(process_batch, batch_id, uploads)
    return {'batch_id': batch_id, 'total': len(items), 'status': 'pending'}


@app.get('/batch/{batch_id}')
def batch_result(batch_id: str):
    with JOBS_LOCK:
        batch = BATCHES.get(batch_id)
        if batch is None:
            raise HTTPException(404, '존재하지 않는 batch_id 입니다.')
        return {**batch, 'items': [dict(it) for it in batch['items']]}


@app.get('/batch/{batch_id}/items/{index}')
def batch_item(batch_id: str, index: int, images: bool = False):
    """여러 장 분석 중 한 장의 결과 (상세 결과 화면용, ?images=1 이면 이미지 포함)"""
    with JOBS_LOCK:
        batch = BATCHES.get(batch_id)
        if batch is None or not 0 <= index < len(batch['items']):
            raise HTTPException(404, '존재하지 않는 결과입니다.')
        item = dict(batch['items'][index])
        if images:
            item.update(BATCH_IMAGES.get((batch_id, index), {}))
    return item


def create_video_job(path, source, mime_hint=None, clip_sec=None):
    """임시 파일로 저장된 영상 → 검사 후 분석 작업 등록 (업로드·링크 공용)
    clip_sec: 길이 제한으로 거절하지 않고 앞부분(초)에서만 장면을 뽑음 (링크 영상)"""
    info = read_video_info(path)
    if info is None:
        os.remove(path)
        raise HTTPException(400, '영상을 읽을 수 없습니다. mp4 또는 mov 영상을 올려 주세요.')
    if clip_sec is None and info['duration_sec'] > VIDEO_MAX_SEC + 1:
        os.remove(path)
        raise HTTPException(400, f'{VIDEO_MAX_SEC}초 이하 영상만 분석할 수 있습니다. '
                                 f'(올린 영상: {info["duration_sec"]:.0f}초)')
    frame_limit = info['frame_count'] if clip_sec is None else min(info['frame_count'], int(clip_sec * info['fps']))
    info['analyzed_sec'] = round(frame_limit / info['fps'], 2)

    metadata = SF.inspect_metadata(path=path, mime_hint=mime_hint)
    metadata['width'], metadata['height'] = info['width'], info['height']
    numbers = sample_frame_numbers(frame_limit)
    video_id = uuid.uuid4().hex
    frames = [{'index': i, 'frame_no': n, 'time_sec': round(n / info['fps'], 2), 'status': 'pending'}
              for i, n in enumerate(numbers)]
    with JOBS_LOCK:
        VIDEOS[video_id] = {'video_id': video_id, 'status': 'pending', 'total': len(frames), 'done': 0,
                            'frames': frames, 'thumbnails': {}, 'metadata': metadata, 'source': source, **info}
    EXECUTOR.submit(process_video, video_id, path)
    return {'video_id': video_id, 'total': len(frames), 'duration_sec': info['duration_sec'], 'status': 'pending'}


@app.post('/predict_video')
async def predict_video(file: UploadFile = File(...)):
    # OpenCV 는 파일 경로로 영상을 읽으므로 임시 파일로 저장 (큰 영상도 메모리에 다 올리지 않음)
    suffix = os.path.splitext(file.filename or '')[1].lower() or '.mp4'
    fd, path = tempfile.mkstemp(prefix='m2f2_video_', suffix=suffix)
    with os.fdopen(fd, 'wb') as out:
        shutil.copyfileobj(file.file, out)
    return create_video_job(path, {'type': 'upload'}, file.content_type)


VIDEO_IMAGE_KEYS = ('thumbnails', 'key_frame_image', 'key_face_image', 'key_heatmap_image')


@app.get('/video/{video_id}')
def video_result(video_id: str, images: bool = False):
    with JOBS_LOCK:
        video = VIDEOS.get(video_id)
        if video is None:
            raise HTTPException(404, '존재하지 않는 video_id 입니다.')
        data = {k: v for k, v in video.items() if k not in VIDEO_IMAGE_KEYS + ('frames',)}
        data['frames'] = [dict(f) for f in video['frames']]
        if images:  # 이미지는 용량이 커서 요청할 때만 포함 (진행 상황 확인 때는 제외)
            for f in data['frames']:
                f['thumbnail'] = video['thumbnails'].get(f['index'])
            for k in VIDEO_IMAGE_KEYS[1:]:
                data[k] = video.get(k)
    return data


class UrlRequest(BaseModel):
    url: str
    single: bool = False  # True: 인스타그램 여러 장·영상 게시물도 사진 한 장으로 (여러 장 결과의 다시 시도)


@app.post('/predict_url')
def predict_url(req: UrlRequest):
    """링크 분석: 서버가 주소에서 이미지·영상을 받아 분석. 웹페이지(기사·SNS)면 대표 이미지/영상을 찾음.
    응답의 kind 로 앱이 사진 분석(job_id) / 영상 분석(video_id) / 여러 장 분석(batch_id) 화면을 고름"""
    url = req.url.strip()
    try:
        media = SF.fetch_media(url, face_check=has_face, single=req.single)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(400, f'주소를 처리하지 못했습니다 ({e}).')

    source = url_source(url, media)
    common = {'media_url': media['media_url'], 'page_title': media['page_title'], 'notice': media.get('notice')}
    if media['kind'] == 'batch':  # 인스타그램 여러 장 게시물 → 사진마다 따로 판정
        uploads = [(item['name'], None, item['url'],
                    {**source, 'media_url': item['url'], 'notice': None,
                     'page_title': f'{media["page_title"]} ({item["name"]} 사진)'})
                   for item in media['items']]
        batch = create_batch(uploads, page_title=media['page_title'], page_url=media['page_url'],
                             notice=media.get('notice'))
        return {'kind': 'batch', **batch, 'thumbs': [item['thumb'] for item in media['items']], **common}

    if media['kind'] == 'image':
        job_id = uuid.uuid4().hex
        with JOBS_LOCK:
            JOBS[job_id] = {'status': 'pending'}
        EXECUTOR.submit(process_job, job_id, media['data'], source, media['mime'])
        return {'kind': 'image', 'job_id': job_id, 'status': 'pending', **common}

    suffix = {'video/quicktime': '.mov', 'video/webm': '.webm'}.get(media['mime'], '.mp4')
    fd, path = tempfile.mkstemp(prefix='m2f2_video_', suffix=suffix)
    with os.fdopen(fd, 'wb') as out:
        out.write(media['data'])
    return {'kind': 'video', **create_video_job(path, source, media['mime'], clip_sec=URL_VIDEO_MAX_SEC), **common}


def main():
    global MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='improved_finetune_full_last_100ep', help='weights/{exp}_best.pth')
    parser.add_argument('--alpha', default=None, help="'dynamic' 또는 숫자. 생략 시 체크포인트에 저장된 값 사용")
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    ckpt_path = C.get_ckpt_path(args.exp)
    if not os.path.exists(ckpt_path):
        sys.exit(f'[오류] 가중치 파일 없음: {ckpt_path}')
    alpha = args.alpha
    if alpha is None:
        alpha = torch.load(ckpt_path, map_location='cpu').get('alpha', 'dynamic')
    elif alpha != 'dynamic':
        alpha = float(alpha)

    MODEL = load_improved_model(args.exp, alpha)
    MODEL.eval()
    print(f'[서버 준비 완료] http://{args.host}:{args.port}  (exp={args.exp}, alpha={alpha}, device={C.device})')
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
