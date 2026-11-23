"""SBI(Self-Blended Images, Shiohara & Yamasaki CVPR 2022) 생성기 — 공식 구현의 변환을 numpy/cv2로 새로 작성(albumentations 없음).
입력: 224×224 RGB uint8 얼굴 프레임 + 81점 랜드마크(h5 좌표). 출력: (진짜, SBI 가짜, 블렌드 마스크 M∈[0,1]) — 같은 쌍 증강 적용됨.
히트맵 GT: 경계 B = 4·M·(1−M) → 11×11(INTER_AREA) → 원본 heatmap_loss.generate_heatmap_gt (PROTOCOL §4.8)"""
import cv2
import numpy as np

# ── 원천(source) 변환: 공식 source_transforms ─────────────────────────


def rgb_shift(img, rng, lim=20):
    s = rng.uniform(-lim, lim, 3)
    return np.clip(img.astype(np.float32) + s, 0, 255).astype(np.uint8)


def hue_sat_val(img, rng, lim=0.3):
    """albumentations HueSaturationValue(±0.3, uint8): OpenCV HSV 단위 LUT 이동(값이 작아 거의 무변화 — 공식 설정 그대로)"""
    h_s, s_s, v_s = rng.uniform(-lim, lim, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    x = np.arange(256, dtype=np.int16)
    lut_h = np.mod(x + h_s, 180).astype(np.uint8)
    lut_s = np.clip(x + s_s, 0, 255).astype(np.uint8)
    lut_v = np.clip(x + v_s, 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def brightness_contrast(img, rng, b_lim=0.1, c_lim=0.1):
    alpha = 1.0 + rng.uniform(-c_lim, c_lim)
    beta = rng.uniform(-b_lim, b_lim) * 255.0
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


def random_downscale(img, rng):
    H, W = img.shape[:2]
    r = [2, 4][rng.integers(2)]
    ds = cv2.resize(img, (W // r, H // r), interpolation=cv2.INTER_NEAREST)
    return cv2.resize(ds, (W, H), interpolation=cv2.INTER_LINEAR)


def sharpen(img, rng):
    a = rng.uniform(0.2, 0.5)
    lt = rng.uniform(0.5, 1.0)
    k = (1 - a) * np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32) + \
        a * np.array([[-1, -1, -1], [-1, 8 + lt, -1], [-1, -1, -1]], np.float32)
    return np.clip(cv2.filter2D(img.astype(np.float32), -1, k), 0, 255).astype(np.uint8)


def source_transforms(img, rng):
    if rng.random() < 0.3:
        img = rgb_shift(img, rng)
    img = hue_sat_val(img, rng)
    img = brightness_contrast(img, rng)
    return random_downscale(img, rng) if rng.random() < 0.5 else sharpen(img, rng)


# ── 마스크 ────────────────────────────────────────────────────────────
def hull_mask(lm, H, W):
    m = np.zeros((H, W), np.float32)
    cv2.fillConvexPoly(m, cv2.convexHull(np.round(lm).astype(np.int32)), 1.0)
    return m


def rand_affine(src, mask, rng):
    """공식 randaffine: Affine(translate x±3%, y±1.5%, scale 0.95~1/0.95) → 마스크만 Elastic(alpha 50, sigma 7)"""
    H, W = mask.shape
    s = rng.uniform(0.95, 1 / 0.95)
    tx, ty = rng.uniform(-0.03, 0.03) * W, rng.uniform(-0.015, 0.015) * H
    M = cv2.getRotationMatrix2D((W / 2 - 0.5, H / 2 - 0.5), 0, s)
    M[0, 2] += tx
    M[1, 2] += ty
    src = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha, sigma = 50.0, 7.0
    dx = cv2.GaussianBlur((rng.random((H, W)).astype(np.float32) * 2 - 1), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random((H, W)).astype(np.float32) * 2 - 1), (17, 17), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    mask = cv2.remap(mask, xx + dx, yy + dy, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                     borderValue=0)
    return src, mask


def blend_mask(mask, rng):
    """공식 get_blend_mask: 임의 크기 리사이즈 → 블러 → 1 미만 버림 → 다시 블러 → 정규화"""
    H, W = mask.shape
    sh, sw = rng.integers(192, 257), rng.integers(192, 257)
    m = cv2.resize(mask, (int(sw), int(sh)))
    k1 = int(rng.choice(np.arange(5, 26, 2)))
    k2 = int(rng.choice(np.arange(5, 26, 2)))
    mb = cv2.GaussianBlur(m, (k1, k1), 0)
    if mb.max() <= 0:
        return None
    mb = mb / mb.max()
    mb[mb < 1] = 0
    mb = cv2.GaussianBlur(mb, (k2, k2), float(rng.integers(5, 46)))
    if mb.max() <= 0:
        return None
    mb = mb / mb.max()
    return cv2.resize(mb, (W, H))


def jpeg(img, q):
    ok, enc = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def self_blend(img, lm, rng, max_try=10):
    """→ real(uint8), fake(uint8), M(float32 H×W, 블렌드 비율을 곱하기 전 마스크, 최대 1), info(ratio 등)"""
    H, W = img.shape[:2]
    lm = np.asarray(lm, np.float32)
    for _ in range(max_try):
        use68 = rng.random() < 0.25
        mask = hull_mask(lm[:68] if use68 else lm, H, W)
        src, tgt = img.copy(), img.copy()
        src_aug = rng.random() < 0.5
        if src_aug:
            src = source_transforms(src, rng)
        else:
            tgt = source_transforms(tgt, rng)
        src, mask = rand_affine(src, mask, rng)
        mb = blend_mask(mask, rng)
        if mb is None:
            continue
        ratio = [0.25, 0.5, 0.75, 1, 1, 1][rng.integers(6)]
        mb = mb.astype(np.float32)                       # 비율 곱하기 전 마스크(최대 1) — 경계 GT·위치 평가용
        ms = mb * ratio                                  # 실제 블렌드 가중치
        fake = (ms[..., None] * src.astype(np.float32) + (1 - ms[..., None]) * tgt.astype(np.float32))
        fake = np.clip(fake, 0, 255).astype(np.uint8)
        # 공식 SBI는 target(=원본 쪽)을 진짜로 쓴다: 원천 변환이 target에 걸렸으면 그 target이 진짜
        real = tgt.astype(np.uint8)
        info = {'use68': bool(use68), 'src_aug': bool(src_aug), 'ratio': float(ratio)}
        # 쌍 공통 증강(공식 get_transforms, 같은 계수를 둘에 적용): RGBShift·HSV·밝기대비 각 p=0.3, 좌우반전 p=0.5, JPEG(40~100) p=0.5
        for fn in (rgb_shift, hue_sat_val, brightness_contrast):
            if rng.random() < 0.3:
                st = rng.bit_generator.state
                real = fn(real, rng)
                rng.bit_generator.state = st          # 같은 계수
                fake = fn(fake, rng)
        if rng.random() < 0.5:
            real, fake, mb = real[:, ::-1].copy(), fake[:, ::-1].copy(), mb[:, ::-1].copy()
            info['flip'] = True
        if rng.random() < 0.5:
            q = int(rng.integers(40, 101))
            real, fake = jpeg(real, q), jpeg(fake, q)
            info['jpeg'] = q
        return real, fake, mb, info
    raise RuntimeError('SBI 마스크 생성 실패')


def boundary_gt(mb, hm=11):
    """블렌드 마스크(비율 곱하기 전, 최대 1) → 11×11 경계 히트맵 GT (원본 heatmap_loss 함수 사용).
    비율을 곱한 마스크로 B를 만들면 비율 ≤0.5일 때 얼굴 내부 전체가 최대가 되어 경계가 아니게 됨 → 곱하기 전 마스크 사용"""
    from heatmap_loss import generate_heatmap_gt, get_boundary_mask
    b = get_boundary_mask(mb)[..., 0]                                   # 4·M·(1−M)
    b11 = cv2.resize(b, (hm, hm), interpolation=cv2.INTER_AREA)
    return generate_heatmap_gt(b11, (hm, hm), sigma_adaptive=True)[0]   # (11,11)
