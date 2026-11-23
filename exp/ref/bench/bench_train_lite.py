import torch
import torch.nn.functional as F
import topk_crop

def crop_no_detach(images, heatmap_head, b3, k=3, crop_size=224, min_distance=2):
    """detach 만 뺀 같은 함수 (측정용 사본)"""
    Bn, _, H, W = images.shape
    hm_logit = heatmap_head(b3)
    hm_prob = torch.sigmoid(hm_logit)
    peaks, peak_vals = topk_crop.nms_heatmap(hm_prob, k=k, min_distance=min_distance)
    sy, sx = H / hm_logit.shape[2], W / hm_logit.shape[3]
    half, crops = crop_size // 2, []
    for ki in range(k):
        row = []
        for b in range(Bn):
            cy, cx = int(peaks[b, ki, 0].item() * sy), int(peaks[b, ki, 1].item() * sx)
            y1 = max(0, cy - half); y2 = y1 + crop_size
            if y2 > H: y2 = H; y1 = max(0, y2 - crop_size)
            x1 = max(0, cx - half); x2 = x1 + crop_size
            if x2 > W: x2 = W; x1 = max(0, x2 - crop_size)
            row.append(F.interpolate(images[b:b + 1, :, y1:y2, x1:x2], size=(crop_size, crop_size), mode='bilinear', align_corners=False)[0])
        crops.append(torch.stack(row))
    return torch.stack(crops, 1), hm_logit, peak_vals


