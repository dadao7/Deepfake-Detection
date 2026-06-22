"""
eval_utils.py
모든 노트북에서 공통으로 사용하는 평가 함수.
원본 stage_1_detection_inference.py의 parse_auc_score와 동일한 로직.
"""
import csv, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def parse_auc_score(result_csv):
    """원본 inference 스크립트와 동일한 video-level AUC/Accuracy 계산."""
    df = pd.read_csv(result_csv)
    last_image_id = ""
    pred_lst, gt_lst, pred_sample_lst = [], [], []

    for _, row in df.iterrows():
        image_id = "_".join(row['image_id'].split('_')[:-1])
        if image_id != last_image_id:
            last_image_id = image_id
            if pred_sample_lst:
                s = sorted(pred_sample_lst)[10:-10]
                pred_lst.append(np.mean(s))
                gt_lst.append(row['ground_truth'])
            pred_sample_lst = []
        pred_sample_lst.append(row['prediction'])

    # 마지막 비디오 처리
    if pred_sample_lst:
        s = sorted(pred_sample_lst)[10:-10]
        pred_lst.append(np.mean(s))
        gt_lst.append(df.iloc[-1]['ground_truth'])

    auc = roc_auc_score(gt_lst, pred_lst)
    best_acc = max(
        sum(1 for p, g in zip(pred_lst, gt_lst) if (1 if p >= t else 0) == int(g)) / len(gt_lst)
        for t in np.arange(0, 1.01, 0.0001)
    )
    return auc, best_acc


def save_results(img_id_lst, gt_lst, pred_lst, output_csv):
    import os
    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'ground_truth', 'prediction'])
        for img_id, gt, pred in zip(img_id_lst, gt_lst, pred_lst):
            writer.writerow([img_id, gt, pred])
