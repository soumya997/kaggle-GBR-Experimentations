from typing import List

import numpy as np
import torch
from torchvision.ops import box_iou


def calculate_score(
    preds: List[torch.Tensor],
    gts: List[torch.Tensor],
    iou_th: float
) -> float:
    num_tp = 0
    num_fp = 0
    num_fn = 0
    for p, gt in zip(preds, gts):
        if len(p) and len(gt):
            iou_matrix = box_iou(p, gt)
            tp = len(torch.where(iou_matrix.max(0)[0] >= iou_th)[0])
            fp = len(p) - tp
            fn = len(torch.where(iou_matrix.max(0)[0] < iou_th)[0])
            num_tp += tp
            num_fp += fp
            num_fn += fn
        elif len(p) == 0 and len(gt):
            num_fn += len(gt)
        elif len(p) and len(gt) == 0:
            num_fp += len(p)
    if (5 * num_tp + 4 * num_fn + num_fp )!=0:
        score = 5 * num_tp / (5 * num_tp + 4 * num_fn + num_fp )
    else:
        score = np.nan
    if (num_tp+num_fn) != 0:
        recall = num_tp/ (num_tp+num_fn)
    else:
        recall=np.nan
    if (num_tp+num_fp)!=0:
        precission = num_tp/ (num_tp+num_fp)
    else:
        precission=np.nan


    return score, precission, recall
def evaluate_f2(confthre):
    scores = []
    prec05 = []
    rec05 = []
    prec03 = []
    rec03 = []
    iou_ths = np.arange(0.3, 0.85, 0.05)
    with torch.no_grad():
        for images, targets in dl_val:
            model.eval()
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            preds = model(images)

            for i in range(len(images)):
                preds[i]['boxes']=preds[i]['boxes'].int()
                preds[i]['boxes']=preds[i]['boxes'][preds[i]['scores']>confthre]
                score = [calculate_score(preds[i]['boxes'].unsqueeze(0), targets[i]['boxes'].unsqueeze(0), iou_th)[0] for iou_th in iou_ths]
                scores.append(np.nanmean(score))
                prec05.append(calculate_score(preds[i]['boxes'].unsqueeze(0), targets[i]['boxes'].unsqueeze(0), 0.5)[1])
                prec03.append(calculate_score(preds[i]['boxes'].unsqueeze(0), targets[i]['boxes'].unsqueeze(0), 0.3)[1])
                rec05.append(calculate_score(preds[i]['boxes'].unsqueeze(0), targets[i]['boxes'].unsqueeze(0), 0.5)[2])
                rec03.append(calculate_score(preds[i]['boxes'].unsqueeze(0), targets[i]['boxes'].unsqueeze(0), 0.3)[2])
    print(f'F2 Score for confthre , {confthre}, :  {np.nanmean(scores):.3f} Precission .5: {np.nanmean(prec05):.3f} Precission .3: {np.nanmean(prec03):.3f}  Recall .5: {np.nanmean(rec05):.3f} Recall .3: {np.nanmean(rec03):.3f}')
