import numpy as np

def ade(pred, gt):
    primary_pred = pred[0]
    primary_gt = gt[0]
    return np.mean(np.linalg.norm(primary_pred - primary_gt, axis=-1))

def fde(pred, gt):
    pred_last = pred[0, -1]   # primary
    gt_last = gt[0, -1]    # primary
    return np.linalg.norm(gt_last - pred_last)


def collision(path1, path2, person_radius=0.1, inter_parts=2):
    def getinsidepoints(p1, p2, parts=2):
        """return: equally distanced points between starting and ending "control" points"""

        return np.array((np.linspace(p1[0], p2[0], parts + 1),
                         np.linspace(p1[1], p2[1], parts + 1)))

    for i in range(len(path1) - 1):
        p1, p2 = [path1[i][0], path1[i][1]], [path1[i + 1][0], path1[i + 1][1]]
        p3, p4 = [path2[i][0], path2[i][1]], [path2[i + 1][0], path2[i + 1][1]]
        if np.min(np.linalg.norm(getinsidepoints(p1, p2, inter_parts) - getinsidepoints(p3, p4, inter_parts), axis=0)) \
           <= 2 * person_radius:
            return True
    return False

def pred_col(pred, gt):
    primary_pred = pred[0]
    for neigh in pred[1:]:
        if collision(primary_pred, neigh):
            return 1.0
    return 0.0


def gt_col(pred, gt):
    primary_pred = pred[0]
    for neigh in gt[1:]:
        if collision(primary_pred, neigh):
            return 1.0
    return 0.0

def topk_ade(preds, gt):
    topk_ade = 1e10
    for pred in preds:
        ade_m = ade(pred, gt)
        if ade_m < topk_ade:
            topk_ade = ade_m
    return topk_ade

def topk_fde(preds, gt):
    topk_fde = 1e10
    for pred in preds:
        fde_m = fde(pred, gt)
        if fde_m < topk_fde:
            topk_fde = fde_m
    return topk_fde

def trajnet_sample_eval(pred, gt):
    return ade(pred, gt), fde(pred, gt), pred_col(pred, gt), gt_col(pred, gt)

def trajnet_sample_multi_eval(preds, gt):
    return topk_ade(preds, gt), topk_fde(preds, gt)
