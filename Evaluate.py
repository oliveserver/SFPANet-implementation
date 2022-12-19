import numpy as np


def calc_hist(labels, preds, num_classes=21):
    hist = np.zeros((num_classes, num_classes))
    for label, pred in zip(labels, preds):
        label = np.array(label, dtype=np.int64).flatten()
        pred = np.array(pred, dtype=np.int64).flatten()
        tmp = label * num_classes + pred
        hist += np.bincount(tmp, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return hist


def global_acc(labels, preds):
    s = 0
    all = 1
    for i, j in zip(labels, preds):
        s += np.sum(np.array(i == j))
    for i in labels.shape:
        all *= i
    return s, all


def cal_iou(pred, target, num_classes):
    ious = []
    for cls in range(0, num_classes):
        pred_inds = pred == cls  # true, false matrix
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if target_inds.sum() == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def cal_pixel_acc(pred, target):
    correct = (pred == target).sum()
    total = (target == target).sum()
    return correct / total


if __name__ == "__main__":
    a = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
    b = np.array([1, 2, 1, 4, 4, 6]).reshape(2, 3)
    print(global_acc(a, b))
