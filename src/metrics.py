# metrics.py
from __future__ import annotations

from typing import List, Tuple


def confusion_matrix(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    """
    Return (TP, FP, TN, FN) with positive class = 1.
    """
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1
        else:
            raise ValueError("Labels must be 0/1")
    return tp, fp, tn, fn


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    denom = tp + fp + tn + fn
    return (tp + tn) / denom if denom else 0.0


def _f1_for_positive(tp: int, fp: int, fn: int) -> float:
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    tp, fp, tn, fn = confusion_matrix(y_true, y_pred)
    f1_pos = _f1_for_positive(tp, fp, fn)
    # For class 0, swap roles: "positive" becomes 0
    f1_neg = _f1_for_positive(tn, fn, fp)
    return 0.5 * (f1_pos + f1_neg)
