import numpy as np
from sklearn.metrics import recall_score, precision_score

def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    denom = precision + recall
    f1_score = 0.0 if denom == 0 else 2 * (precision * recall) / denom
    return f1_score, recall, precision


def pa_f1_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    y_pred_adj = y_pred.copy()
    
    diff = np.diff(np.r_[0, y_true, 0])
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        if np.sum(y_pred[start:end]) > 0:
            y_pred_adj[start:end] = 1
    
    pa_f1, pa_recall, pa_precision = f1_score(y_true, y_pred_adj)
    return pa_f1, pa_recall, pa_precision


def get_metric_fn(metric: str, args=None):
    name = (metric or "f1").lower()

    if name in ("f1", "plain_f1", "basic_f1"):
        return f1_score

    if name in ("pa_f1", "point_adjustment_f1", "point_adjusted_f1"):
        return pa_f1_score

    raise ValueError(f"Unsupported metric: {metric}")


def evaluate_metrics(y_true, y_pred):
    f1, recall, precision = f1_score(y_true, y_pred)
    pa_f1, pa_recall, pa_precision = pa_f1_score(y_true, y_pred)
    
    return f1, recall, precision, pa_f1, pa_recall, pa_precision