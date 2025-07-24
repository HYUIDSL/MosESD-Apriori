import numpy as np


def f1_score(y_true, y_pred):
    """
    실제 레이블과 예측 레이블을 입력받아 F1 score를 계산하는 함수입니다.

    Parameters:
        y_true (array-like): 실제 레이블.
        y_pred (array-like): 예측 레이블.

    Returns:
        float: 계산된 F1 score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # True Positive: 실제와 예측이 모두 1인 경우
    tp = np.sum((y_true == 1) & (y_pred == 1))
    # False Positive: 실제는 0인데 예측이 1인 경우
    fp = np.sum((y_true == 0) & (y_pred == 1))
    # False Negative: 실제는 1인데 예측이 0인 경우
    fn = np.sum((y_true == 1) & (y_pred == 0))

    # Precision과 Recall 계산
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # F1 Score 계산: Precision과 Recall의 조화평균
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return f1
