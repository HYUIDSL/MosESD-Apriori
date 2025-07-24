"""

data processing, creating data set module, save data

"""

import numpy as np
import pandas as pd
from collections import Counter
import os
from datetime import datetime, timedelta


def create_pre_dataset(L, cols_len):
    pre_dataset = []
    y = []
    for _ in range(L):
        if np.random.uniform(0, 1) > 0.2:
            y.append(0)
            row = [0] * (cols_len * 4)
        else:
            y.append(1)
            c = 0
            while c == 0:
                row = []
                for _ in range(cols_len):
                    if np.random.uniform(0, 1) < 0.3:
                        c += 1
                        r = np.random.uniform(0, 1)
                        if r < 0.25:
                            row += [np.random.uniform(3, 6), 0, 1, 0]
                        elif r < 0.5:
                            row += [0, np.random.uniform(3, 6), 0, 1]
                        else:
                            row += [
                                np.random.uniform(3, 6),
                                np.random.uniform(4, 8),
                                1,
                                1,
                            ]
                    else:
                        row += [0, 0, 0, 0]
        pre_dataset.append(row)

    df = pd.DataFrame(pre_dataset)
    return df.values, y


def index_to_preds(index_val, length):
    preds = [0 for _ in range(length)]
    for i in index_val:
        preds[i] = 1
    return preds


def anom_replace(orig_data, anom_preds):
    new_df = orig_data.copy()
    for col in new_df.columns:
        col_data = new_df[col]
        for i, pred in enumerate(anom_preds):
            if pred == 1:
                j = i + 1
                while j < len(anom_preds) and anom_preds[j] != 0:
                    j += 1
                if i > 0 and j < len(col_data):
                    mean_val = (col_data[i - 1] + col_data[j]) / 2
                    col_data = col_data.copy()
                    col_data[i] = int(mean_val)
        new_df[col] = col_data
    return new_df


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
RESULT_DIR = os.path.join(BASE_DIR, "result")  # 결과 저장 폴더 생성


def save_result_to_csv(best_y_pred, dataset: str, policy: str, anomaly_col: str):
    """
    예측 결과를 CSV 파일로 저장하는 함수
    :param best_y_pred: 예측 결과
    :param dataset: 데이터셋 이름 (예: labeled, unlabeled)
    :param policy: 정책 이름 (예: naive, hard, soft)
    """

    timestamp = (datetime.now() + timedelta(hours=9)).strftime(
        "%Y%m%d_%H%M%S"
    )  # KST 기준 시간
    result_dir = os.path.join(RESULT_DIR, timestamp[:-7])  # result/YYYYMMDD 폴더 생성
    os.makedirs(result_dir, exist_ok=True)  # 디렉토리 생성

    file_name = f"{dataset}_{policy}_{anomaly_col}_{timestamp}.csv"  # 파일명 지정
    result_file = os.path.join(result_dir, file_name)  # 파일 경로 설정

    # CSV 파일 저장
    best_y_pred = pd.Series(best_y_pred, name="y_pred")
    best_y_pred.to_csv(result_file, index=False, encoding="utf-8-sig")
