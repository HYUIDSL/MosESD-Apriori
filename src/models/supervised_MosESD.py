import numpy as np
import pandas as pd
import math
from math import e
from .logistic_regression import LogisticRegressionSequential
from .osESD import osESD
from src.utils.data_config import index_to_preds, create_pre_dataset, anom_replace
from sklearn.metrics import f1_score
from tqdm import tqdm  # tqdm 추가


def multi_osESD_supervised(x_data, y_data, args, train_percent, testing_size):

    # define the args
    rwin_size = args.rwin_size
    dwin_size = args.dwin_size
    init_size = args.init_size
    alpha = args.alpha
    maxr = args.maxr
    epochs = args.epochs
    early_stop = args.early_stop
    total_change_rate = args.total_change_rate
    total_o_change_rate = args.total_o_change_rate

    # set train_size
    if train_percent > 1:
        train_size = train_percent
    else:
        train_size = int(len(x_data) * train_percent)

    # set testing_size
    if testing_size > 1:
        test_return_size = len(x_data) - testing_size
    else:
        test_return_size = int(len(x_data) * round(1 - train_percent, 5))

    print(f"Testing_return_size : {test_return_size}")
    uni_col_preds = []

    cols = [col for col in x_data.columns]

    # Use osESD to calculate each column
    for col in cols:
        col_df = list(x_data[col])
        col_class = osESD(
            data=col_df,
            dwins=dwin_size,
            rwins=rwin_size,
            init_size=init_size,
            alpha=alpha,
            maxr=maxr,
            condition=True,
        )
        col_index = col_class.predict_all()
        uni_col_preds.append(index_to_preds(col_index, len(col_df)))

    Col_num = len(cols) * 4
    train_online_len = train_size - init_size
    transformed_data = [[-1 for _ in range(Col_num)] for _ in range(train_online_len)]

    # logistic regression learning rate (offline)
    # +1 means bias
    log_lr = [total_change_rate for _ in range(Col_num + 1)]
    log_lr_array = np.array(log_lr).reshape(1, -1)
    log_lr_array_transposed = log_lr_array.T
    lr_sum = sum(log_lr)

    # online logistic regression learning rate (online)
    # +1 means bias
    log_online_lr = [total_o_change_rate for _ in range(Col_num + 1)]
    log_online_lr_array = np.array(log_online_lr).reshape(1, -1)
    log_online_lr_array_transposed = log_online_lr_array.T
    online_lr_sum = sum(log_online_lr)

    # For comparing for early stop
    best_f1 = -1
    best_anoms = [0 for _ in range(len(x_data) - train_online_len - init_size)]
    early_stop_count = 0  # For counting consecutive stops.
    past_i_f1_cols = [-1 for _ in range(len(cols))]  # for saving past epoch f1 scores
    lr_update = [
        1 for _ in range(len(cols))
    ]  # for setting direction in moving next gradient
    col_idx_list = [i for i in range(len(past_i_f1_cols))]

    # train
    for train_epoch in tqdm(range(epochs), desc="Epochs"):
        # 각 열별 osESD 클래스 객체 생성
        CLASSES = {}
        for col in cols:
            col_df = list(x_data[col])
            col_class = osESD(
                data=col_df,
                dwins=dwin_size,
                rwins=rwin_size,
                init_size=init_size,
                alpha=alpha,
                maxr=maxr,
                condition=True,
            )
            CLASSES[col] = col_class

        # 학습 데이터 변환 (offline 부분)
        for i in range(train_online_len):
            for idx, col in enumerate(cols):  # 각 열에 대해
                col_class = CLASSES[col]
                c_val, r_val, c_anom, r_anom = col_class.test_values(i)
                transformed_data[i][idx * 4] = c_val
                transformed_data[i][idx * 4 + 1] = r_val
                transformed_data[i][idx * 4 + 2] = c_anom
                transformed_data[i][idx * 4 + 3] = r_anom
                _ = col_class.check_values(c_anom, r_anom)

        # logistic regression 학습 (offline)
        train_x = []
        train_y = []
        for idx in range(train_online_len):
            vals = transformed_data[idx]
            if sum(vals) == 0:
                if np.random.uniform(0, 1) > 0.1:
                    continue
            train_x.append(vals)
            train_y.append(y_data[idx + init_size])

        log_model = LogisticRegressionSequential()
        log_model.train(
            np.array(train_x),
            np.array(train_y).astype(int).reshape(-1, 1),
            num_iterations=100,
            lrs=log_lr_array_transposed,
        )

        # incremental training with tqdm progress bar
        anoms = []
        for i in tqdm(
            range(train_online_len, len(x_data) - init_size),
            desc="Incremental training",
            leave=False,
        ):
            row_vals = []
            real_y = y_data[i + init_size]
            for idx, col in enumerate(cols):  # 각 열에 대해
                col_class = CLASSES[col]
                results = col_class.test_values(i)  # c_val, r_val, c_anom, r_anom
                row_vals.extend(results)

            row_pred = log_model.predict(row_vals)[0]
            anoms.append(row_pred)

            # incremental 학습 및 loss 반환 (loss 값 출력)
            loss = log_model.train_incremental(
                np.array([row_vals]),
                np.array([[real_y]]),
                lrs=log_online_lr_array_transposed,
            )
            # tqdm.write 사용해서 진행바 위에 출력 (터미널에 loss 표시)
            if (i - train_online_len) % 1000 == 0:
                tqdm.write(
                    f"Epoch {train_epoch}, Index {i}, Incremental training loss: {loss}"
                )
            # anomaly 발생시 처리 (기존 코드 유지)
            if row_pred == 1:
                skip_next = False
                for single_anom_idx, single_anom_val in enumerate(row_vals):
                    if skip_next:
                        skip_next = False
                        continue
                    if single_anom_val == 1:
                        skip_next = True
                        col_idx = single_anom_idx // 4
                        col_class = CLASSES[cols[col_idx]]
                        if single_anom_idx % 2 == 0:
                            _ = col_class.check_values(
                                row_vals[single_anom_idx], row_vals[single_anom_idx + 1]
                            )
                        else:
                            _ = col_class.check_values(
                                row_vals[single_anom_idx - 1], row_vals[single_anom_idx]
                            )

        check = pd.DataFrame({"y": y_data[train_size : len(y_data)], "preds": anoms})
        recent_f1 = f1_score(check["y"], check["preds"])

        # Early stopping 처리
        if best_f1 > recent_f1:
            early_stop_count += 1
            if early_stop_count == early_stop:
                return best_anoms[-test_return_size:]
        else:
            early_stop_count = 0
            best_f1 = recent_f1
            best_anoms = anoms

        # 마지막 epoch이면 결과 반환
        if train_epoch == epochs - 1:
            return best_anoms[-test_return_size:]

        # learning rate 업데이트 (기존 코드 유지)
        check_f1s = []
        i_col_f1s = []

        for idx, col in enumerate(cols):
            uni_preds = uni_col_preds[idx][train_online_len + init_size :]
            i_col_f1s.append(1 - f1_score(uni_preds, anoms))
            check_f1s.append(f1_score(uni_preds, anoms))

        i_f1_sum = sum(i_col_f1s)

        for idx, past, now in zip(col_idx_list, past_i_f1_cols, i_col_f1s):
            if now < past:
                lr_update[idx] *= -1

        past_i_f1_cols = i_col_f1s
        i_f1_sum = sum(i_col_f1s)

        d_lr = [total_change_rate * i / i_f1_sum for i in i_col_f1s]
        d_o_lr = [total_o_change_rate * i / i_f1_sum for i in i_col_f1s]

        for col_idx in range(len(cols)):
            for times in range(4):
                log_lr[col_idx * 4 + times] += d_lr[col_idx] * lr_update[col_idx]
                log_online_lr[col_idx * 4 + times] += (
                    d_o_lr[col_idx] * lr_update[col_idx]
                )

        new_sum = sum(log_lr)
        log_lr = [i * lr_sum / new_sum for i in log_lr]

        new_o_sum = sum(log_online_lr)
        log_online_lr = [i * lr_sum / new_o_sum for i in log_online_lr]

        log_lr_array = np.array(log_lr).reshape(1, -1)
        log_lr_array_transposed = log_lr_array.T

        log_online_lr_array = np.array(log_online_lr).reshape(1, -1)
        log_online_lr_array_transposed = log_online_lr_array.T

        # total_o_change_rate 업데이트 (추가 로직 필요 시 수정)
        # total_o_change_rate *= e**(-k*i)
