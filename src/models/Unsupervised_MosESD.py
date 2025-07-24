"""

Unsupervised_MosESD

input: data, params
output: new_df, new_anomalys

"""

import numpy as np
import pandas as pd
import math
from math import e
from .logistic_regression import LogisticRegressionSequential
from .osESD import osESD
from src.utils.data_config import index_to_preds, create_pre_dataset, anom_replace
from sklearn.metrics import f1_score


def multi_osESD_unsupervised(x_data, args):

    # comment : args로 받아서 처리해주면 될 거 같고

    rwin_size = args.rwin_size
    dwin_size = args.dwin_size
    init_size = args.init_size
    alpha = args.alpha
    maxr = args.maxr
    epochs = args.epochs
    early_stop = args.early_stop
    total_change_rate = args.total_change_rate
    total_o_change_rate = args.total_o_change_rate

    L = len(x_data)
    cols = [col for col in x_data.columns]

    uni_col_preds = []
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

    # creating the unsupervised part

    train_online_len = 0

    log_lr = [total_change_rate for _ in range(Col_num + 1)]
    log_lr_array = np.array(log_lr).reshape(1, -1)
    log_lr_array_transposed = log_lr_array.T
    lr_sum = sum(log_lr)

    log_online_lr = [total_o_change_rate for _ in range(Col_num + 1)]
    log_online_lr_array = np.array(log_online_lr).reshape(1, -1)

    past_i_f1_cols = [-1 for _ in range(len(cols))]  ### for saving past epoch f1 scores
    lr_update = [
        1 for _ in range(len(cols))
    ]  ### for setting direction in moving next gradient
    col_idx_list = [i for i in range(len(past_i_f1_cols))]

    k = math.log(2) / L  # for decaying lr, making it to 0.5 at the end

    for train_epoch in range(epochs):

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

        transformed_data, y_data = create_pre_dataset(1000, len(cols))

        train_x = []
        train_y = []

        for idx in range(1000):
            vals = transformed_data[idx]
            train_x.append(vals)
            train_y.append(y_data[idx])  # unsupervised

        log_model = LogisticRegressionSequential()
        log_model.train(
            np.array(train_x),
            np.array(train_y).astype(int).reshape(-1, 1),
            num_iterations=100,
            lrs=log_lr_array_transposed,
        )

        anoms = []

        for i in range(train_online_len, len(x_data) - init_size):
            row_vals = []
            for idx, col in enumerate(cols):  # cols
                col_class = CLASSES[col]
                results = col_class.test_values(i)  # c_val, r_val, c_anom, r_anom
                row_vals.append(results[0])
                row_vals.append(results[1])
                row_vals.append(results[2])
                row_vals.append(results[3])

            row_pred = log_model.predict(row_vals)[0]
            anoms.append(row_pred)

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
                            anom_val = col_class.check_values(
                                row_vals[single_anom_idx], row_vals[single_anom_idx + 1]
                            )
                        else:
                            anom_val = col_class.check_values(
                                row_vals[single_anom_idx - 1], row_vals[single_anom_idx]
                            )

        ### End of training
        if train_epoch == epochs - 1:

            break

        i_col_f1s = []
        testing_f1 = []  ## delete later
        for idx, col in enumerate(cols):
            uni_preds = uni_col_preds[idx][train_online_len + init_size :]
            i_col_f1s.append(1 - f1_score(uni_preds, anoms))
            testing_f1.append(f1_score(uni_preds, anoms))
        i_f1_sum = sum(i_col_f1s)

        for idx, past, now in zip(col_idx_list, past_i_f1_cols, i_col_f1s):
            if now < past:
                lr_update[idx] *= -1

        past_i_f1_cols = i_col_f1s
        i_f1_sum = sum(i_col_f1s)

        epsilon = 1e-6  # d_lr 0 되는 것을 방지하기 위한 epsilon

        d_lr = [total_change_rate * i / (i_f1_sum + epsilon) for i in i_col_f1s]
        d_o_lr = [total_o_change_rate * i / (i_f1_sum + epsilon) for i in i_col_f1s]

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

        total_o_change_rate *= e ** (-k * i)

    # print("Detected Anomalies : {}".format(sum(anoms)))
    new_anoms = [0] * init_size + anoms
    new_df = anom_replace(x_data, new_anoms)

    return new_df, new_anoms
