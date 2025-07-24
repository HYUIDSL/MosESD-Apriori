from src.models.Apriori_Alogrithm_for_MosESD import (
    feature_selection_using_apriori_algorhtim,
)
from src.models.basis import MosESD
import numpy as np
import pandas as pd
import argparse
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import os

from src.data.internal import load_dataset

from src.log.logger import setup_logger
from src.utils.data_config import save_result_to_csv

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def main(args):
    # load data
    args.dataset = args.dataset.lower()

    if args.dataset == "labeled":
        x_data, y_true = load_dataset(
            log_num=args.log_num, step=args.step, anomaly_col=args.anomaly_col
        )
    else:
        raise ValueError("Invalid dataset")

    # set logger
    logger = setup_logger(args.dataset, args.policy)

    # run
    if args.policy == "unsupervised":
        best_score, best_combinations = MosESD(x_data, y_true, args)
    elif args.policy in ["naive", "hard", "soft"]:
        best_score, best_combinations, best_y_pred = (
            feature_selection_using_apriori_algorhtim(x_data, y_true, args, logger)
        )
        save_result_to_csv(best_y_pred, args.dataset, args.policy, args.anomaly_col)
    else:
        raise ValueError("Invalid policy")

    # result
    print(best_score, best_combinations)
    logger.info(
        f"Hyper-parameters: log_num: {args.log_num}, step: {args.step}, policy: {args.policy}, anomaly_col: {args.anomaly_col}"
    )
    logger.info(f"{args.log_num}_{args.step}_Best score: {best_score}")
    logger.info(f"{args.log_num}_{args.step}_Best combinations: {best_combinations}")

    # csv에 log_num	step	policy	best score	best comb 저장
    if best_combinations:
        result_csv_path = os.path.join("log", f"{args.policy}.csv")
        with open(result_csv_path, "a") as f:
            f.write(
                f"{args.log_num},{args.step},{args.policy},{best_score},{best_combinations}\n"
            )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=" Unsupervised TAD, MosESD with Apriori Algorithm for feature selection"
    )
    parser.add_argument("--rwin_size", type=int, default=20, help="Time_step_TRES")
    parser.add_argument("--dwin_size", type=int, default=20, help="Time_step_TCHA")
    parser.add_argument(
        "--init_size", type=int, default=100, help="Initial dataset size"
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="")
    parser.add_argument(
        "--maxr", type=int, default=10, help="Maximum number of anomalies to detect"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs"
    )
    parser.add_argument(
        "--early_stop", type=int, default=3, help="Early stopping patience"
    )
    parser.add_argument("--total_change_rate", type=float, default=0.001, help="??")
    parser.add_argument("--total_o_change_rate", type=float, default=0.001, help="??")
    parser.add_argument(
        "--policy",
        type=str,
        default="hard",
        help="Feature selection policy ex) naive, hard, soft or unsupervised",
    )
    parser.add_argument(
        "--threshold", type=int, default=0.6, help="Threshold for feature selection"
    )
    parser.add_argument("--probability", type=int, default=0.2, help="??")

    parser.add_argument(
        "--dataset",
        type=str,
        default="labeled",
        help="Dataset name ex) labeled, unlabeled",
    )
    parser.add_argument("--log_num", type=int, default=260, help="Log_num of DATASET")
    parser.add_argument("--step", type=int, default=6, help="Step of DATASET")
    parser.add_argument(
        "--anomaly_col",
        type=str,
        default="label",
        help="Anomaly column name of DATASET",
    )

    args = parser.parse_args()

    main(args)
