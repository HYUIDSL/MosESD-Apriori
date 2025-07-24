from .Unsupervised_MosESD import multi_osESD_unsupervised
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import f1_score
from datetime import datetime
from tqdm import tqdm


"""

input: x_data, y_true, args(parameters for MosESD)
output: best_score, best_comb

"""


def feature_selection_using_apriori_algorhtim(
    x_data: pd.DataFrame, y_true: pd.Series, args, logger
):

    if args.policy == "naive":
        best_score, best_combinations, best_y_pred = naive_policy(
            x_data, y_true, args, logger
        )

    if args.policy == "hard":
        best_score, best_combinations, best_y_pred = hard_apriori_alg(
            x_data, y_true, args, logger
        )

    if args.policy == "soft":
        best_score, best_combinations, best_y_pred = soft_arpriori_alg(
            x_data, y_true, args, logger
        )

    return best_score, best_combinations, best_y_pred


def naive_policy(x_data: pd.DataFrame, y_true: pd.Series, args, logger):
    """
    policy 1 (naive policy)

    threshold 넘는 comb는 다 살리고, 구분없이 set으로 통합해서 줄여가는 방식으로 진행
    """
    selected_variables = set(x_data.columns)
    best_score = 0
    best_comb = None
    prev_best_score = 0
    best_y_pred = None

    for stage in range(1, len(x_data.columns) + 1):
        candidate_comb = list(combinations(selected_variables, stage))
        variable_score_dict = {}

        for comb in tqdm(candidate_comb, desc=f"Stage {stage} Progress", leave=False):
            subset_x_data = x_data[list(comb)]
            _, y_pred = multi_osESD_unsupervised(subset_x_data, args)
            score = f1_score(y_true, y_pred)

            if score > args.threshold:
                variable_score_dict[tuple(comb)] = score

                if score > best_score:
                    best_score = score
                    best_comb = comb
                    best_y_pred = y_pred

        selected_variables = set(
            col for comb in variable_score_dict.keys() for col in comb
        )
        logger.info(
            f"Stage {stage}\nselected variables: {selected_variables}\nbest score: {best_score}\nbest comb: {best_comb}"
        )

        if not variable_score_dict:
            break

        if best_score <= prev_best_score:
            break

        prev_best_score = best_score

    return best_score, best_comb, best_y_pred


def hard_apriori_alg(x_data: pd.DataFrame, y_true: pd.Series, args, logger):
    """

    stage 1: 각각의 variable에 대해 MosESD -> f1_score 기반 priority 부여
    stage >=2: 이전 stage(vaild 한 조합: >= threshold)를 pairwise로 union
    이때 stage N에 경우 N-2개의 우선순위가 동일해야 merge 가능

    """
    print("start hard_apriori_alg")

    best_score = 0
    best_comb = None
    prev_best_score = 0
    best_y_pred = None
    stage_dict = {}
    individual_priority = {}
    selected_variables = set(x_data.columns)

    for stage in range(1, len(x_data.columns) + 1):
        # print(f"stage {stage}")

        candidate_dict = {}

        if stage == 1:

            for comb in tqdm(
                combinations(selected_variables, stage),
                desc=f"Stage {stage} Progress",
                leave=False,
            ):
                union = tuple(sorted(comb))
                subset_x_data = x_data[list(union)]
                _, y_pred = multi_osESD_unsupervised(subset_x_data, args)
                score = f1_score(y_true, y_pred)
                if score > args.threshold:
                    candidate_dict[union] = (score, None)
                    if score > best_score:
                        best_score = score
                        best_comb = union
                        best_y_pred = y_pred
        # stage >= 2
        else:
            prev_combs = list(stage_dict.keys())
            for comb1, comb2 in tqdm(
                combinations(prev_combs, 2), desc=f"Stage {stage} Progress", leave=False
            ):
                union = tuple(sorted(set(comb1) | set(comb2)))
                if len(union) != stage:
                    continue

                if stage >= 3:
                    prio1 = stage_dict[comb1][1]
                    prio2 = stage_dict[comb2][1]

                    if prio1 is None or prio2 is None:
                        continue
                    if prio1[: stage - 2] != prio2[: stage - 2]:
                        continue

                if union in candidate_dict:
                    continue

                subset_x_data = x_data[list(union)]
                _, y_pred = multi_osESD_unsupervised(subset_x_data, args)
                score = f1_score(y_true, y_pred)

                if score > args.threshold:
                    # stage 1에서 부여한 개별 우선순위를 모아 priority tuple 생성
                    prio_tuple = tuple(
                        sorted([individual_priority[var] for var in union])
                    )
                    candidate_dict[union] = (score, prio_tuple)
                    if score > best_score:
                        best_score = score
                        best_comb = union
                        best_y_pred = y_pred

            # stage에서 평가된 조합이 없으면 종료
        if not candidate_dict:
            break

        # stage 1인 경우: 개별 변수 score 순으로 개별 우선순위 부여
        if stage == 1:
            sorted_stage1 = sorted(
                candidate_dict.items(), key=lambda x: x[1][0], reverse=True
            )
            new_dict = {}
            for rank, (comb, (score, _)) in enumerate(sorted_stage1, start=1):
                var = comb[0]  # 단일 변수이므로
                individual_priority[var] = rank
                new_dict[comb] = (score, (rank,))
            candidate_dict = new_dict

        # stage별 valid 조합 업데이트
        stage_dict = candidate_dict

        selected_variables = set(var for comb in stage_dict.keys() for var in comb)
        logger.info(
            f"Stage {stage}\nselected variables: {selected_variables}\nbest score: {best_score}\nbest comb: {best_comb}"
        )

        if best_score <= prev_best_score:
            break

        prev_best_score = best_score

    return best_score, best_comb, best_y_pred


def soft_arpriori_alg(x_data: pd.DataFrame, y_true: pd.Series, args, logger):
    """

    stage 1: 각각의 variable에 대해 MosESD -> f1_score 기반 priority 부여
    stage >=2: 이전 stage(vaild 한 조합: >= threshold)를 pairwise로 union
    이때 stage N에 경우 N-2개의 우선순위가 동일해야 merge 가능

    + stage_dict에 threshold 못 넘은 comb 중 일부를 선택해서 삽입

    """
    print("start soft_apriori_alg")

    best_score = 0
    best_comb = None
    prev_best_score = 0
    best_y_pred = None
    stage_dict = {}
    individual_priority = {}
    selected_variables = set(x_data.columns)

    for stage in range(1, len(x_data.columns) + 1):
        # print(f"stage {stage}")

        candidate_dict = {}

        if stage == 1:

            for comb in tqdm(
                combinations(selected_variables, stage),
                desc=f"Stage {stage} Progress",
                leave=False,
            ):
                union = tuple(sorted(comb))
                subset_x_data = x_data[list(union)]
                _, y_pred = multi_osESD_unsupervised(subset_x_data, args)
                score = f1_score(y_true, y_pred)
                if score > args.threshold:
                    candidate_dict[union] = (score, None)
                    if score > best_score:
                        best_score = score
                        best_comb = union
                        best_y_pred = y_pred

                # stage = 1 경우에서, args.probability의 확률로 candidate_dict에 union이 포함될 수 있게 진행
                else:
                    rand_val = np.random.rand()  # 0과 1사이의 flaot 반환
                    if rand_val <= args.probability:
                        candidate_dict[union] = (score, None)
        # stage >= 2
        else:
            prev_combs = list(stage_dict.keys())
            for comb1, comb2 in tqdm(
                combinations(prev_combs, 2), desc=f"Stage {stage} Progress", leave=False
            ):
                union = tuple(sorted(set(comb1) | set(comb2)))
                if len(union) != stage:
                    continue

                if stage >= 3:
                    prio1 = stage_dict[comb1][1]
                    prio2 = stage_dict[comb2][1]

                    if prio1 is None or prio2 is None:
                        continue
                    if prio1[: stage - 2] != prio2[: stage - 2]:
                        continue

                if union in candidate_dict:
                    continue

                subset_x_data = x_data[list(union)]
                _, y_pred = multi_osESD_unsupervised(subset_x_data, args)
                score = f1_score(y_true, y_pred)

                if score > args.threshold:
                    # stage 1에서 부여한 개별 우선순위를 모아 priority tuple 생성
                    prio_tuple = tuple(
                        sorted([individual_priority[var] for var in union])
                    )
                    candidate_dict[union] = (score, prio_tuple)
                    if score > best_score:
                        best_score = score
                        best_comb = union
                        best_y_pred = y_pred
                # args.probability로 threshold 못 넘은 친구 중에 집어넣기
                else:
                    rand_val = np.random.rand()
                    if rand_val <= args.probability:
                        prio_tuple = tuple(
                            sorted([individual_priority[var] for var in union])
                        )
                        candidate_dict[union] = (score, prio_tuple)

            # stage에서 평가된 조합이 없으면 종료
        if not candidate_dict:
            break

        # stage 1인 경우: 개별 변수 score 순으로 개별 우선순위 부여
        if stage == 1:
            sorted_stage1 = sorted(
                candidate_dict.items(), key=lambda x: x[1][0], reverse=True
            )
            new_dict = {}
            for rank, (comb, (score, _)) in enumerate(sorted_stage1, start=1):
                var = comb[0]  # 단일 변수이므로
                individual_priority[var] = rank
                new_dict[comb] = (score, (rank,))
            candidate_dict = new_dict

        # stage별 valid 조합 업데이트
        stage_dict = candidate_dict

        selected_variables = set(var for comb in stage_dict.keys() for var in comb)
        logger.info(
            f"Stage {stage}\nselected variables: {selected_variables}\nbest score: {best_score}\nbest comb: {best_comb}"
        )

        if best_score <= prev_best_score:
            break

        prev_best_score = best_score

    return best_score, best_comb, best_y_pred
