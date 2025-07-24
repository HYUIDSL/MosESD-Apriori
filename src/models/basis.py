from .Unsupervised_MosESD import multi_osESD_unsupervised
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


def MosESD(x_data: pd.DataFrame, y_true: pd.Series, args):

    _, y_pred = multi_osESD_unsupervised(x_data, args)
    score = f1_score(y_true, y_pred)

    comb = None

    return score, comb
