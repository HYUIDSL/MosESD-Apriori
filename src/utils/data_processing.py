import numpy as np
import pandas as pd
from collections import Counter
import os
from datetime import datetime, timedelta


def index_to_preds(index_val, length):
    preds = [0 for _ in range(length)]
    for i in index_val:
        preds[i] = 1
    return preds