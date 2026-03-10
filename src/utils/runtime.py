import config
import numpy as np
import random

def set_random_seeds(seed=None):
    if seed is None:
        seed = config.RANDOM_SEED
    np.random.seed(seed)
    random.seed(seed)