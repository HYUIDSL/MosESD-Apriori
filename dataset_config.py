import os

BASE_DIR = (os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

DATASET_CONFIG = {
    # SEMES 데이터셋
    "SEMES": {
        "base_path": os.path.join(DATA_DIR, "SEMES"),
    },

    # PSM 데이터셋
    "PSM": {
        "base_path": os.path.join(DATA_DIR, "PSM"),
    },

    # MSL 데이터셋
    "MSL": {
        "base_path": os.path.join(DATA_DIR, "MSL"),
    },

    # SMAP 데이터셋
    "SMAP": {
        "base_path": os.path.join(DATA_DIR, "SMAP"),
    },

    # SMD 데이터셋
    "SMD": {
        "base_path": os.path.join(DATA_DIR, "SMD"),
    },

    # SWAT 데이터셋
    "SWAT": {
        "base_path": os.path.join(DATA_DIR, "SWAT"),
    },

    # WADI 데이터셋
    "WADI": {
        "base_path": os.path.join(DATA_DIR, "WADI"),
    }
}