import os
import re
import pandas as pd
import numpy as np
from typing import Tuple, List, Union, Optional
import dataset_config

# Dataset categorization
GLOBAL_DATASETS = ['PSM', 'WADI', 'SWAT']
MACHINE_WISE_DATASETS = ['SMD', 'MSL', 'SMAP', 'SEMES']

class DataLoader:
    def __init__(self, dataset_name: str):
        dataset_key = dataset_name.upper()
        if dataset_key not in dataset_config.DATASET_CONFIG:
            raise ValueError(f"Dataset '{dataset_name}' is not defined in dataset_config.py")
        
        self.dataset_name = dataset_key
        self.config = dataset_config.DATASET_CONFIG[dataset_key]

    def load_data(self, machine_id: Optional[str] = None):
        base_path = self.config["base_path"]
        
        if self.dataset_name in GLOBAL_DATASETS:
            test_data_path = os.path.join(base_path, 'test_data.npy')
            test_labels_path = os.path.join(base_path, 'test_labels.npy')
            
            X = np.load(test_data_path)
            y = np.load(test_labels_path)

            n_features = X.shape[1]
            columns = [f"feature_{i}" for i in range(n_features)]

            X = pd.DataFrame(X, columns=columns)
            y = pd.Series(y, name='label')
            
            return X, y
        
        elif self.dataset_name in MACHINE_WISE_DATASETS:
            if machine_id is None:
                raise ValueError(f"[{self.dataset_name}] machine_id is required")
            
            machine_id = os.path.splitext(machine_id)[0]
            
            test_data_path = os.path.join(base_path, 'test', f'{machine_id}.npy')
            test_labels_path = os.path.join(base_path, 'test', f'{machine_id}_labels.npy')
            
            if not os.path.exists(test_data_path):
                raise FileNotFoundError(f"Test data not found: {test_data_path}")
            
            X = np.load(test_data_path)
            
            if os.path.exists(test_labels_path):
                y = np.load(test_labels_path)
            else:
                y = np.zeros(len(X), dtype=int)
            
            n_features = X.shape[1]
            columns = [f"feature_{i}" for i in range(n_features)]
            X = pd.DataFrame(X, columns=columns)
            y = pd.Series(y, name='label')
            return X, y
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_name}")

    def scan_available_machines(self) -> List[str]:
        test_path = os.path.join(self.config["base_path"], 'test')
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"test directory not found: {test_path}")
        
        machines = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(test_path)
            if f.endswith('.npy') and not f.endswith('_labels.npy')
        )
        
        return machines


def get_dataset(dataset_name, log_num=None):
    loader = DataLoader(dataset_name)
    return loader.load_data(log_num)


def get_all_machines(dataset_name):
    loader = DataLoader(dataset_name)
    if dataset_name in MACHINE_WISE_DATASETS:
        return loader.scan_available_machines()
    else:
        return ['test_data']