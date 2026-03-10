import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from shutil import copy2
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


def _preprocess_semes_machine(machine_name, semes_labeled_dir, semes_out_test):
    """Worker function for single SEMES machine preprocessing."""
    f_name = f"{machine_name}.csv"
    
    file_path = os.path.join(semes_labeled_dir, f_name)
    df = pd.read_csv(file_path)
    
    labels = df['label'].values
    np.save(os.path.join(semes_out_test, f'{machine_name}_labels.npy'), labels)
    
    feature_cols = [col for col in df.columns if col not in ['label', 'anomaly_rules']]
    data = df[feature_cols].values
    np.save(os.path.join(semes_out_test, f'{machine_name}.npy'), data)


def preprocess_semes(data_dir, output_dir, num_workers=1):
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all machine files
    machine_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    machine_names = [f.replace('.csv', '') for f in machine_files]
    
    print(f"  - Processing {len(machine_files)} machines with {num_workers} workers...")
    
    if num_workers > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_preprocess_semes_machine, machine_name, 
                               data_dir, output_dir)
                for machine_name in machine_names
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="SEMES machines"):
                pass
    else:
        # Sequential processing
        for machine_name in tqdm(machine_names, desc="  SEMES machines"):
            _preprocess_semes_machine(machine_name, data_dir, output_dir)
    
    print(f"  ✓ SEMES preprocessed: {len(machine_files)} machines")


def main():
    parser = argparse.ArgumentParser(description='Preprocess datasets')
    parser.add_argument('--data-dir', type=str, default='./data_raw/SEMES_LABELED', help='Raw data directory')
    parser.add_argument('--output-dir', type=str, default='./data/SEMES/test', help='Output directory')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers for preprocessing')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of workers: {args.num_workers}")
    
    preprocess_semes(args.data_dir, args.output_dir, args.num_workers)

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print(f"Preprocessed data saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
