import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from shutil import copy2
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed


def _read_swat_xlsx(path):
    """Read SWaT Excel files with special header handling."""
    df = pd.read_excel(path, header=1)
    df = df.loc[:, ~df.columns.astype(str).str.contains('^Unnamed')]
    return df


def preprocess_psm(data_dir, output_dir):
    """Preprocess PSM dataset."""
    print("\n[PSM] Preprocessing...")
    psm_dir = os.path.join(data_dir, 'PSM')
    psm_out = os.path.join(output_dir, 'PSM')
    os.makedirs(psm_out, exist_ok=True)
    
    # Train data
    print("  - Processing train data...")
    train_df = pd.read_csv(os.path.join(psm_dir, 'train.csv'))
    train_data = train_df.drop(columns=['timestamp_(min)']).values
    np.save(os.path.join(psm_out, 'train_data.npy'), train_data)
    
    # Test data
    print("  - Processing test data...")
    test_df = pd.read_csv(os.path.join(psm_dir, 'test.csv'))
    test_data = test_df.drop(columns=['timestamp_(min)']).values
    np.save(os.path.join(psm_out, 'test_data.npy'), test_data)
    
    # Test labels
    test_labels = pd.read_csv(os.path.join(psm_dir, 'test_label.csv'))['label'].values
    np.save(os.path.join(psm_out, 'test_labels.npy'), test_labels)
    
    # Train scaler
    print("  - Creating train scaler...")
    train_scaler = StandardScaler()
    train_scaler.fit(train_data)
    joblib.dump(train_scaler, os.path.join(psm_out, 'train_scaler.pkl'))
    
    print(f"  ✓ PSM preprocessed: train {train_data.shape}, test {test_data.shape}")


def preprocess_wadi(data_dir, output_dir):
    """Preprocess WADI dataset."""
    print("\n[WADI] Preprocessing...")
    wadi_dir = os.path.join(data_dir, 'WADI')
    wadi_out = os.path.join(output_dir, 'WADI')
    os.makedirs(wadi_out, exist_ok=True)
    
    # Train data (14 days normal)
    print("  - Processing train data...")
    train_df = pd.read_csv(os.path.join(wadi_dir, 'WADI_14days.csv'), skiprows=3)
    train_df = train_df.drop(columns=['Row', 'Date', 'Time'], errors='ignore')
    train_df = train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    train_data = train_df.values
    np.save(os.path.join(wadi_out, 'train_data.npy'), train_data)
    
    # Train scaler
    train_scaler = StandardScaler()
    train_scaler.fit(train_data)
    joblib.dump(train_scaler, os.path.join(wadi_out, 'train_scaler.pkl'))
    
    # Test data (attack data) - read once and reuse
    print("  - Processing test data...")
    test_df_full = pd.read_csv(os.path.join(wadi_dir, 'WADI_attackdata.csv'))
    
    # Parse datetime once
    test_df_full['DateTime'] = pd.to_datetime(
        test_df_full['Date'].astype(str) + ' ' + test_df_full['Time'].astype(str), 
        format='%m/%d/%Y %I:%M:%S.%f %p'
    )
    
    # Generate labels
    attack_labels_df = pd.read_csv(os.path.join(wadi_dir, 'WADI_attacklabels.csv'))
    test_labels = np.zeros(len(test_df_full), dtype=int)
    
    for _, row in attack_labels_df.iterrows():
        attack_start = pd.to_datetime(row['Date'] + ' ' + row['Start Time'], 
                                     format='%m/%d/%Y %H:%M:%S')
        attack_end = pd.to_datetime(row['Date'] + ' ' + row['End Time'], 
                                   format='%m/%d/%Y %H:%M:%S')
        attack_mask = (test_df_full['DateTime'] >= attack_start) & (test_df_full['DateTime'] <= attack_end)
        test_labels[attack_mask] = 1
    
    np.save(os.path.join(wadi_out, 'test_labels.npy'), test_labels)
    
    # Preprocess test data (drop datetime columns before processing)
    test_df = test_df_full.drop(['Row', 'Date', 'Time', 'DateTime'], axis=1, errors='ignore')
    test_df = test_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    test_data = test_df.values
    np.save(os.path.join(wadi_out, 'test_data.npy'), test_data)
    
    print(f"  ✓ WADI preprocessed: train {train_data.shape}, test {test_data.shape}")


def preprocess_swat(data_dir, output_dir):
    """Preprocess SWAT dataset."""
    print("\n[SWAT] Preprocessing...")
    swat_dir = os.path.join(data_dir, 'SWAT')
    swat_out = os.path.join(output_dir, 'SWAT')
    os.makedirs(swat_out, exist_ok=True)
    
    # Train data (normal)
    print("  - Processing train data...")
    train_df = _read_swat_xlsx(os.path.join(swat_dir, 'SWaT_Dataset_Normal_v1.xlsx'))
    train_df = train_df.drop(columns=['Timestamp', 'Normal/Attack'], errors='ignore')
    train_df = train_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    valid_cols = train_df.columns.tolist()
    joblib.dump(valid_cols, os.path.join(swat_out, 'valid_columns.pkl'))
    
    train_data = train_df.values
    np.save(os.path.join(swat_out, 'train_data.npy'), train_data)
    
    # Train scaler
    train_scaler = StandardScaler()
    train_scaler.fit(train_data)
    joblib.dump(train_scaler, os.path.join(swat_out, 'train_scaler.pkl'))
    
    # Test data (attack)
    print("  - Processing test data...")
    test_df = _read_swat_xlsx(os.path.join(swat_dir, 'SWaT_Dataset_Attack_v0.xlsx'))
    label_col = 'Normal/Attack'
    
    if label_col in test_df.columns:
        test_labels = (~test_df[label_col].astype(str).str.lower().eq('normal')).astype(int).values
    else:
        test_labels = np.zeros(len(test_df), dtype=int)
    
    np.save(os.path.join(swat_out, 'test_labels.npy'), test_labels)
    
    test_df = test_df.drop(columns=['Timestamp', label_col], errors='ignore')
    test_df = test_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    test_df = test_df.reindex(columns=valid_cols, fill_value=0)
    test_data = test_df.values
    np.save(os.path.join(swat_out, 'test_data.npy'), test_data)
    
    print(f"  ✓ SWAT preprocessed: train {train_data.shape}, test {test_data.shape}")


def _preprocess_smd_machine(machine_name, smd_train_dir, smd_test_dir, smd_label_dir, 
                            smd_out_train, smd_out_test, smd_out_scalers):
    """Worker function for single SMD machine preprocessing."""
    f_name = f"{machine_name}.txt"
    
    # Load train data
    train_path = os.path.join(smd_train_dir, f_name)
    train_data = np.loadtxt(train_path, delimiter=',')
    np.save(os.path.join(smd_out_train, f'{machine_name}.npy'), train_data)
    
    # Load test data
    test_path = os.path.join(smd_test_dir, f_name)
    test_data = np.loadtxt(test_path, delimiter=',')
    np.save(os.path.join(smd_out_test, f'{machine_name}.npy'), test_data)
    
    # Load test labels
    label_path = os.path.join(smd_label_dir, f_name)
    if os.path.exists(label_path):
        test_labels = np.loadtxt(label_path, delimiter=',')
        np.save(os.path.join(smd_out_test, f'{machine_name}_labels.npy'), test_labels)
    
    # Train scaler
    train_scaler = StandardScaler()
    train_scaler.fit(train_data)
    joblib.dump(train_scaler, os.path.join(smd_out_scalers, f'{machine_name}_train_scaler.pkl'))


def preprocess_smd(data_dir, output_dir, num_workers=1):
    """Preprocess SMD dataset with parallel processing."""
    print("\n[SMD] Preprocessing...")
    smd_train_dir = os.path.join(data_dir, 'SMD', 'train')
    smd_test_dir = os.path.join(data_dir, 'SMD', 'test')
    smd_label_dir = os.path.join(data_dir, 'SMD', 'test_label')
    
    smd_out_train = os.path.join(output_dir, 'SMD', 'train')
    smd_out_test = os.path.join(output_dir, 'SMD', 'test')
    smd_out_scalers = os.path.join(output_dir, 'SMD', 'scalers')
    
    os.makedirs(smd_out_train, exist_ok=True)
    os.makedirs(smd_out_test, exist_ok=True)
    os.makedirs(smd_out_scalers, exist_ok=True)
    
    # Get all machine files
    machine_files = [f for f in os.listdir(smd_train_dir) if f.endswith('.txt')]
    machine_names = [f.replace('.txt', '') for f in machine_files]
    
    print(f"  - Processing {len(machine_files)} machines with {num_workers} workers...")
    
    if num_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_preprocess_smd_machine, machine_name, 
                               smd_train_dir, smd_test_dir, smd_label_dir,
                               smd_out_train, smd_out_test, smd_out_scalers)
                for machine_name in machine_names
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="  SMD machines"):
                pass
    else:
        # Sequential processing
        for machine_name in tqdm(machine_names, desc="  SMD machines"):
            _preprocess_smd_machine(machine_name, smd_train_dir, smd_test_dir, smd_label_dir,
                                   smd_out_train, smd_out_test, smd_out_scalers)
    
    print(f"  ✓ SMD preprocessed: {len(machine_files)} machines")


def _preprocess_msl_smap_channel(machine_name, src_train_dir, src_test_dir, 
                                  dst_train_dir, dst_test_dir, dst_scalers_dir,
                                  anomalies_df):
    """Worker function for MSL/SMAP channel preprocessing."""
    import ast
    f_name = f"{machine_name}.npy"
    
    # Copy train data
    copy2(os.path.join(src_train_dir, f_name), os.path.join(dst_train_dir, f_name))
    
    # Load train data for scaler
    train_data = np.load(os.path.join(src_train_dir, f_name))
    train_scaler = StandardScaler()
    train_scaler.fit(train_data)
    joblib.dump(train_scaler, os.path.join(dst_scalers_dir, f'{machine_name}_train_scaler.pkl'))
    
    # Copy test data
    copy2(os.path.join(src_test_dir, f_name), os.path.join(dst_test_dir, f_name))
    
    # Load test data for enroll creation
    test_data = np.load(os.path.join(src_test_dir, f_name))
    
    # Generate test labels
    record = anomalies_df[anomalies_df['chan_id'] == machine_name]
    if len(record) > 0 and not pd.isna(record.iloc[0]['anomaly_sequences']):
        anomaly_ranges = ast.literal_eval(record.iloc[0]['anomaly_sequences'])
        test_labels = np.zeros(len(test_data), dtype=int)
        for anom_range in anomaly_ranges:
            start, end = anom_range[0], anom_range[1]
            if start < len(test_data):
                test_labels[start:min(end + 1, len(test_data))] = 1
    else:
        test_labels = np.zeros(len(test_data), dtype=int)
    
    np.save(os.path.join(dst_test_dir, f'{machine_name}_labels.npy'), test_labels)


def preprocess_msl_smap(data_dir, output_dir, dataset_name, num_workers=1):
    """Copy MSL/SMAP datasets (already in .npy format) with optional parallel processing."""
    print(f"\n[{dataset_name}] Copying (already .npy format)...")
    src_train_dir = os.path.join(data_dir, dataset_name, 'train')
    src_test_dir = os.path.join(data_dir, dataset_name, 'test')
    
    dst_train_dir = os.path.join(output_dir, dataset_name, 'train')
    dst_test_dir = os.path.join(output_dir, dataset_name, 'test')
    dst_scalers_dir = os.path.join(output_dir, dataset_name, 'scalers')
    
    os.makedirs(dst_train_dir, exist_ok=True)
    os.makedirs(dst_test_dir, exist_ok=True)
    os.makedirs(dst_scalers_dir, exist_ok=True)
    
    # Get all machine files
    machine_files = [f for f in os.listdir(src_train_dir) if f.endswith('.npy')]
    machine_names = [f.replace('.npy', '') for f in machine_files]
    
    # Read labeled anomalies
    anomalies_df = pd.read_csv(os.path.join(data_dir, dataset_name, 'labeled_anomalies.csv'))
    
    print(f"  - Processing {len(machine_files)} channels with {num_workers} workers...")
    
    if num_workers > 1:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_preprocess_msl_smap_channel, machine_name,
                               src_train_dir, src_test_dir, dst_train_dir, dst_test_dir, 
                               dst_scalers_dir, anomalies_df)
                for machine_name in machine_names
            ]
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"  {dataset_name} channels"):
                pass
    else:
        # Sequential processing
        for machine_name in tqdm(machine_names, desc=f"  {dataset_name} channels"):
            _preprocess_msl_smap_channel(machine_name, src_train_dir, src_test_dir,
                                        dst_train_dir, dst_test_dir, dst_scalers_dir,
                                        anomalies_df)
    
    print(f"  ✓ {dataset_name} copied and indexed: {len(machine_files)} channels")


def main():
    parser = argparse.ArgumentParser(description='Preprocess datasets')
    parser.add_argument('--data-dir', type=str, default='./data_raw', help='Raw data directory')
    parser.add_argument('--output-dir', type=str, default='./data', help='Output directory')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel workers for preprocessing')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['PSM', 'WADI', 'SWAT', 'SMD', 'MSL', 'SMAP'],
                       help='Datasets to preprocess')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Preprocessing datasets: {args.datasets}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of workers: {args.num_workers}")
    
    for dataset in args.datasets:
        dataset = dataset.upper()
        try:
            if dataset == 'PSM':
                preprocess_psm(args.data_dir, args.output_dir)
            elif dataset == 'WADI':
                preprocess_wadi(args.data_dir, args.output_dir)
            elif dataset == 'SWAT':
                preprocess_swat(args.data_dir, args.output_dir)
            elif dataset == 'SMD':
                preprocess_smd(args.data_dir, args.output_dir, args.num_workers)
            elif dataset == 'MSL' or dataset == 'SMAP':
                preprocess_msl_smap(args.data_dir, args.output_dir, dataset, args.num_workers)
            else:
                print(f"  [WARNING] Unknown dataset: {dataset}")
        except Exception as e:
            print(f"  [ERROR] Failed to preprocess {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print(f"Preprocessed data saved to: {args.output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
