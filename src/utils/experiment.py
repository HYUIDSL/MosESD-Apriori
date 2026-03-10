import os
import numpy as np

import dataset_config
from src.models.Apriori_MosESD import multi_osESD_with_apriori
from src.utils.data_loader import get_dataset, get_all_machines
from src.utils.runtime import set_random_seeds
from src.utils.metrics import evaluate_metrics


def run_experiment(args):
    set_random_seeds()
    dataset_key = args.dataset.upper()
    
    results = {
        'dataset': dataset_key,
        'config': dataset_config.DATASET_CONFIG[dataset_key]
    }
    
    print(f"\n{'='*70}")
    print(f"🚀 Experiment: {dataset_key}")
    print(f"{'='*70}")
    
    # Global 데이터셋 (PSM, SWAT, WADI) - 단일 파일
    if dataset_key in ['PSM', 'SWAT', 'WADI']:
        try:
            print(f"Loading {dataset_key} (Global Dataset)...")
            
            x_data, y_true = get_dataset(dataset_name=dataset_key)
    
            print(f"  - Data shape: {x_data.shape}")
            print(f"  - Label shape: {y_true.shape}")
            
            # 실험 수행
            anoms, comb = multi_osESD_with_apriori(x_data, args)
            f1, recall, precision, pa_f1, pa_recall, pa_precision = evaluate_metrics(y_true, anoms)
            
            results['metrics'] = {
                'f1': f1, 'recall': recall, 'precision': precision,
                'pa_f1': pa_f1, 'pa_recall': pa_recall, 'pa_precision': pa_precision,
                'best_combinations': comb
            }
            
            print(f"✅ {dataset_key} Results:")
            print(f"   F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
            print(f"   PA-F1: {pa_f1:.4f}, PA-Recall: {pa_recall:.4f}, PA-Precision: {pa_precision:.4f}")
            
        except Exception as e:
            print(f"❌ Error processing {dataset_key}: {e}")
            return None
    
    # Machine-wise 데이터셋 (SMD, MSL, SMAP) - 머신별 파일
    elif dataset_key in ['SMD', 'MSL', 'SMAP', 'SEMES']:
        try:
            # 머신 리스트 자동 스캔
            if args.log_num is None:
                print(f"Scanning machines for {dataset_key}...")
                machine_list = get_all_machines(dataset_key)
                print(f"Found {len(machine_list)} machines: {machine_list[:5]}{'...' if len(machine_list) > 5 else ''}")
            else:
                machine_list = [f'{args.log_num}']

            machine_metrics = {}
            all_metrics = {
                'f1': [], 'recall': [], 'precision': [],
                'pa_f1': [], 'pa_recall': [], 'pa_precision': []
            }
            
            # 각 머신별 처리
            for idx, machine_id in enumerate(machine_list):
                try:
                    if idx % 5 == 0:
                        print(f"Progress: {idx}/{len(machine_list)}")
                    
                    # 데이터 로드
                    x_data, y_true = get_dataset(dataset_name=dataset_key, log_num=machine_id)
                    
                    # 실험 수행
                    anoms, comb = multi_osESD_with_apriori(x_data, args)
                    f1, recall, precision, pa_f1, pa_recall, pa_precision = evaluate_metrics(y_true, anoms)
                    
                    # 결과 저장
                    machine_metrics[machine_id] = {
                        'f1': f1, 'recall': recall, 'precision': precision,
                        'pa_f1': pa_f1, 'pa_recall': pa_recall, 'pa_precision': pa_precision,
                        'best_combinations': comb
                    }
                    
                    # 전체 평균 계산용
                    all_metrics['f1'].append(f1)
                    all_metrics['recall'].append(recall)
                    all_metrics['precision'].append(precision)
                    all_metrics['pa_f1'].append(pa_f1)
                    all_metrics['pa_recall'].append(pa_recall)
                    all_metrics['pa_precision'].append(pa_precision)
                    
                except Exception as e:
                    print(f"⚠️  Machine {machine_id} error: {e}")
                    continue
            
            # 개별 머신 결과와 평균 계산
            results['machines'] = machine_metrics
            results['average'] = {
                'f1': np.mean(all_metrics['f1']),
                'recall': np.mean(all_metrics['recall']),
                'precision': np.mean(all_metrics['precision']),
                'pa_f1': np.mean(all_metrics['pa_f1']),
                'pa_recall': np.mean(all_metrics['pa_recall']),
                'pa_precision': np.mean(all_metrics['pa_precision'])
            }
            
            # 결과 출력
            num_success = len(machine_metrics)
            print(f"✅ {dataset_key} Results - {num_success}/{len(machine_list)} machines processed:")
            print(f"   Average F1: {results['average']['f1']:.4f}, Recall: {results['average']['recall']:.4f}, Precision: {results['average']['precision']:.4f}")
            print(f"   Average PA-F1: {results['average']['pa_f1']:.4f}, PA-Recall: {results['average']['pa_recall']:.4f}, PA-Precision: {results['average']['pa_precision']:.4f}")
            
        except Exception as e:
            print(f"❌ Error processing {dataset_key}: {e}")
            return None
    
    print(f"{'='*70}\n")
    return results