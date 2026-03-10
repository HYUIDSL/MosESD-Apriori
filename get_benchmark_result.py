import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from joblib import Parallel, delayed

import math
from math import e
from src.models.osESD import osESD
from src.utils.data_processing import index_to_preds
from src.utils.metrics import f1_score
from src.utils.runtime import set_random_seeds

import dataset_config
import time
from src.utils.data_loader import DataLoader, get_dataset, get_all_machines

def process_single_column(col_data, dwin_size, rwin_size, init_size, alpha, maxr):
    col_class = osESD(
        data=col_data,
        dwins=dwin_size,
        rwins=rwin_size,
        init_size=init_size,
        alpha=alpha,
        maxr=maxr,
        condition=True,
    )
    col_index = col_class.predict_all()
    return index_to_preds(col_index, len(col_data))


def multi_osESD_with_true_apriori(x_data, args):
    '''
    True Apriori 알고리즘으로 핵심 변수를 선별한 뒤,
    해당 변수들의 결과를 취합하여 최종 이상 여부를 판별하는 초경량 통합 함수
    '''
    # 파라미터 셋업 (args에서 로드, 없으면 기본값)
    rwin_size = getattr(args, 'rwin_size', 20)
    dwin_size = getattr(args, 'dwin_size', 20)
    init_size = getattr(args, 'init_size', 100)
    alpha = getattr(args, 'alpha', 0.01)
    maxr = getattr(args, 'maxr', 10)
    
    # Apriori 및 결합 파라미터
    min_support = getattr(args, 'apriori_support', 0.05) # 최소 지지도
    voting_threshold = getattr(args, 'voting_threshold', 1) # 1이면 단순 OR, 2 이상이면 N-Voting => 2로 설정하면 osESD 특성 상 안잡힘

    cols = x_data.columns.tolist()
    cols_data = [list(x_data[col]) for col in cols]
    
    # [Step 1] 전체 변수에 대해 병렬로 osESD 1회 수행 (초고속 연산)
    n_jobs = -1
    print(f"n_jobs: {n_jobs}")
    uni_col_preds = Parallel(n_jobs=n_jobs)(
        delayed(process_single_column)(
            col_data, dwin_size, rwin_size, init_size, alpha, maxr
        ) for col_data in cols_data
    )
    
    # [Step 2] Apriori 연관 규칙 탐색을 위한 Transaction 데이터프레임 구성
    anomaly_df = pd.DataFrame(uni_col_preds).T
    anomaly_df.columns = cols
    
    # 이상(1)이 한 번이라도 발생한 시점만 장바구니로 추출
    anomaly_transactions = anomaly_df[(anomaly_df.T != 0).any()]
    
    best_combinations = tuple(cols) # 기본값: 전체 변수 사용
    
    if not anomaly_transactions.empty:
        # mlxtend Apriori 적용
        frequent_itemsets = apriori(anomaly_transactions.astype(bool), min_support=min_support, use_colnames=True)
        
        if not frequent_itemsets.empty:
            frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(len)
            multi_itemsets = frequent_itemsets[frequent_itemsets['length'] >= 2]
            
            if not multi_itemsets.empty:
                # 2개 이상 묶인 조합 중 지지도(Support)가 가장 높은 묶음 선택
                best_itemset = multi_itemsets.sort_values(by=['support', 'length'], ascending=[False, False]).iloc[0]['itemsets']
            else:
                best_itemset = frequent_itemsets.sort_values(by='support', ascending=False).iloc[0]['itemsets']
                
            best_combinations = tuple(best_itemset)
        else:
            # 빈발 항목 집합이 없으면, 단순히 알람이 가장 많이 울린 Top 5 변수만 선택
            print("No frequent itemsets found. Falling back to top individual variables based on anomaly counts.")
            top_vars = anomaly_transactions.sum().sort_values(ascending=False).head(5).index
            best_combinations = tuple(top_vars)

    # [Step 3] 선별된 핵심 변수들의 결과만 추려내어 병합 (연산량 제로)
    # 굳이 subset을 만들어 osESD를 다시 돌릴 필요 없이, Step 1의 결과를 재활용합니다.
    selected_indices = [cols.index(col) for col in best_combinations]
    selected_preds = np.array(uni_col_preds)[selected_indices]  # shape: (선택된 변수 개수, Time)
    
    # 지정된 임계값(voting_threshold) 이상 알람이 울리면 최종 이상으로 판정
    anoms = (np.sum(selected_preds, axis=0) >= voting_threshold).astype(int).tolist()

    # [Step 4] 출력 포맷팅 및 반환 (통일된 형태)
    new_anoms = anoms
    new_df = x_data.copy()  # 원본 데이터프레임 복사 (필요 시 이상치 처리 등 추가 가능)

    return new_df, new_anoms, best_combinations


def pa_f1_score(y_true, y_pred):
    '''Point Adjustment(PA) F1 스코어 계산 함수'''
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    y_pred_adj = y_pred.copy()
    
    diff = np.diff(np.r_[0, y_true, 0])
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        if np.sum(y_pred[start:end]) > 0:
            y_pred_adj[start:end] = 1
    
    pa_f1, pa_recall, pa_precision = f1_score(y_true, y_pred_adj)
    return pa_f1, pa_recall, pa_precision


def evaluate_metrics(y_true, y_pred):
    f1, recall, precision = f1_score(y_true, y_pred)
    pa_f1, pa_recall, pa_precision = pa_f1_score(y_true, y_pred)

    return f1, recall, precision, pa_f1, pa_recall, pa_precision


def run_experiment_on_dataset(dataset_name, machine_list=None, verbose=True):
    """
    지정된 데이터셋에 대해 통합 실험을 수행합니다.
    
    Args:
        dataset_name (str): 'psm', 'smd', 'msl', 'smap', 'swat', 'wadi' 등
        machine_list (list): 특정 머신만 처리. None이면 모든 머신 자동 스캔
        verbose (bool): 상세 로그 출력 여부
        
    Returns:
        dict: 결과 사전 (metrics, machines, average 등)
    """
    set_random_seeds()
    dataset_key = dataset_name.upper()
    args = None
    
    # dataset_config에서 설정 확인
    if dataset_key not in dataset_config.DATASET_CONFIG:
        print(f"❌ Dataset '{dataset_name}' not found in dataset_config.py")
        return None
    
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
            if verbose:
                print(f"Loading {dataset_key} (Global Dataset)...")
            
            x_data, y_true = get_dataset(dataset_name=dataset_key)
            
            if verbose:
                print(f"  - Data shape: {x_data.shape}")
                print(f"  - Label shape: {y_true.shape}")
            
            # 실험 수행
            df, anoms, comb = multi_osESD_with_true_apriori(x_data, args)
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
    elif dataset_key in ['SMD', 'MSL', 'SMAP']:
        try:
            # 머신 리스트 자동 스캔
            if machine_list is None:
                if verbose:
                    print(f"Scanning machines for {dataset_key}...")
                machine_list = get_all_machines(dataset_key)
                if verbose:
                    print(f"Found {len(machine_list)} machines: {machine_list[:5]}{'...' if len(machine_list) > 5 else ''}")
            
            machine_metrics = {}
            all_metrics = {
                'f1': [], 'recall': [], 'precision': [],
                'pa_f1': [], 'pa_recall': [], 'pa_precision': []
            }
            
            # 각 머신별 처리
            for idx, machine_id in enumerate(machine_list):
                try:
                    if verbose and idx % 5 == 0:
                        print(f"Progress: {idx}/{len(machine_list)}")
                    
                    # 데이터 로드
                    x_data, y_true = get_dataset(dataset_name=dataset_key, log_num=machine_id)
                    
                    # 실험 수행
                    df, anoms, comb = multi_osESD_with_true_apriori(x_data, args)
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
                    if verbose:
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


def run_all_experiments():
    """모든 데이터셋에 대해 실험을 순차적으로 수행합니다."""
    
    all_results = {}
    
    # Global 데이터셋
    for dataset_name in ['psm', 'swat', 'wadi']:
        start_time = time.time()
        result = run_experiment_on_dataset(dataset_name)
        end_time = time.time()
        print(f"⏰ {dataset_name} experiment took {end_time - start_time:.2f} seconds")
        if result:
            all_results[dataset_name] = result
        return all_results
    
    # Machine-wise 데이터셋
    for dataset_name in ['smd', 'msl', 'smap']:
        start_time = time.time()
        result = run_experiment_on_dataset(dataset_name)
        end_time = time.time()
        print(f"⏰ {dataset_name} experiment took {end_time - start_time:.2f} seconds")
        if result:
            all_results[dataset_name] = result
    
    return all_results

def main():
    all_results = run_all_experiments()
    print(all_results)
    

if __name__ == "__main__":
    main()