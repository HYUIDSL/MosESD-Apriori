import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori
from joblib import Parallel, delayed

from src.models.osESD import osESD
from src.utils.data_processing import index_to_preds

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


def multi_osESD_with_apriori(x_data, args):
    print(f"args: {args}")
    cols = x_data.columns.tolist()
    cols_data = [list(x_data[col]) for col in cols]
    
    uni_col_preds = Parallel(n_jobs=args.num_workers)(
        delayed(process_single_column)(
            col_data, args.dwin_size, args.rwin_size, args.init_size, args.alpha, args.maxr
        ) for col_data in cols_data
    )
    
    anomaly_df = pd.DataFrame(uni_col_preds).T
    anomaly_df.columns = cols
    anomaly_transactions = anomaly_df[(anomaly_df.T != 0).any()]
    best_combinations = tuple(cols)
    
    if not anomaly_transactions.empty:
        frequent_itemsets = apriori(anomaly_transactions.astype(bool), min_support=args.apriori_support, use_colnames=True)
        
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

    selected_indices = [cols.index(col) for col in best_combinations]
    selected_preds = np.array(uni_col_preds)[selected_indices]
    anoms = (np.sum(selected_preds, axis=0) >= args.voting_threshold).astype(int).tolist()

    return anoms, best_combinations