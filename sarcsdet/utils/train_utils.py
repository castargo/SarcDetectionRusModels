import numpy as np
from tqdm import tqdm
from sarcsdet.models.count_model_metrics import get_test_classification_metrics


def chunks(list_like, n):
    for i in range(0, len(list_like), n):
        yield list_like[i:i + n]

        
def get_best_threshold(test_df, preds):
    best_prec = 0.0
    best_th = None
    for th in tqdm(np.linspace(0.1, 0.9, 250)):
        bilstm_test_metrics = get_test_classification_metrics(
            test_df.target.values, (preds > th).astype(int), preds)
        if bilstm_test_metrics['Precision'] > best_prec and bilstm_test_metrics['Recall'] >= 0.55:
            best_prec = bilstm_test_metrics['Precision']
            best_th = th
    return best_prec, best_th
