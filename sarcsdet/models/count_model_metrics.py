import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scikitplot.metrics import plot_roc, plot_precision_recall
from sklearn.metrics import (f1_score, precision_score, average_precision_score, roc_auc_score,
                             classification_report, accuracy_score, make_scorer, recall_score)

from tqdm.auto import tqdm


def custom_scorer(y_true, y_pred):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return p * r * f1


def get_best_model_metrics(X, y, cv, best_estimator):
    scores_f1 = []
    scores_auc = []
    scores_acc = []
    scores_pr_auc = []

    for train_index, test_index in tqdm(cv.split(X, y), total=cv.get_n_splits(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        best_estimator.fit(X_train, y_train)
        y_pred = best_estimator.predict(X_test)
        y_pred_prob = best_estimator.predict_proba(X_test)[:, 1]
        
        scores_f1.append(f1_score(y_test, y_pred))
        scores_auc.append(roc_auc_score(y_test, y_pred_prob))
        scores_acc.append(accuracy_score(y_test, y_pred))
        scores_pr_auc.append(average_precision_score(y_test, y_pred_prob))

    print(f"F1: {np.mean(scores_f1):.5}")
    print(f"ROC-AUC: {np.mean(scores_auc):.5}")
    print(f"ACCURACY: {np.mean(scores_acc):.5}")
    print(f"PR-AUC: {np.mean(scores_pr_auc):.5}")

    
def get_test_classification_metrics(y_test, y_pred, y_pred_prob):
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred_prob)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    return {'F1': f1, 'Precision': precision, 'Recall': recall, 'PR_AUC': pr_auc, 'ROC_AUC': roc_auc}


def show_test_classification_metrics(y_test, y_pred, y_pred_prob, X_test=None, classifier=None, y_pred_probas=None):
    print(f"F1: {f1_score(y_test, y_pred):.5}")
    print(f"PREC: {precision_score(y_test, y_pred):.5}")
    print(f"PR-AUC: {average_precision_score(y_test, y_pred_prob):.5}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.5}")
    print('-------------------------------------------------------')
    print(classification_report(y_test, y_pred, labels=[0, 1]))
    print('-------------------------------------------------------')
    if classifier:
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].set_title('Precision-Recall curve')
        plot_precision_recall(y_test, y_pred_probas, ax=ax[0])
        ax[1].set_title('ROC-AUC curve')
        plot_roc(y_test, y_pred_probas, ax=ax[1])
        plt.show()
