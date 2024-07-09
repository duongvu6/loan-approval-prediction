import pickle
    
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix
from scipy.stats import ks_2samp
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def calculate_metrics(y_true, y_pred, y_prob):
    roc_auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    gini_coefficient = 2 * roc_auc - 1
    ks_statistic = ks_2samp(y_prob[y_true == 0], y_prob[y_true == 1]).statistic
    log_loss_value = log_loss(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'gini_coefficient': gini_coefficient,
        'ks_statistic': ks_statistic,
        'log_loss': log_loss_value,
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn
    }
    
    return metrics