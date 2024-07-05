from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import pickle


def check_accuracy(y_test, predicton):
    df_acc = {
        "accuracy_score": accuracy_score(y_test, predicton),
        "roc_auc_score": roc_auc_score(y_test, predicton),
        "precision_score": precision_score(y_test, predicton),
        "recall_score": recall_score(y_test, predicton),
        "f1_score": f1_score(y_test, predicton)
    }

    return df_acc

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)