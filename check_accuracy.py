from db import get_test_data, insert_metrics
from utils import transform_data, load_model
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

def calculate_metrics(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    gini_coefficient = 2 * roc_auc - 1
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        'roc_auc': float(roc_auc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'gini_coefficient': float(gini_coefficient),
        'true_positive': int(tp),
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn)
    }
    
    return metrics

def main():
    target = "loan_status"
    data = get_test_data()

    # Transform data
    X_test = data.drop(columns=[target]).to_numpy()
    X_test = transform_data(X_test)
    y_test = data[[target]]

    model = load_model()
    
    y_pred = model.predict(X_test)

    accuracy = list(calculate_metrics(y_test, y_pred).values())
    return insert_metrics(accuracy)


if __name__ == "__main__":
    main()