import os
from dotenv import load_dotenv

import mlflow
from mlflow.models import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

from hyperopt import hp, STATUS_OK, fmin, tpe, Trials
from hyperopt.pyll import scope

from utils import calculate_metrics, load_pickle

load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
HPO_EXPERIMENT_NAME = os.getenv("HPO_EXPERIMENT_NAME")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
CLEAR_DATA_PATH = os.getenv("CLEAR_DATA_PATH")


mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


DIR_PATH = os.getcwd()
data_path = os.path.join(DIR_PATH, CLEAR_DATA_PATH)

X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
# X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))


criterion = ['gini', 'entropy']
search_space = {
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 500, 10)),
    'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 5, 1)),
    'criterion': hp.choice('criterion', criterion)
}

def objective(params):
    with mlflow.start_run():
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        accuracy = calculate_metrics(y_val, y_pred, y_prob)
        cross_val_accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
        accuracy["cross_val_score"] = cross_val_accuracy

        mlflow.log_metrics(accuracy)
        mlflow.set_tag("model", "rfc")
        mlflow.log_params(params)

        return {'loss': -accuracy["cross_val_score"], 'status': STATUS_OK}
    
def hpo():
    mlflow.set_experiment(HPO_EXPERIMENT_NAME)
    
    best_params = fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=10, trials=Trials())

    best_params["criterion"] = criterion[best_params["criterion"]]
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_samples_split'] = int(best_params['min_samples_split'])
    best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

    return best_params


def main(best_params):
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    accuracy = calculate_metrics(y_val, y_pred, y_prob)
    signature = infer_signature(X_val, y_pred)
    
    mlflow.log_params(best_params)
    mlflow.log_metrics(accuracy)
    mlflow.set_tag("best-model", "Best model")
    
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train,
        registered_model_name=f"best-rfc-model",
    )


if __name__ == "__main__":
    best_params = hpo()
    main(best_params)

    