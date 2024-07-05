import os
from dotenv import load_dotenv

import mlflow
from mlflow.models import infer_signature

import xgboost
from sklearn.model_selection import cross_val_score

from hyperopt import hp, STATUS_OK, fmin, tpe, Trials
from hyperopt.pyll import scope

from utils import check_accuracy, load_pickle

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
X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

train = xgboost.DMatrix(X_train, label=y_train)
valid = xgboost.DMatrix(X_val, label=y_val)

search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 1),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'objective': 'binary:logistic',
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 50),
    "seed": 42
}

def objective(params):
    with mlflow.start_run():
        # booster = xgboost.train(
        #     params=params,
        #     dtrain=train,
        #     num_boost_round=10,
        #     evals=[(valid, 'validation')],
        #     early_stopping_rounds=50,
        #     verbose_eval=False
        # )
        # preds = booster.predict(valid)
        # preds = (preds > 0.5).astype(int)
        # accuracy = check_accuracy(y_test, preds)
        # print(accuracy["accuracy_score"], params)
        params['max_depth'] = int(params['max_depth'])
        params['n_estimators'] = int(params['n_estimators'])
        params['min_child_weight'] = int(params['min_child_weight'])
        
        
        model = xgboost.XGBClassifier(**params, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        accuracy = check_accuracy(y_val, y_pred)

        mlflow.log_metrics(accuracy)
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)

        return {'loss': -accuracy["accuracy_score"], 'status': STATUS_OK}
    
def hpo():
    mlflow.set_experiment(HPO_EXPERIMENT_NAME)

    trials = Trials()
    best_params = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,
        trials=trials
    )
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['n_estimators'] = int(best_params['n_estimators'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])
    return best_params

def main(best_params):
    mlflow.set_experiment(EXPERIMENT_NAME)

    model = xgboost.XGBClassifier(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_val)
    
    accuracy = check_accuracy(y_val, y_pred)
    signature = infer_signature(X_test, y_pred)
    
    mlflow.log_params(best_params)
    mlflow.log_metrics(accuracy)
    mlflow.set_tag("best-model", "Best model")
    mlflow.set_tag("model", "xgboost")
    
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train[0],
        registered_model_name="best-xgboost-model",
    )


if __name__ == "__main__":
    best_params = hpo()
    main(best_params)