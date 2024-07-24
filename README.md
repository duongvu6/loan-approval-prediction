# Loan Approval Predict Project

## Problem Statement
  - This project aims to predict whether a loan should be approved based on 13 features, which include both categorical and numerical data. The models used for this task are RandomForestClassifier and XGBoostClassifier. The project includes various components for data preprocessing, model training, hyperparameter optimization, model registration, and deployment.

- Data Columns

  - Loan_id: the Number of Loan
  - no_of_dependents
  - education
  - self_employed
  - income_annum
  - loan_amount
  - loan_term
  - cibil_score
  - residential_assets_value
  - commercial_assets_value
  - luxury_assets_value
  - bank_asset_value
  - loan_status: Our target Column for prediction
 
## Project Structure

- config/: Contains configuration files for Grafana dashboards and data sources, and the MLflow database.
  - grafana_dashboards.yaml
  - grafana_datasources.yaml
  - mlflow.db

- dashboards/: Contains JSON files for Grafana dashboards.
  - data.json

- data/: Contains data files for training, validation, and raw data.
  - clear_data/: Processed data files.
    - train.pkl
    - transformer.pkl
    - val.pkl
  - raw_data/: Raw data file.
    - loan_approval.csv

- hpo/: Contains scripts for hyperparameter optimization.
  - models_hpo.py: Automatically starts hyperparameter optimization for the models.
  - rfc.py: Script for training RandomForestClassifier with hyperparameter optimization.
  - utils.py: Utility functions for hyperparameter optimization.
  - xgb.py: Script for training XGBoostClassifier with hyperparameter optimization.
  - tests/: Contains test scripts.
    - test_connection.py
    - test_data.py
    - test_preprocessing.py

- app.py: Starts a server that can receive requests and return predictions.
- check_accuracy.py: Retrieves the best model from MLflow, measures metrics on data from the database, and uploads the metrics back to the database.
- db.py: Handles connection to the PostgreSQL database with two tables, testadat for test data and accuracy for metrics.
- docker-compose.yml: Docker configuration file.
- init.py: Automates the execution of the project's tasks.
- preprocess_data.py: Responsible for data preprocessing, including the use of pipelines for categorical and numerical data. Saves processed data to PKL files.
- reg_model.py: Selects the best model and registers it as the production model.
- requirements.txt: Lists the dependencies required for the project.
- utils.py: Contains utility functions for the project.

## Getting Started

### Prerequisites

- Python 3.x
- PostgreSQL
- Docker (for containerized deployment)

### Installation

Install the required Python packages:
```   
pip install -r requirements.txt
```   

### Start ML Project

1) run mlflow server (ui)
```
mlflow server --backend-store-uri sqlite:///config/mlflow.db
```
2) run grafana and postgres
```
docker compose up
```
3) run init.py
```
python init.py
```
4) test the result
```      
curl -X POST -H 'Content-Type: application/json' -d '{"data": [3,"Graduate","No",5000000,12700000,14,865,4700000,8100000,19500000,6300000]}' http://127.0.0.1:8000/predict
```
