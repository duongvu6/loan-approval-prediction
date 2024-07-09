import subprocess
import tqdm
from dotenv import load_dotenv
import os
import pickle
import pandas as pd
import numpy as np

from mlflow import MlflowClient
import mlflow


load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
CLEAR_DATA_PATH = os.getenv("CLEAR_DATA_PATH")


DIR_PATH = os.getcwd()
transformer_path = os.path.join(DIR_PATH, CLEAR_DATA_PATH)
with open(os.path.join(transformer_path, "transformer.pkl"), "rb") as file:
    transformer = pickle.load(file)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

    
def transform_data(data):
    df = pd.DataFrame(data, columns=[
            "no_of_dependents", "education", "self_employed", "income_annum", 
            "loan_amount", "loan_term", "cibil_score", "residential_assets_value", 
            "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
        ])
    return transformer.transform(df)

def setup():
    steps = ["preprocess_data.py", "./hpo/models_hpo.py", "reg_model.py"]
    for step in tqdm.tqdm(steps):
        subprocess.run(['python', step], capture_output=True, text=True)

def load_model():
    run_id = list(filter(lambda run: run.current_stage == "Production", client.search_model_versions()))[0].run_id
    logged_model = f'runs:/{run_id}/model'

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model
