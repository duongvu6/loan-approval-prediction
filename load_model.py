import subprocess
import tqdm
from dotenv import load_dotenv
import os

from mlflow import MlflowClient
import mlflow


load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")
mlflow.set_experiment(EXPERIMENT_NAME)


client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

def setup():
    steps = ["preprocess_data.py", "./hpo/models_hpo.py", "reg_model.py"]
    for step in tqdm.tqdm(steps):
        subprocess.run(['python', step], capture_output=True, text=True)

def load_model():
    run_id = list(filter(lambda run: run.current_stage == "Production", client.search_model_versions()))[0].run_id
    logged_model = f'runs:/{run_id}/model'

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    return loaded_model
