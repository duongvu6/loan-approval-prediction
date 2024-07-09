import os
from dotenv import load_dotenv

from mlflow import MlflowClient
import mlflow


load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def set_stage_to_best_model():
    best_model = {}
    for run in client.search_model_versions():
        accuracy = client.get_run(run.run_id).data.metrics["f1_score"]
        if accuracy > best_model.get("accuracy", 0):
            best_model["run_id"] = run.run_id
            best_model["version"] = run.version
            best_model["name"] = run.name
            best_model["accuracy"] = accuracy

    client.set_registered_model_alias(name=best_model["name"], alias="Production", version=best_model["version"])
    client.transition_model_version_stage(
            name=best_model["name"],
            version=best_model["version"],
            stage="Production",
            archive_existing_versions=True
        )


if __name__ == "__main__":
    set_stage_to_best_model()
