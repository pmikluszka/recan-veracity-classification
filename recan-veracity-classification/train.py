import mlflow
import os
import torch

DEVICE = (
    f"cuda:{torch.cuda.current_device()}"
    if torch.cuda.is_available()
    else "cpu"
)

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "http://0.0.0.0:5000"
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment("ReCAN test")
