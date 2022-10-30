import mlflow
import os
import pytorch_lightning as pl

from recan_veracity_classification.model.recan import ReCAN
from recan_veracity_classification.preprocessing.pheme_data import (
    PHEMEDataModule,
)

# MLFLOW_TRACKING_URI = os.environ.get(
#     "MLFLOW_TRACKING_URI", "http://0.0.0.0:5000"
# )
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment("ReCAN test")

recan = ReCAN(768, use_lstm_out=True)
pheme = PHEMEDataModule()
trainer = pl.Trainer(fast_dev_run=True, accelerator="gpu")
trainer.fit(recan, pheme)
