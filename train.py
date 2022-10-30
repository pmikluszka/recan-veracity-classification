from argparse import ArgumentParser
import mlflow
import os
from os.path import abspath, dirname, join
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


from recan_veracity_classification.model.recan import ReCAN
from recan_veracity_classification.preprocessing.pheme_data import (
    PHEMEDataModule,
)

MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", "http://0.0.0.0:5000"
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("ReCAN single model")
mlflow.pytorch.autolog(log_models=False)

CHECKPOINT_PATH = abspath(join(dirname(__file__), "./trained_models"))


parser = ArgumentParser()
parser = PHEMEDataModule.add_specific_args(parser)
args, _ = parser.parse_known_args()
dict_args = vars(args)
pheme = PHEMEDataModule(**dict_args, batch_size=16)

parser = ArgumentParser()
parser = ReCAN.add_model_specific_args(parser)
args, _ = parser.parse_known_args()
dict_args = vars(args)
recan = ReCAN(
    **dict_args, dim_input=pheme.dim_input, num_classes=pheme.num_classes
)

early_stopping_cb = EarlyStopping(monitor="val_loss", mode="min", patience=20)
trainer = pl.Trainer(
    profiler="simple",
    accelerator="gpu",
    default_root_dir=CHECKPOINT_PATH,
    callbacks=[early_stopping_cb],
)


with mlflow.start_run():
    trainer.fit(recan, pheme)
    trainer.test(recan, pheme)
