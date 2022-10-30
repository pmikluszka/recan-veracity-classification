"""
This script tests ReCAN implementation using real data.
"""

import pytorch_lightning as pl

from recan_veracity_classification import PHEMEDataModule, ReCAN

pheme = PHEMEDataModule()

trainer = pl.Trainer(fast_dev_run=True, accelerator="gpu")
recan = ReCAN(pheme.dim_input, num_classes=pheme.num_classes)
trainer.fit(recan, pheme)

trainer = pl.Trainer(fast_dev_run=True, accelerator="gpu")
recan = ReCAN(pheme.dim_input, num_classes=pheme.num_classes, use_lstm_out=True)
trainer.fit(recan, pheme)
