import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchmetrics.classification import (
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassF1Score,
)
import pytorch_lightning as pl

from .coattention_module import CoAttentionModule


class ReCAN(pl.LightningModule):
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int = 120,
        dim_feedforward: int = 2048,
        nhead: int = 6,
        num_layers: int = 1,
        mha_dropout: float = 0.5,
        dropout: float = 0.8,
        num_classes: int = 3,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super(ReCAN, self).__init__()
        self.save_hyperparameters()

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()
        self.test_precision = MulticlassPrecision(
            num_classes=num_classes, average="macro"
        )
        self.test_recall = MulticlassRecall(
            num_classes=num_classes, average="macro"
        )
        self.test_f1 = MulticlassF1Score(
            num_classes=num_classes, average="macro"
        )

        self.lr = lr
        self.weight_decay = weight_decay

        self.coattention = CoAttentionModule(
            dim_input,
            dim_hidden,
            dim_feedforward,
            nhead,
            num_layers,
            mha_dropout,
            dropout,
        )

        dim_coattention_output = dim_hidden * 8
        self.attn = nn.Linear(dim_coattention_output, 1, bias=False)
        self.fc = nn.Linear(dim_coattention_output, num_classes)

    def forward(self, x, lengths, masks, batch_lens):
        # casa_out shape (num_pairs, self.hidden_dim * 8)
        x = self.coattention(x, lengths, masks)
        x = self._condense_attn(x)
        x = self.fc(x)
        return x

    def _condense_attn(self, x: Tensor) -> Tensor:
        attn_values = self.attn(x)
        attn_values = F.softmax(attn_values, dim=0)
        x = x * attn_values
        weighted_out = torch.sum(x, axis=0)
        return weighted_out

    def training_step(self, batch, batch_idx):
        X, masks, lengths, batch_lens, y = batch
        y_hat = self.forward(X, masks, lengths, batch_lens)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1)
        self.train_acc(pred, y)

        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        X, masks, lengths, batch_lens, y = batch
        y_hat = self.forward(X, masks, lengths, batch_lens)
        loss = F.cross_entropy(y_hat, y)
        pred = y_hat.argmax(dim=1)
        self.valid_acc(pred, y)

        self.log("val_loss", loss)
        self.log("val_acc", self.valid_acc)

    def test_step(self, batch, batch_idx):
        X, masks, lengths, batch_lens, y = batch
        y_hat = self.forward(X, masks, lengths, batch_lens)
        pred = y_hat.argmax(dim=1)
        self.test_acc(pred, y)
        self.test_precision(pred, y)
        self.test_recall(pred, y)
        self.test_f1(pred, y)

        self.log("test_acc", self.valid_acc)
        self.log("test_precision", self.test_precision)
        self.log("test_recall", self.test_recall)
        self.log("test_f1", self.test_f1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ReCAN")
        parser.add_argument("--dim_hidden", type=int, default=120)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--nhead", type=int, default=6)
        parser.add_argument("--num_layers", type=int, default=1)
        parser.add_argument("--mha_dropout", type=float, default=0.5)
        parser.add_argument("--dropout", type=float, default=0.8)
        parser.add_argument("--lr", type=float, default=1e-4)
        parser.add_argument("--weight_decay", type=float, default=1e-5)
        return parent_parser
