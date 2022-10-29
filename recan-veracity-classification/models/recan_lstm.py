import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
)

from .coattention_module import CoAttentionModule


class ReCANLSTM(nn.Module):
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
        lstm_layers: int = 1,
        lstm_dim: int = 240,
    ):
        super(ReCANLSTM, self).__init__()

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
        self.cond_lstm = nn.LSTM(
            dim_coattention_output, lstm_dim, lstm_layers, batch_first=True
        )
        self.fc = nn.Linear(lstm_dim, num_classes)

    def forward(self, x, lengths, masks, batch_lens):
        # casa_out shape (num_pairs, self.hidden_dim * 8)
        x = self.coattention(x, lengths, masks)

        c = torch.cumsum(torch.tensor([0] + batch_lens), 0)
        x = [x[i:j] for i, j in zip(c[:-1], c[1:])]
        x = pad_sequence(x, batch_first=True)
        x = pack_padded_sequence(
            x, batch_lens, batch_first=True, enforce_sorted=False
        )

        _, (h_n, _) = self.cond_lstm(x)
        x = self.fc(h_n[0])
        x = F.softmax(x, dim=-1)
        return x

    def _condense_attn(self, x: Tensor, batch_lens) -> Tensor:
        c = torch.cumsum(torch.tensor([0] + batch_lens), 0)
        x = [x[i:j] for i, j in zip(c[:-1], c[1:])]
        x = pad_sequence(x, batch_first=True)
        x = pack_padded_sequence(
            x, batch_lens, batch_first=True, enforce_sorted=False
        )

        _, (h_n, _) = self.cond_lstm(x)
        return h_n[0]
