import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from typing import Tuple

from .transformer import TransformerEncoder, TransformerEncoderLayer


class CoAttentionModule(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int = 120,
        dim_feedforward: int = 2048,
        nhead: int = 6,
        num_layers: int = 1,
        mha_dropout: float = 0.5,
        dropout: float = 0.5,
    ):
        super(CoAttentionModule, self).__init__()

        assert (dim_hidden * 2) % nhead == 0

        self.bilstm_claim = nn.LSTM(
            input_size=dim_input,
            hidden_size=dim_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.bilstm_evidence = nn.LSTM(
            input_size=dim_input,
            hidden_size=dim_hidden,
            batch_first=True,
            bidirectional=True,
        )

        dim_encoder_input = dim_hidden * 2
        encoder_layer = TransformerEncoderLayer(
            dim_encoder_input, nhead, dim_feedforward, dropout, mha_dropout
        )
        self.self_attn_claim = TransformerEncoder(
            encoder_layer, num_layers, nn.LayerNorm(dim_encoder_input)
        )
        self.self_attn_evidence = TransformerEncoder(
            encoder_layer, num_layers, nn.LayerNorm(dim_encoder_input)
        )

    def forward(self, x, lengths, masks):
        seq_rep_claim, seq_rep_evidence = self._lstm(x, lengths)
        c, e = self._attn(seq_rep_claim, seq_rep_evidence, masks)
        out = self._condese(c, e)
        return out

    def _lstm(self, x: Tensor, lengths: Tensor) -> Tuple[Tensor, Tensor]:
        # x shape (pairs, 2, seq_len, embed_dim)
        # seq_rep_x shape (pairs, seq_len, 2 * hidden_dim)
        raw_claims = x[:, 0, :, :]
        raw_claims = pack_padded_sequence(
            raw_claims, lengths[:, 0], enforce_sorted=False, batch_first=True
        )
        seq_rep_claim = self.bilstm_claim(raw_claims)[0]
        seq_rep_claim = pad_packed_sequence(
            seq_rep_claim, batch_first=True, total_length=30
        )[0]

        raw_evidence = x[:, 1, :, :]
        raw_evidence = pack_padded_sequence(
            raw_evidence,
            lengths[:, 1],
            enforce_sorted=False,
            batch_first=True,
        )
        seq_rep_evidence = self.bilstm_evidence(raw_evidence)[0]
        seq_rep_evidence = pad_packed_sequence(
            seq_rep_evidence, batch_first=True, total_length=30
        )[0]

        return seq_rep_claim, seq_rep_evidence

    def _attn(
        self, seq_rep_claim: Tensor, seq_rep_evidence: Tensor, masks: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # input shape (pairs, seq_len, 2 * hidden_dim)
        # output shape (pairs, seq_len, 2 * hidden_dim)
        c = self.self_attn_claim(
            seq_rep_evidence,
            seq_rep_claim,
            seq_rep_claim,
            key_padding_mask=masks[:, 0, :],
        )
        e = self.self_attn_evidence(
            c,
            seq_rep_evidence,
            seq_rep_evidence,
            key_padding_mask=masks[:, 1, :],
        )

        return c, e

    def _condese(self, c: Tensor, e: Tensor) -> Tensor:
        c_pooled = c.permute(0, 2, 1).contiguous()
        c_pooled = F.adaptive_max_pool1d(c_pooled, 1).squeeze(-1)

        e_pooled = e.permute(0, 2, 1).contiguous()
        e_pooled = F.adaptive_max_pool1d(e_pooled, 1).squeeze(-1)

        out = torch.cat(
            [
                e_pooled,
                torch.abs(e_pooled - c_pooled),
                e_pooled * c_pooled,
                c_pooled,
            ],
            dim=1,
        )

        return out
