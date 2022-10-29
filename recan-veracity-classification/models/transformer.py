import copy
from torch import nn, Tensor
from typing import Optional


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.8,
        mha_dropout: float = 0.5,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=mha_dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = nn.GELU(approximate="tanh")

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.norm(
            query
            + self._sa_block(
                query,
                key,
                value,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )
        )
        x = self.norm(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
    ) -> Tensor:
        out = self.self_attn(
            query,
            key,
            value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout(out)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Optional[nn.LayerNorm] = None,
    ):
        super(TransformerEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

        if self.norm is not None:
            query = self.norm(query)

        return query


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
