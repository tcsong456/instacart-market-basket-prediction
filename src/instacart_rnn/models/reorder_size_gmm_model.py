import torch
import torch.nn as nn

from instacart_rnn.models.reorder_size_gmm import ReorderSizeGMM, ReorderSizeGMMOutput
from instacart_rnn.models.reorder_size_input_encoder import ReorderSizeInputEncoder


class ReorderSizeGmmModel(nn.Module):
    def __init__(
        self,
        *,
        lstm_size: int,
    ) -> None:
        super().__init__()

        self.encoder = ReorderSizeInputEncoder()

        self.rnn = ReorderSizeGMM(
            input_size=self.encoder.output_dim,
            lstm_size=lstm_size,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> ReorderSizeGMMOutput:
        encoded = self.encoder(batch)

        return self.rnn(
            encoded,
            batch["history_length"],
        )
