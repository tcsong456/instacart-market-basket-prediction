import torch
import torch.nn as nn

from instacart_rnn.models.reorder_size_input_encoder import ReorderSizeInputEncoder
from instacart_rnn.models.reorder_size_rnn import ReorderSizeRNN, ReorderSizeRNNOutput


class ReorderSizeRNNModel(nn.Module):
    """Combine input encoding and reorder size sequence prediction."""

    def __init__(
        self,
        *,
        lstm_size: int,
    ) -> None:
        super().__init__()

        self.encoder = ReorderSizeInputEncoder()

        self.rnn = ReorderSizeRNN(
            input_size=self.encoder.output_dim,
            lstm_size=lstm_size,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> ReorderSizeRNNOutput:
        encoded = self.encoder(batch)

        return self.rnn(
            encoded,
            batch["history_length"],
        )
