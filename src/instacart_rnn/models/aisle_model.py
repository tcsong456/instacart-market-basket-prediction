import torch
import torch.nn as nn

from instacart_rnn.models.aisle_input_encoder import AisleInputEncoder
from instacart_rnn.models.aisle_rnn import AisleRNN, AisleRNNOutput


class AisleModel(nn.Module):
    """Combine input encoding and aisle sequence prediction."""

    def __init__(
        self,
        *,
        lstm_size: int,
    ) -> None:
        super().__init__()

        self.encoder = AisleInputEncoder(
            lstm_size=lstm_size,
        )

        self.rnn = AisleRNN(
            input_size=self.encoder.output_dim,
            lstm_size=lstm_size,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> AisleRNNOutput:
        encoded = self.encoder(batch)

        return self.rnn(
            encoded,
            batch["history_length"],
        )
