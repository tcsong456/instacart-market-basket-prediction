import torch
import torch.nn as nn

from instacart_rnn.models.product_input_encoder import ProductInputEncoder
from instacart_rnn.models.product_rnn import ProductRNN, ProductRNNOutput


class ProductModel(nn.Module):
    """Combine input encoding and product sequence prediction."""

    def __init__(
        self,
        *,
        lstm_size: int,
        dilations: list[int],
        filter_widths: list[int],
        skip_channels: int,
        residual_channels: int,
    ) -> None:
        super().__init__()

        self.encoder = ProductInputEncoder(
            lstm_size=lstm_size,
        )

        self.rnn = ProductRNN(
            input_size=self.encoder.output_dim,
            lstm_size=lstm_size,
            dilations=dilations,
            filter_widths=filter_widths,
            skip_channels=skip_channels,
            residual_channels=residual_channels,
        )

    def forward(
        self,
        batch: dict[str, torch.Tensor],
    ) -> ProductRNNOutput:
        encoded = self.encoder(batch)

        return self.rnn(
            encoded,
            batch["history_length"],
        )
