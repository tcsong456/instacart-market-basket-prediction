from dataclasses import dataclass

import torch
from torch import nn

from instacart_rnn.models.lstm import LSTMLayer


@dataclass
class ReorderSizeRNNOutput:
    predictions: torch.Tensor
    final_states: torch.Tensor
    final_predictions: torch.Tensor


class ReorderSizeRNN(nn.Module):
    def __init__(self, input_size: int, lstm_size: int) -> None:
        super().__init__()

        self.lstm = LSTMLayer(
            input_size=input_size,
            hidden_size=lstm_size,
        )

        self.dense_1 = nn.Linear(lstm_size, 50)
        self.relu = nn.ReLU()
        self.dense_2 = nn.Linear(50, 1)

    def forward(
        self,
        x: torch.Tensor,
        history_length: torch.Tensor,
    ) -> ReorderSizeRNNOutput:
        h, _ = self.lstm(
            x,
            history_length,
        )

        hidden_states = self.relu(self.dense_1(h))

        predictions = self.dense_2(hidden_states).squeeze(-1)

        batch_indices = torch.arange(
            x.size(0),
            device=x.device,
        )

        final_temporal_indices = history_length - 1

        final_states = hidden_states[
            batch_indices,
            final_temporal_indices,
        ]

        final_predictions = predictions[
            batch_indices,
            final_temporal_indices,
        ]

        return ReorderSizeRNNOutput(
            predictions=predictions,
            final_states=final_states,
            final_predictions=final_predictions,
        )
