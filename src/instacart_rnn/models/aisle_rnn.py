from dataclasses import dataclass

import torch
from torch import nn

from instacart_rnn.models.lstm import LSTMLayer


@dataclass
class AisleRNNOutput:
    logits: torch.Tensor
    final_states: torch.Tensor
    final_logits: torch.Tensor


class AisleRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_size: int,
        hidden_size: int = 50,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.lstm = LSTMLayer(
            input_size=input_size,
            hidden_size=lstm_size,
        )

        combined_dim = lstm_size + input_size

        self.dense_1 = nn.Linear(
            combined_dim,
            hidden_size,
        )

        self.dense_2 = nn.Linear(
            hidden_size,
            1,
        )

        self.relu = nn.ReLU()

    def forward(
        self,
        x: torch.Tensor,
        history_length: torch.Tensor,
    ) -> AisleRNNOutput:
        # [B, T, lstm_size]
        lstm_output, _ = self.lstm(
            x,
            history_length,
        )

        # [B, T, combined_dim]
        h = torch.cat(
            [
                lstm_output,
                x,
            ],
            dim=-1,
        )

        # [B, T, hidden_size]
        hidden_states = self.relu(self.dense_1(h))

        # [B, T, 1]
        logits = self.dense_2(hidden_states)

        # [B, T]
        logits = logits.squeeze(-1)

        final_indices = history_length - 1

        batch_indices = torch.arange(
            x.size(0),
            device=x.device,
        )

        # [B, hidden_size]
        final_states = hidden_states[
            batch_indices,
            final_indices,
        ]

        # [B]
        final_logits = logits[
            batch_indices,
            final_indices,
        ]

        return AisleRNNOutput(
            logits=logits,
            final_states=final_states,
            final_logits=final_logits,
        )
