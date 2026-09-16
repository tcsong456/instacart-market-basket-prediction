from dataclasses import dataclass

import torch
from torch import nn

from instacart_rnn.models.lstm import LSTMLayer


@dataclass
class ReorderSizeGMMOutput:
    hidden_states: torch.Tensor
    means: torch.Tensor
    log_variances: torch.Tensor
    mixing_logits: torch.Tensor
    final_means: torch.Tensor
    final_log_variances: torch.Tensor
    final_mixing_logits: torch.Tensor


class ReorderSizeGMM(nn.Module):
    def __init__(
        self,
        input_size: int,
        lstm_size: int,
        n_components: int = 3,
    ):
        super().__init__()

        self.n_components = n_components

        self.lstm = LSTMLayer(
            input_size=input_size,
            hidden_size=lstm_size,
        )

        self.dense_1 = nn.Linear(lstm_size, 50)
        self.relu = nn.ReLU()

        self.dense_2 = nn.Linear(
            50,
            n_components * 3,
        )

    def forward(self, x: torch.Tensor, history_length: torch.Tensor):
        h, _ = self.lstm(
            x,
            history_length,
        )

        hidden_states = self.relu(self.dense_1(h))

        params = self.dense_2(hidden_states)

        means, log_variances, mixing_logits = torch.chunk(
            params,
            3,
            dim=-1,
        )

        final_indices = history_length - 1
        batch_indices = torch.arange(
            x.size(0),
            device=x.device,
        )

        return ReorderSizeGMMOutput(
            hidden_states=hidden_states,
            means=means,
            log_variances=log_variances,
            mixing_logits=mixing_logits,
            final_means=means[batch_indices, final_indices],
            final_log_variances=log_variances[batch_indices, final_indices],
            final_mixing_logits=mixing_logits[batch_indices, final_indices],
        )
