import torch
from torch import nn

from instacart_rnn.models.encoders import (
    ClampOneHotEncoding,
    NormalizeTensor,
    OneHotFeature,
)


class ReorderSizeInputEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.one_hot_history_encoders = nn.ModuleDict(
            {
                "history_order_size": ClampOneHotEncoding(60),
                "order_dow_history": OneHotFeature(8),
                "order_hour_history": OneHotFeature(25),
                "days_since_prior_order_history": OneHotFeature(31),
                "order_number_history": OneHotFeature(101),
                "reorder_size_history": ClampOneHotEncoding(50),
            }
        )

        self.normalized_history_encoders = nn.ModuleDict(
            {
                "history_order_size": NormalizeTensor(60),
                "order_dow_history": NormalizeTensor(8),
                "order_hour_history": NormalizeTensor(25),
                "days_since_prior_order_history": NormalizeTensor(31),
                "order_number_history": NormalizeTensor(100),
                "reorder_size_history": NormalizeTensor(50),
            }
        )

    @property
    def output_dim(self) -> int:
        one_hot_history_dim = sum(
            encoder.output_dim for encoder in self.one_hot_history_encoders.values()
        )

        normalized_history_dim = sum(
            encoder.output_dim for encoder in self.normalized_history_encoders.values()
        )

        return one_hot_history_dim + normalized_history_dim

    def forward(self, batch: dict[str, torch.Tensor]):
        one_hot_history_features = [
            encoder(batch[name])
            for name, encoder in self.one_hot_history_encoders.items()
        ]

        normalized_history_features = [
            encoder(batch[name])
            for name, encoder in self.normalized_history_encoders.items()
        ]
        features = one_hot_history_features + normalized_history_features

        x = torch.cat(features, dim=2)

        return x
