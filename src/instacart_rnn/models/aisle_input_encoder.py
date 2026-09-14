import torch
from torch import nn

from instacart_rnn.models.encoders import (
    ClampOneHotEncoding,
    NormalizeTensor,
    OneHotFeature,
    ReturnEmbedding,
)


class AisleInputEncoder(nn.Module):
    def __init__(
        self,
        lstm_size: int,
        *,
        aisle_vocab_size: int = 250,
        aisle_embedding_dim: int = 50,
        department_vocab_size: int = 50,
        department_embedding_dim: int = 10,
        user_vocab_size: int = 207000,
    ) -> None:
        super().__init__()

        self.aisle_encoders = nn.ModuleDict(
            {
                "aisle_id": ReturnEmbedding(aisle_vocab_size, aisle_embedding_dim),
                "department_id": ReturnEmbedding(
                    department_vocab_size,
                    department_embedding_dim,
                ),
            }
        )

        self.user_encoder = nn.ModuleDict(
            {"user_id": ReturnEmbedding(user_vocab_size, lstm_size)}
        )

        self.one_hot_history_encoders = nn.ModuleDict(
            {
                "position_in_order_history": ClampOneHotEncoding(20),
                "history_order_size": ClampOneHotEncoding(60),
                "is_ordered_history": OneHotFeature(2),
                "order_dow_history": OneHotFeature(8),
                "order_hour_history": OneHotFeature(25),
                "days_since_prior_order_history": OneHotFeature(31),
                "order_number_history": OneHotFeature(101),
                "num_products_from_aisle_history": ClampOneHotEncoding(50),
            }
        )

        self.normalized_history_encoders = nn.ModuleDict(
            {
                "position_in_order_history": NormalizeTensor(20),
                "history_order_size": NormalizeTensor(60),
                "order_dow_history": NormalizeTensor(8),
                "order_hour_history": NormalizeTensor(25),
                "days_since_prior_order_history": NormalizeTensor(31),
                "order_number_history": NormalizeTensor(100),
                "num_products_from_aisle_history": NormalizeTensor(50),
            }
        )

    @property
    def output_dim(self) -> int:
        aisle_dim = sum(encoder.output_dim for encoder in self.aisle_encoders.values())

        user_dim = sum(encoder.output_dim for encoder in self.user_encoder.values())

        one_hot_history_dim = sum(
            encoder.output_dim for encoder in self.one_hot_history_encoders.values()
        )

        normalized_history_dim = sum(
            encoder.output_dim for encoder in self.normalized_history_encoders.values()
        )

        return aisle_dim + user_dim + one_hot_history_dim + normalized_history_dim

    @staticmethod
    def _expand_static_feature(
        feature: torch.Tensor,
        sequence_length: int,
    ) -> torch.Tensor:
        return feature.unsqueeze(1).expand(
            -1,
            sequence_length,
            -1,
        )

    def forward(self, batch: dict[str, torch.Tensor]):
        sequence_length = batch["is_ordered_history"].shape[1]

        aisle_features = [
            self._expand_static_feature(encoder(batch[name]), sequence_length)
            for name, encoder in self.aisle_encoders.items()
        ]

        user_features = [
            self._expand_static_feature(encoder(batch[name]), sequence_length)
            for name, encoder in self.user_encoder.items()
        ]

        one_hot_history_features = [
            encoder(batch[name])
            for name, encoder in self.one_hot_history_encoders.items()
        ]

        normalized_history_features = [
            encoder(batch[name])
            for name, encoder in self.normalized_history_encoders.items()
        ]
        features = (
            aisle_features
            + user_features
            + one_hot_history_features
            + normalized_history_features
        )

        x = torch.cat(features, dim=2)

        return x
