from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pyarrow.dataset as pads
import torch
from torch import nn
from torch.optim import Optimizer

from instacart_rnn.dataset import (
    create_product_dataloader,
)
from instacart_rnn.models.product_model import ProductModel
from instacart_rnn.training.losses import (
    bce_train_loss,
    bce_validation_loss,
)

TensorBatch = dict[str, torch.Tensor]
LossFn = Callable[[Any, TensorBatch], torch.Tensor]


@dataclass(frozen=True)
class InferenceSpec:
    batch_tensor_names: tuple[str, ...]
    output_tensor_names: tuple[str, ...]


@dataclass(frozen=True)
class ModelSpec:
    model_factory: Callable[[], nn.Module]

    train_loss_factory: Callable[[], LossFn]
    validation_loss_factory: Callable[[], LossFn]

    dataloader_factory: Callable[..., Iterable[TensorBatch]]
    count_rows: Callable[[str], int]

    optimizer_factory: Callable[..., Optimizer]

    inference: InferenceSpec


def _count_parquet_rows(
    path: str,
) -> int:
    return pads.dataset(
        path,
        format="parquet",
    ).count_rows()


def _build_product_model() -> nn.Module:
    return ProductModel(
        lstm_size=300,
        dilations=[2**i for i in range(6)],
        filter_widths=[2] * 6,
        skip_channels=64,
        residual_channels=128,
    )


def _adamw(
    parameters,
    *,
    lr: float,
    weight_decay: float,
) -> Optimizer:
    return torch.optim.AdamW(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
    )


MODEL_REGISTRY = {
    "product": ModelSpec(
        model_factory=_build_product_model,
        train_loss_factory=lambda: bce_train_loss,
        validation_loss_factory=lambda: bce_validation_loss,
        dataloader_factory=create_product_dataloader,
        count_rows=_count_parquet_rows,
        optimizer_factory=_adamw,
        inference=InferenceSpec(
            batch_tensor_names=(
                "user_id",
                "product_id",
                "aisle_id",
            ),
            output_tensor_names=(
                "final_states",
                "final_logits",
            ),
        ),
    ),
}


def get_model_spec(
    model_name: str,
) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(MODEL_REGISTRY))

        raise ValueError(
            f"Unsupported model {model_name!r}. Supported models: {supported}"
        ) from exc
