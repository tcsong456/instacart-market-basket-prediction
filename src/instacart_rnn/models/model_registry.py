from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import pyarrow.dataset as pads
import torch
from torch import nn
from torch.optim import Optimizer

from instacart_rnn.dataset import (
    create_aisle_dataloader,
    create_product_dataloader,
    create_reorder_size_dataloader,
)
from instacart_rnn.models.aisle_model import AisleModel
from instacart_rnn.models.product_model import ProductModel
from instacart_rnn.models.reorder_size_gmm_model import ReorderSizeGmmModel
from instacart_rnn.models.representation import (
    binary_output_transform,
    gmm_output_transform,
)
from instacart_rnn.training.losses import (
    bce_train_loss,
    bce_validation_loss,
    gmm_train_loss,
    gmm_validation_loss,
)

TensorBatch = dict[str, torch.Tensor]
LossFn = Callable[[Any, TensorBatch], torch.Tensor]


InferenceTransform = Callable[
    [Any, TensorBatch],
    dict[str, torch.Tensor],
]

InferenceTransformFactory = Callable[..., InferenceTransform]


@dataclass(frozen=True)
class InferenceSpec:
    batch_tensor_names: tuple[str, ...]
    output_tensor_names: tuple[str, ...]
    output_transform_factory: InferenceTransformFactory | None = None


@dataclass(frozen=True)
class ModelSpec:
    model_factory: Callable[[int], nn.Module]

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


def _build_product_model(lstm_size: int) -> nn.Module:
    return ProductModel(
        lstm_size=lstm_size,
        dilations=[2**i for i in range(6)],
        filter_widths=[2] * 6,
        skip_channels=64,
        residual_channels=128,
    )


def _build_aisle_model(lstm_size: int) -> nn.Module:
    return AisleModel(
        lstm_size=lstm_size,
    )


def _build_reorder_size_gmm(lstm_size: int) -> nn.Module:
    return ReorderSizeGmmModel(lstm_size=lstm_size)


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
            output_transform_factory=binary_output_transform,
        ),
    ),
    "aisle": ModelSpec(
        model_factory=_build_aisle_model,
        train_loss_factory=lambda: bce_train_loss,
        validation_loss_factory=lambda: bce_validation_loss,
        dataloader_factory=create_aisle_dataloader,
        count_rows=_count_parquet_rows,
        optimizer_factory=_adamw,
        inference=InferenceSpec(
            batch_tensor_names=(
                "user_id",
                "aisle_id",
            ),
            output_tensor_names=(
                "final_states",
                "final_logits",
            ),
            output_transform_factory=binary_output_transform,
        ),
    ),
    "reorder_size_gmm": ModelSpec(
        model_factory=_build_reorder_size_gmm,
        train_loss_factory=lambda: gmm_train_loss,
        validation_loss_factory=lambda: gmm_validation_loss,
        dataloader_factory=create_reorder_size_dataloader,
        count_rows=_count_parquet_rows,
        optimizer_factory=_adamw,
        inference=InferenceSpec(
            batch_tensor_names=("user_id",),
            output_tensor_names=("final_states",),
            output_transform_factory=gmm_output_transform,
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
