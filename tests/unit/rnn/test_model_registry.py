import pytest
import torch
from torch import nn

from instacart_rnn.dataset import create_product_dataloader
from instacart_rnn.models.model_registry import get_model_spec
from instacart_rnn.training.losses import bce_train_loss, bce_validation_loss
from tests.unit.rnn.test_dataset import _write_training_dataset


def test_get_model_spec_returns_product_training_stack():
    spec = get_model_spec("product")

    assert spec.dataloader_factory is create_product_dataloader
    assert spec.train_loss_factory() is bce_train_loss
    assert spec.validation_loss_factory() is bce_validation_loss
    assert spec.inference.batch_tensor_names == (
        "user_id",
        "product_id",
        "aisle_id",
    )
    assert spec.inference.output_tensor_names == (
        "final_states",
        "final_logits",
    )


@pytest.mark.parametrize("model_name", ["unknown", "", "Product"])
def test_get_model_spec_raises_for_unsupported_model(model_name):
    with pytest.raises(ValueError, match="Supported models: product") as exc_info:
        get_model_spec(model_name)

    assert str(exc_info.value).startswith(f"Unsupported model {model_name!r}")


def test_product_spec_model_factory_builds_configured_product_model(mocker):
    constructed = mocker.patch(
        "instacart_rnn.models.model_registry.ProductModel",
    )

    get_model_spec("product").model_factory()

    constructed.assert_called_once_with(
        lstm_size=300,
        dilations=[1, 2, 4, 8, 16, 32],
        filter_widths=[2, 2, 2, 2, 2, 2],
        skip_channels=64,
        residual_channels=128,
    )


def test_product_spec_optimizer_factory_builds_adamw_with_requested_hparams():
    parameter = nn.Parameter(torch.zeros(1))

    optimizer = get_model_spec("product").optimizer_factory(
        [parameter],
        lr=1e-3,
        weight_decay=0.1,
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.1)


def test_product_spec_count_rows_counts_parquet_dataset_rows(tmp_path):
    dataset_path = tmp_path / "training"
    _write_training_dataset(dataset_path, [[1, 2], [3]])

    assert get_model_spec("product").count_rows(str(dataset_path)) == 3
