import random

import numpy as np
import pytest
import torch
from torch import nn

from instacart_rnn.models.model_registry import InferenceSpec, ModelSpec
from instacart_rnn.training.runner import (
    InferenceRunConfig,
    TrainingRunConfig,
    resolve_device,
    run_inference,
    run_training,
    set_random_seed,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)


def _sgd(parameters, *, lr, weight_decay):
    return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)


def _tiny_spec(
    *,
    dataloader_factory,
    count_rows,
    train_loss_fn=None,
    validation_loss_fn=None,
):
    return ModelSpec(
        model_factory=TinyModel,
        train_loss_factory=lambda: train_loss_fn,
        validation_loss_factory=lambda: validation_loss_fn,
        dataloader_factory=dataloader_factory,
        count_rows=count_rows,
        optimizer_factory=_sgd,
        inference=InferenceSpec(
            batch_tensor_names=("user_id", "product_id", "aisle_id"),
            output_tensor_names=("final_states", "final_logits"),
        ),
    )


def _training_config(**overrides):
    values = {
        "model_name": "tiny",
        "train_path": "train.parquet",
        "validation_path": "val.parquet",
        "checkpoint_path": "ckpt",
        "epochs": 2,
        "batch_size": 8,
        "read_batch_size": 16,
        "learning_rate": 0.01,
        "weight_decay": 0.1,
        "num_workers": 2,
        "seed": 7,
        "amp": True,
        "grad_clip_norm": 1.5,
        "early_stopping": 3,
        "warm_start": True,
        "pin_memory": False,
    }
    values.update(overrides)
    return TrainingRunConfig(**values)


def _inference_config(**overrides):
    values = {
        "model_name": "tiny",
        "eval_path": "eval.parquet",
        "checkpoint_path": "ckpt",
        "output_path": "output",
        "batch_size": 8,
        "read_batch_size": 16,
        "rows_per_write": 1000,
        "num_workers": 2,
        "pin_memory": False,
        "amp": True,
    }
    values.update(overrides)
    return InferenceRunConfig(**values)


@pytest.mark.parametrize(
    ("cuda_available", "expected"),
    [
        (False, torch.device("cpu")),
        (True, torch.device("cuda")),
    ],
)
def test_resolve_device_uses_cuda_when_available_otherwise_cpu(
    mocker,
    cuda_available,
    expected,
):
    mocker.patch(
        "instacart_rnn.training.runner.torch.cuda.is_available",
        return_value=cuda_available,
    )

    assert resolve_device() == expected


def test_set_random_seed_makes_python_numpy_and_torch_draws_reproducible():
    set_random_seed(123)
    python_draw = random.random()
    numpy_draw = float(np.random.rand())
    torch_draw = torch.rand(1).item()

    set_random_seed(123)

    assert random.random() == python_draw
    assert float(np.random.rand()) == pytest.approx(numpy_draw)
    assert torch.rand(1).item() == pytest.approx(torch_draw)


def test_run_training_raises_for_unsupported_model():
    with pytest.raises(ValueError, match="Unsupported model 'unknown'"):
        run_training(_training_config(model_name="unknown"))


def test_run_training_fits_registered_model_with_train_and_validation_loaders(
    mocker,
):
    train_loader = object()
    val_loader = object()
    train_loss_fn = object()
    validation_loss_fn = object()
    dataloader_calls = []

    def dataloader_factory(**kwargs):
        dataloader_calls.append(kwargs)
        if kwargs["path"] == "train.parquet":
            return train_loader
        return val_loader

    def count_rows(path):
        return {"train.parquet": 10, "val.parquet": 4}[path]

    get_spec = mocker.patch(
        "instacart_rnn.training.runner.get_model_spec",
        return_value=_tiny_spec(
            dataloader_factory=dataloader_factory,
            count_rows=count_rows,
            train_loss_fn=train_loss_fn,
            validation_loss_fn=validation_loss_fn,
        ),
    )
    mocker.patch(
        "instacart_rnn.training.runner.resolve_device",
        return_value=torch.device("cpu"),
    )
    trainer_cls = mocker.patch("instacart_rnn.training.runner.Trainer", autospec=True)

    run_training(_training_config())

    get_spec.assert_called_once_with("tiny")
    assert dataloader_calls == [
        {
            "path": "train.parquet",
            "batch_size": 8,
            "read_batch_size": 16,
            "num_workers": 2,
            "pin_memory": False,
            "drop_last": True,
        },
        {
            "path": "val.parquet",
            "batch_size": 8,
            "read_batch_size": 16,
            "num_workers": 2,
            "pin_memory": False,
            "drop_last": False,
        },
    ]

    trainer_cls.assert_called_once()
    kwargs = trainer_cls.call_args.kwargs
    assert isinstance(kwargs["model"], TinyModel)
    assert kwargs["train_loss_fn"] is train_loss_fn
    assert kwargs["val_loss_fn"] is validation_loss_fn
    assert kwargs["device"] == torch.device("cpu")
    assert kwargs["epochs"] == 2
    assert kwargs["checkpoint_path"] == "ckpt"
    assert kwargs["grad_clip_norm"] == pytest.approx(1.5)
    assert kwargs["amp"] is True
    assert kwargs["early_stopping"] == 3
    assert kwargs["optimizer"].param_groups[0]["lr"] == pytest.approx(0.01)
    assert kwargs["optimizer"].param_groups[0]["weight_decay"] == pytest.approx(0.1)

    trainer_cls.return_value.fit.assert_called_once_with(
        train_num_rows=10,
        val_num_rows=4,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        warm_start=True,
    )


def test_run_inference_loads_checkpoint_and_writes_representations(mocker):
    eval_loader = object()
    dataloader_calls = []

    def dataloader_factory(**kwargs):
        dataloader_calls.append(kwargs)
        return eval_loader

    get_spec = mocker.patch(
        "instacart_rnn.training.runner.get_model_spec",
        return_value=_tiny_spec(
            dataloader_factory=dataloader_factory,
            count_rows=lambda path: 5,
        ),
    )
    mocker.patch(
        "instacart_rnn.training.runner.resolve_device",
        return_value=torch.device("cpu"),
    )
    load_ckpt = mocker.patch("instacart_rnn.training.runner.load_checkpoint")
    write_parquet = mocker.patch(
        "instacart_rnn.training.runner.write_inference_parquet",
    )

    run_inference(_inference_config())

    get_spec.assert_called_once_with("tiny")
    assert dataloader_calls == [
        {
            "path": "eval.parquet",
            "batch_size": 8,
            "read_batch_size": 16,
            "num_workers": 2,
            "pin_memory": False,
            "drop_last": False,
        },
    ]

    load_ckpt.assert_called_once()
    load_kwargs = load_ckpt.call_args.kwargs
    assert load_kwargs["path"] == "ckpt"
    assert isinstance(load_kwargs["model"], TinyModel)
    assert load_kwargs["device"] == torch.device("cpu")

    write_parquet.assert_called_once()
    write_kwargs = write_parquet.call_args.kwargs
    assert write_kwargs["dataloader"] is eval_loader
    assert write_kwargs["device"] == torch.device("cpu")
    assert write_kwargs["output_path"] == "output"
    assert write_kwargs["rows_per_write"] == 1000
    assert write_kwargs["amp"] is True
    assert write_kwargs["batch_tensor_names"] == (
        "user_id",
        "product_id",
        "aisle_id",
    )
    assert write_kwargs["output_tensor_names"] == (
        "final_states",
        "final_logits",
    )
    assert isinstance(write_kwargs["model"], TinyModel)
