from types import SimpleNamespace

import pytest
import torch
from torch import nn

from instacart_rnn.training.trainer import (
    Trainer,
    evaluate,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
)

DEVICE = torch.device("cpu")


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(self, batch):
        pred = self.linear(batch["x"])
        final_logits = pred.squeeze(-1)

        return SimpleNamespace(
            logits=final_logits.unsqueeze(-1).expand(-1, 2).contiguous(),
            final_logits=final_logits,
            final_states=final_logits.unsqueeze(-1).expand(-1, 4).contiguous(),
        )


class ScriptedLoss:
    def __init__(self, values):
        self.values = list(values)
        self.calls = 0

    def __call__(self, output, batch):
        value = self.values[self.calls]
        self.calls += 1
        return torch.tensor(value)


def _batch(*, x, y, user_ids=None):
    row_count = x.size(0)

    if user_ids is None:
        user_ids = torch.arange(1, row_count + 1)

    return {
        "user_id": user_ids,
        "product_id": torch.ones(row_count, dtype=torch.long),
        "aisle_id": torch.ones(row_count, dtype=torch.long),
        "x": x,
        "y": y,
        "history_length": torch.ones(row_count, dtype=torch.long),
        "sequence_loss_length": torch.ones(row_count, dtype=torch.long),
    }


def _mse_loss(output, batch):
    return torch.nn.functional.mse_loss(output.final_logits, batch["y"])


def _trainer(*, checkpoint_path, epochs, early_stopping=None, val_loss_fn=None):
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    return Trainer(
        model=model,
        optimizer=optimizer,
        train_loss_fn=_mse_loss,
        val_loss_fn=val_loss_fn or _mse_loss,
        checkpoint_path=str(checkpoint_path),
        device=DEVICE,
        epochs=epochs,
        early_stopping=early_stopping,
    )


def test_train_one_epoch_updates_parameters_and_runs_backward():
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    before = model.linear.weight.detach().clone()
    batch = _batch(
        x=torch.ones(2, 1),
        y=torch.tensor([1.0, 1.0]),
    )

    train_one_epoch(
        model=model,
        dataloader=[batch],
        optimizer=optimizer,
        loss_fn=_mse_loss,
        device=DEVICE,
    )

    assert not torch.equal(model.linear.weight.detach(), before)
    assert model.linear.weight.grad is not None
    assert torch.count_nonzero(model.linear.weight.grad) > 0


def test_evaluate_returns_scalar_without_progress():
    model = TinyModel()
    model.linear.weight.data.fill_(1.0)
    batch = _batch(
        x=torch.ones(2, 1),
        y=torch.tensor([1.0, 1.0]),
    )

    loss = evaluate(
        model=model,
        dataloader=[batch],
        loss_fn=_mse_loss,
        device=DEVICE,
    )

    assert isinstance(loss, float)
    assert loss == pytest.approx(0.0)


def test_evaluate_does_not_populate_gradients():
    model = TinyModel()
    model.zero_grad(set_to_none=True)
    batch = _batch(
        x=torch.ones(2, 1),
        y=torch.tensor([0.0, 1.0]),
    )

    evaluate(
        model=model,
        dataloader=[batch],
        loss_fn=_mse_loss,
        device=DEVICE,
    )

    assert all(parameter.grad is None for parameter in model.parameters())
    assert not model.training


def test_save_and_load_checkpoint_round_trip(tmp_path):
    model = TinyModel()
    model.linear.weight.data.fill_(3.0)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    save_checkpoint(
        path=str(tmp_path),
        model=model,
        optimizer=optimizer,
        epoch=4,
        validation_loss=0.123,
    )

    loaded_model = TinyModel()

    loaded_optimizer = torch.optim.SGD(
        loaded_model.parameters(),
        lr=0.5,
    )

    checkpoint = load_checkpoint(
        path=str(tmp_path),
        model=loaded_model,
        optimizer=loaded_optimizer,
        device=DEVICE,
    )

    assert torch.equal(
        loaded_model.linear.weight,
        model.linear.weight,
    )

    assert loaded_optimizer.param_groups[0]["lr"] == pytest.approx(0.01)

    assert checkpoint["epoch"] == 4
    assert checkpoint["validation_loss"] == pytest.approx(0.123)


def test_fit_saves_best_checkpoint_and_does_not_overwrite_on_worse_loss(
    tmp_path,
):
    val_loss_fn = ScriptedLoss([0.5, 0.4, 0.9])
    trainer = _trainer(
        checkpoint_path=tmp_path,
        epochs=3,
        val_loss_fn=val_loss_fn,
    )
    batch = _batch(
        x=torch.ones(2, 1),
        y=torch.tensor([1.0, 1.0]),
    )

    trainer.fit(
        train_num_rows=2,
        val_num_rows=2,
        train_dataloader=[batch],
        val_dataloader=[batch],
    )

    checkpoint = load_checkpoint(
        path=str(tmp_path),
        model=TinyModel(),
        device=DEVICE,
    )

    assert val_loss_fn.calls == 3
    assert checkpoint["epoch"] == 1
    assert checkpoint["validation_loss"] == pytest.approx(0.4)


def test_fit_stops_when_early_stopping_patience_is_exceeded(tmp_path):
    val_loss_fn = ScriptedLoss([0.5, 0.8])
    trainer = _trainer(
        checkpoint_path=tmp_path,
        epochs=5,
        early_stopping=1,
        val_loss_fn=val_loss_fn,
    )
    batch = _batch(
        x=torch.ones(2, 1),
        y=torch.tensor([1.0, 1.0]),
    )

    trainer.fit(
        train_num_rows=2,
        val_num_rows=2,
        train_dataloader=[batch],
        val_dataloader=[batch],
    )

    checkpoint = load_checkpoint(
        path=str(tmp_path),
        model=TinyModel(),
        device=DEVICE,
    )

    assert val_loss_fn.calls == 2
    assert checkpoint["epoch"] == 0
    assert checkpoint["validation_loss"] == pytest.approx(0.5)
