from types import SimpleNamespace

import torch
import torch.nn.functional as F

from instacart_rnn.training.losses import (
    bce_train_loss,
    bce_validation_loss,
    masked_sequence_bce_with_logits,
    masked_sequence_rmse,
)


def test_masked_sequence_bce_with_logits_returns_scalar():
    loss = masked_sequence_bce_with_logits(
        logits=torch.zeros(2, 3),
        targets=torch.zeros(2, 3),
        sequence_lengths=torch.tensor([2, 1]),
    )

    assert loss.ndim == 0


def test_masked_sequence_bce_with_logits_ignores_padded_timesteps():
    logits = torch.tensor(
        [
            [0.0, 100.0],
            [0.0, 100.0],
        ]
    )
    targets = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    loss = masked_sequence_bce_with_logits(
        logits,
        targets,
        sequence_lengths=torch.tensor([1, 1]),
    )
    expected = F.binary_cross_entropy_with_logits(
        torch.zeros(2),
        torch.ones(2),
    )

    assert torch.allclose(loss, expected)


def test_masked_sequence_bce_with_logits_averages_over_valid_positions():
    logits = torch.tensor(
        [
            [0.0, 1.0, 99.0],
            [0.5, 99.0, 99.0],
        ]
    )
    targets = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )

    loss = masked_sequence_bce_with_logits(
        logits,
        targets,
        sequence_lengths=torch.tensor([2, 1]),
    )
    expected = F.binary_cross_entropy_with_logits(
        torch.tensor([0.0, 1.0, 0.5]),
        torch.tensor([1.0, 0.0, 1.0]),
    )

    assert torch.allclose(loss, expected)


def test_masked_sequence_bce_with_logits_returns_zero_when_no_valid_positions():
    loss = masked_sequence_bce_with_logits(
        logits=torch.ones(2, 3) * 100,
        targets=torch.zeros(2, 3),
        sequence_lengths=torch.zeros(2, dtype=torch.long),
    )

    assert torch.allclose(loss, torch.tensor(0.0))


def test_masked_sequence_bce_with_logits_has_zero_grad_on_padded_timesteps():
    logits = torch.tensor(
        [
            [0.0, 1.0],
            [0.5, 2.0],
        ],
        requires_grad=True,
    )
    targets = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )

    masked_sequence_bce_with_logits(
        logits,
        targets,
        sequence_lengths=torch.tensor([1, 1]),
    ).backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 0]) == 2
    assert torch.count_nonzero(logits.grad[:, 1]) == 0


def test_masked_sequence_rmse_returns_scalar():
    loss = masked_sequence_rmse(
        y=torch.zeros(2, 3),
        y_hat=torch.zeros(2, 3),
        sequence_lengths=torch.tensor([2, 1]),
    )

    assert loss.ndim == 0


def test_masked_sequence_rmse_ignores_padded_timesteps():
    loss = masked_sequence_rmse(
        y=torch.tensor(
            [
                [1.0, 100.0],
                [2.0, 100.0],
            ]
        ),
        y_hat=torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
            ]
        ),
        sequence_lengths=torch.tensor([1, 1]),
    )

    assert torch.allclose(loss, torch.tensor(0.0))


def test_masked_sequence_rmse_matches_root_mean_of_valid_squared_errors():
    loss = masked_sequence_rmse(
        y=torch.tensor(
            [
                [1.0, 3.0, 9.0],
                [2.0, 9.0, 9.0],
            ]
        ),
        y_hat=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        sequence_lengths=torch.tensor([2, 1]),
    )
    expected = torch.sqrt(torch.tensor(13.0 / 3.0))

    assert torch.allclose(loss, expected)


def test_masked_sequence_rmse_has_zero_grad_on_padded_timesteps():
    y_hat = torch.tensor(
        [
            [0.0, 5.0],
            [0.0, 5.0],
        ],
        requires_grad=True,
    )

    masked_sequence_rmse(
        y=torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        ),
        y_hat=y_hat,
        sequence_lengths=torch.tensor([1, 1]),
    ).backward()

    assert y_hat.grad is not None
    assert torch.count_nonzero(y_hat.grad[:, 0]) == 2
    assert torch.count_nonzero(y_hat.grad[:, 1]) == 0


def test_bce_train_loss_uses_masked_sequence_logits():
    output = SimpleNamespace(
        logits=torch.tensor(
            [
                [0.0, 1.0, 99.0],
                [0.5, 99.0, 99.0],
            ]
        ),
        final_logits=torch.ones(2) * 50,
    )
    batch = {
        "next_is_ordered": torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        "label": torch.zeros(2),
        "sequence_loss_length": torch.tensor([2, 1]),
    }

    loss = bce_train_loss(output, batch)
    expected = masked_sequence_bce_with_logits(
        logits=output.logits,
        targets=batch["next_is_ordered"],
        sequence_lengths=batch["sequence_loss_length"],
    )

    assert torch.allclose(loss, expected)

    padded_output = SimpleNamespace(
        logits=output.logits.clone(),
        final_logits=output.final_logits,
    )
    padded_output.logits[:, 2] = -99.0
    padded_output.logits[1, 1] = -99.0

    assert torch.allclose(
        bce_train_loss(padded_output, batch),
        expected,
    )


def test_bce_validation_loss_uses_final_logits_and_label():
    output = SimpleNamespace(
        logits=torch.ones(2, 3) * 99,
        final_logits=torch.tensor([0.0, 1.5]),
    )
    batch = {
        "next_is_ordered": torch.ones(2, 3),
        "label": torch.tensor([1.0, 0.0]),
        "sequence_loss_length": torch.tensor([1, 1]),
    }

    loss = bce_validation_loss(output, batch)
    expected = F.binary_cross_entropy_with_logits(
        output.final_logits,
        batch["label"].float(),
    )

    assert torch.allclose(loss, expected)
