import torch
import torch.nn.functional as F

from instacart_rnn.training.losses import (
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
