from types import SimpleNamespace

import torch
import torch.nn.functional as F

from instacart_rnn.training.losses import (
    bce_train_loss,
    bce_validation_loss,
    gaussian_mixture_nll,
    gmm_nll,
    gmm_train_loss,
    gmm_validation_loss,
    masked_sequence_bce_with_logits,
    masked_sequence_gmm_nll,
    masked_sequence_rmse,
    rmse_train_loss,
    rmse_validation_loss,
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


def test_rmse_train_loss_uses_next_reorder_size_and_sequence_loss_length():
    output = SimpleNamespace(
        predictions=torch.tensor(
            [
                [0.0, 1.0, 99.0],
                [2.0, 99.0, 99.0],
            ]
        ),
        final_predictions=torch.ones(2) * 50,
    )
    batch = {
        "next_reorder_size": torch.tensor(
            [
                [0.0, 1.0, 99.0],
                [2.0, 99.0, 99.0],
            ]
        ),
        "label": torch.zeros(2),
        "sequence_loss_length": torch.tensor([2, 1]),
    }

    loss = rmse_train_loss(output, batch)
    expected = masked_sequence_rmse(
        y=batch["next_reorder_size"],
        y_hat=output.predictions,
        sequence_lengths=batch["sequence_loss_length"],
    )

    assert torch.allclose(loss, expected)


def test_rmse_validation_loss_uses_final_predictions_and_label():
    output = SimpleNamespace(
        predictions=torch.ones(2, 3) * 99,
        final_predictions=torch.tensor([1.0, 3.0]),
    )
    batch = {
        "next_reorder_size": torch.ones(2, 3),
        "label": torch.tensor([1.0, 0.0]),
        "sequence_loss_length": torch.tensor([1, 1]),
    }

    loss = rmse_validation_loss(output, batch)
    expected = torch.sqrt(
        F.mse_loss(
            output.final_predictions,
            batch["label"].float(),
        )
    )

    assert torch.allclose(loss, expected)


def test_masked_sequence_gmm_nll_ignores_padded_timesteps():
    means = torch.zeros(2, 2, 1)
    log_variances = torch.zeros(2, 2, 1)
    mixing_logits = torch.zeros(2, 2, 1)
    targets = torch.tensor(
        [
            [0.0, 100.0],
            [0.0, 100.0],
        ]
    )

    loss = masked_sequence_gmm_nll(
        means=means,
        log_variances=log_variances,
        mixing_logits=mixing_logits,
        targets=targets,
        sequence_lengths=torch.tensor([1, 1]),
    )
    expected = gaussian_mixture_nll(
        means=means[:, :1],
        log_variances=log_variances[:, :1],
        mixing_logits=mixing_logits[:, :1],
        targets=targets[:, :1].unsqueeze(-1),
    ).mean()

    assert torch.allclose(loss, expected)


def test_gmm_train_loss_uses_next_reorder_size_and_sequence_loss_length():
    output = SimpleNamespace(
        means=torch.zeros(2, 3, 1),
        log_variances=torch.zeros(2, 3, 1),
        mixing_logits=torch.zeros(2, 3, 1),
        final_means=torch.ones(2, 1) * 50,
        final_log_variances=torch.zeros(2, 1),
        final_mixing_logits=torch.zeros(2, 1),
    )
    batch = {
        "next_reorder_size": torch.tensor(
            [
                [0.0, 1.0, 99.0],
                [2.0, 99.0, 99.0],
            ]
        ),
        "label": torch.zeros(2),
        "sequence_loss_length": torch.tensor([2, 1]),
    }

    loss = gmm_train_loss(output, batch)
    expected = masked_sequence_gmm_nll(
        means=output.means,
        log_variances=output.log_variances,
        mixing_logits=output.mixing_logits,
        targets=batch["next_reorder_size"],
        sequence_lengths=batch["sequence_loss_length"],
    )

    assert torch.allclose(loss, expected)


def test_gmm_validation_loss_uses_final_mixture_params_and_label():
    output = SimpleNamespace(
        means=torch.ones(2, 3, 1) * 99,
        log_variances=torch.zeros(2, 3, 1),
        mixing_logits=torch.zeros(2, 3, 1),
        final_means=torch.zeros(2, 1),
        final_log_variances=torch.zeros(2, 1),
        final_mixing_logits=torch.zeros(2, 1),
    )
    batch = {
        "next_reorder_size": torch.ones(2, 3),
        "label": torch.tensor([0.0, 1.0]),
        "sequence_loss_length": torch.tensor([1, 1]),
    }

    loss = gmm_validation_loss(output, batch)
    expected = gmm_nll(
        means=output.final_means,
        log_variances=output.final_log_variances,
        mixing_logits=output.final_mixing_logits,
        targets=batch["label"],
    )

    assert torch.allclose(loss, expected)
