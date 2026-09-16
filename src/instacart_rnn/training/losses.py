import math

import torch
import torch.nn.functional as F


def masked_sequence_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Average binary log loss over valid sequence positions.

    Args:
        logits: [B, T] raw model logits.
        targets: [B, T] binary labels.
        sequence_lengths: [B] number of valid loss positions.

    Returns:
        Scalar loss.
    """
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
    )

    time_indices = torch.arange(
        logits.size(1),
        device=logits.device,
    ).unsqueeze(0)

    mask = time_indices < sequence_lengths.unsqueeze(1)

    return (losses * mask).sum() / mask.sum().clamp_min(1)


def masked_sequence_rmse(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute RMSE over valid timesteps in padded sequences.

    Timesteps beyond each sequence's length are masked out and do not
    contribute to the loss.

    Args:
        y: Target values with shape ``(batch_size, sequence_length)``.
        y_hat: Predicted values with shape
            ``(batch_size, sequence_length)``.
        sequence_lengths: Number of valid timesteps for each sequence,
            with shape ``(batch_size,)``.

    Returns:
        Scalar tensor containing the root mean squared error over all
        valid timesteps in the batch.
    """

    y = y.float()

    squared_error = (y - y_hat).square()

    max_sequence_length = y.size(1)

    time_indices = torch.arange(
        max_sequence_length,
        device=y.device,
    ).unsqueeze(0)

    sequence_mask = time_indices < sequence_lengths.unsqueeze(1)

    squared_error = squared_error * sequence_mask

    mean_squared_error = squared_error.sum() / sequence_lengths.sum()

    return torch.sqrt(mean_squared_error)


def gaussian_mixture_nll(
    means: torch.Tensor,
    log_variances: torch.Tensor,
    mixing_logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    component_dim: int = -1,
) -> torch.Tensor:
    """Return unreduced Gaussian mixture negative log-likelihoods.

    ``targets`` must broadcast against ``means``. Mixture weights come from
    log-softmax over ``component_dim``. Log-variances are clamped for
    numerical stability.

    Args:
        means: Gaussian component means.
        log_variances: Gaussian component log-variances.
        mixing_logits: Unnormalized mixture weights.
        targets: Target values broadcastable against ``means``.
        component_dim: Dimension indexing mixture components.

    Returns:
        Negative log-likelihoods with ``component_dim`` removed.
    """
    targets = targets.float()

    log_variances = log_variances.clamp(
        min=-10.0,
        max=10.0,
    )

    log_component_likelihoods = -0.5 * (
        math.log(2.0 * math.pi)
        + log_variances
        + (targets - means).square() * torch.exp(-log_variances)
    )

    log_mixing_coefs = torch.log_softmax(
        mixing_logits,
        dim=component_dim,
    )

    return -torch.logsumexp(
        log_mixing_coefs + log_component_likelihoods,
        dim=component_dim,
    )


def masked_sequence_gmm_nll(
    means: torch.Tensor,
    log_variances: torch.Tensor,
    mixing_logits: torch.Tensor,
    targets: torch.Tensor,
    sequence_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute Gaussian mixture NLL over valid timesteps in padded sequences.

    Computes the negative log-likelihood of each target under a Gaussian
    mixture model (GMM). Timesteps beyond each sequence's length are masked
    out and do not contribute to the loss.

    Args:
        means: Gaussian component means with shape
            ``(batch_size, sequence_length, n_components)``.
        log_variances: Gaussian component log-variances with shape
            ``(batch_size, sequence_length, n_components)``.
        mixing_logits: Unnormalized mixture weights with shape
            ``(batch_size, sequence_length, n_components)``.
        targets: Target values with shape
            ``(batch_size, sequence_length)``.
        sequence_lengths: Number of valid timesteps for each sequence,
            with shape ``(batch_size,)``.

    Returns:
        Scalar tensor containing the mean negative log-likelihood over
        all valid timesteps in the batch.
    """
    nlls = gaussian_mixture_nll(
        means=means,
        log_variances=log_variances,
        mixing_logits=mixing_logits,
        targets=targets.unsqueeze(-1),
    )

    time_indices = torch.arange(
        nlls.size(1),
        device=nlls.device,
    ).unsqueeze(0)

    mask = time_indices < sequence_lengths.unsqueeze(1)

    return (nlls * mask).sum() / mask.sum().clamp_min(1)


def gmm_nll(
    means: torch.Tensor,
    log_variances: torch.Tensor,
    mixing_logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute the mean negative log-likelihood of a Gaussian mixture model.

    Args:
        means: Gaussian component means with shape
            ``(..., n_components)``.
        log_variances: Gaussian component log-variances with shape
            ``(..., n_components)``.
        mixing_logits: Unnormalized mixture weights with shape
            ``(..., n_components)``.
        targets: Target values with shape ``(...)``.

    Returns:
        Scalar tensor containing the mean negative log-likelihood.
    """
    return gaussian_mixture_nll(
        means=means,
        log_variances=log_variances,
        mixing_logits=mixing_logits,
        targets=targets.unsqueeze(-1),
    ).mean()


def bce_train_loss(
    output,
    batch,
):
    return masked_sequence_bce_with_logits(
        logits=output.logits,
        targets=batch["next_is_ordered"],
        sequence_lengths=(batch["sequence_loss_length"]),
    )


def bce_validation_loss(
    output,
    batch,
):
    return F.binary_cross_entropy_with_logits(
        output.final_logits,
        batch["label"].float(),
    )


def gmm_train_loss(
    output,
    batch,
):
    return masked_sequence_gmm_nll(
        means=output.means,
        log_variances=output.log_variances,
        mixing_logits=output.mixing_logits,
        targets=batch["next_reorder_size"],
        sequence_lengths=batch["sequence_loss_length"],
    )


def gmm_validation_loss(
    output,
    batch,
):
    return gmm_nll(
        means=output.final_means,
        log_variances=output.final_log_variances,
        mixing_logits=output.final_mixing_logits,
        targets=batch["label"],
    )
