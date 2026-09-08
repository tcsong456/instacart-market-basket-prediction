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
