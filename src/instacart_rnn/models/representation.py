from typing import Any

import torch

from instacart_rnn.models.reorder_size_gmm import ReorderSizeGMMOutput
from instacart_rnn.training.losses import gaussian_mixture_nll
from instacart_rnn.training.trainer import TensorBatch


def build_final_representation(
    output: ReorderSizeGMMOutput,
    history_length: torch.Tensor,
    max_candidate: int = 24,
) -> torch.Tensor:
    """Build the final representation from hidden states and candidate NLLs.

    Selects the final valid timestep of each sequence and concatenates its
    hidden state with the Gaussian mixture negative log-likelihood for each
    candidate reorder size from 0 through ``max_candidate``.

    Args:
        output: Model output containing hidden states, Gaussian component
            means and log-variances, and mixture logits.
        history_length: Number of valid timesteps for each sequence, with
            shape ``(batch_size,)``.
        max_candidate: Maximum candidate reorder size to evaluate.

    Returns:
        Tensor containing the final hidden state concatenated with candidate
        NLLs, with shape
        ``(batch_size, hidden_size + max_candidate + 1)``.
    """

    batch_indices = torch.arange(
        history_length.size(0),
        device=history_length.device,
    )
    final_indices = history_length - 1

    # [B, T, H] -> [B, H]
    final_hidden_states = output.hidden_states[
        batch_indices,
        final_indices,
    ]

    # [C]
    candidates = torch.arange(
        max_candidate + 1,
        device=output.final_means.device,
        dtype=output.final_means.dtype,
    )

    # Broadcast final GMM params [B, K, 1] against candidates [C].
    final_candidate_nlls = gaussian_mixture_nll(
        means=output.final_means.unsqueeze(-1),
        log_variances=output.final_log_variances.unsqueeze(-1),
        mixing_logits=output.final_mixing_logits.unsqueeze(-1),
        targets=candidates,
        component_dim=1,
    )

    # [B, H] + [B, C] -> [B, H + C]
    return torch.cat(
        [
            final_hidden_states,
            final_candidate_nlls,
        ],
        dim=-1,
    )


def gmm_output_transform(
    *,
    max_candidate: int = 24,
    **_: Any,
):
    """Build a GMM inference transform for the given candidate range."""

    def transform(
        output: ReorderSizeGMMOutput,
        batch: TensorBatch,
    ) -> dict[str, torch.Tensor]:
        return {
            "final_states": build_final_representation(
                output=output,
                history_length=batch["history_length"],
                max_candidate=max_candidate,
            ),
        }

    return transform


def binary_output_transform(**_: Any):
    """Build a binary-classification inference transform."""

    def transform(
        output,
        batch,
    ) -> dict[str, torch.Tensor]:
        return {
            "final_states": output.final_states,
            "final_logits": output.final_logits,
            "final_probabilities": torch.sigmoid(output.final_logits),
        }

    return transform
