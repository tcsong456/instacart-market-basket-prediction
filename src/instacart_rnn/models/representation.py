from typing import Any

import torch

from instacart_rnn.models.reorder_size_gmm import ReorderSizeGMMOutput
from instacart_rnn.training.losses import gaussian_mixture_nll
from instacart_rnn.training.trainer import TensorBatch

GMM_OUTPUT_TENSOR_NAMES = (
    "nll_0",
    "mode_size",
    "candidate_expected_size",
    "candidate_entropy",
    "gmm_expected_size",
    "candidate_nlls",
    "final_states",
)


def build_gmm_features(
    output: ReorderSizeGMMOutput,
    history_length: torch.Tensor,
    max_candidate: int = 24,
) -> dict[str, torch.Tensor]:
    """Build named GMM inference tensors from hidden states and candidate NLLs.

    Selects the final valid timestep of each sequence and evaluates the
    Gaussian mixture on integer reorder sizes from 0 through
    ``max_candidate``. The candidate NLL curve is exported together with
    named summaries; it is not concatenated onto the hidden state.

    Args:
        output: Model output containing hidden states, Gaussian component
            means and log-variances, and mixture logits.
        history_length: Number of valid timesteps for each sequence, with
            shape ``(batch_size,)``.
        max_candidate: Maximum candidate reorder size to evaluate.

    Returns:
        Mapping of inference tensor names to tensors. ``final_states`` has
        shape ``(batch_size, hidden_size)``, ``candidate_nlls`` has shape
        ``(batch_size, max_candidate + 1)``, and the remaining tensors are
        scalars per row with shape ``(batch_size,)``.
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

    log_probs = torch.log_softmax(
        -final_candidate_nlls,
        dim=-1,
    )
    probs = torch.exp(log_probs)

    mixing_probs = torch.softmax(
        output.final_mixing_logits.float(),
        dim=-1,
    )

    return {
        "nll_0": final_candidate_nlls[:, 0],
        # Candidate-grid mode, not continuous GMM mode.
        "mode_size": torch.argmin(
            final_candidate_nlls,
            dim=-1,
        ),
        # Expectation of normalized density values over
        # the truncated integer grid, not E[Y] of the GMM.
        "candidate_expected_size": (probs * candidates).sum(dim=-1),
        # Entropy of the pseudo-PMF over the truncated grid.
        "candidate_entropy": -(probs * log_probs).sum(dim=-1),
        "gmm_expected_size": (mixing_probs * output.final_means.float()).sum(dim=-1),
        "candidate_nlls": final_candidate_nlls,
        "final_states": final_hidden_states,
    }


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
        return build_gmm_features(
            output=output,
            history_length=batch["history_length"],
            max_candidate=max_candidate,
        )

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
