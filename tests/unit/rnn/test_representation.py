import torch

from instacart_rnn.models.reorder_size_gmm import ReorderSizeGMMOutput
from instacart_rnn.models.representation import (
    build_final_representation,
    gmm_output_transform,
)

HIDDEN_SIZE = 50
N_COMPONENTS = 3


def _gmm_output(batch_size: int = 2, sequence_length: int = 5) -> ReorderSizeGMMOutput:
    means = torch.randn(batch_size, sequence_length, N_COMPONENTS)
    log_variances = torch.zeros(batch_size, sequence_length, N_COMPONENTS)
    mixing_logits = torch.zeros(batch_size, sequence_length, N_COMPONENTS)
    batch_indices = torch.arange(batch_size)
    final_indices = torch.tensor([2, 4])

    return ReorderSizeGMMOutput(
        hidden_states=torch.randn(batch_size, sequence_length, HIDDEN_SIZE),
        means=means,
        log_variances=log_variances,
        mixing_logits=mixing_logits,
        final_means=means[batch_indices, final_indices],
        final_log_variances=log_variances[batch_indices, final_indices],
        final_mixing_logits=mixing_logits[batch_indices, final_indices],
    )


def test_build_final_representation_concatenates_hidden_state_and_candidate_nlls():
    output = _gmm_output()
    history_length = torch.tensor([3, 5])

    representation = build_final_representation(
        output,
        history_length,
        max_candidate=24,
    )

    assert representation.shape == (2, HIDDEN_SIZE + 25)
    assert torch.isfinite(representation).all()


def test_gmm_output_transform_honors_max_candidate():
    output = _gmm_output()
    batch = {"history_length": torch.tensor([3, 5])}

    transform = gmm_output_transform(max_candidate=40)
    tensors = transform(output, batch)

    assert set(tensors) == {"final_states"}
    assert tensors["final_states"].shape == (2, HIDDEN_SIZE + 41)
