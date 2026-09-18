import pytest
import torch

from instacart_rnn.models.reorder_size_gmm import ReorderSizeGMMOutput
from instacart_rnn.models.representation import (
    GMM_OUTPUT_TENSOR_NAMES,
    build_gmm_features,
    gmm_output_transform,
)

HIDDEN_SIZE = 50
N_COMPONENTS = 3
GMM_VECTOR_TENSOR_NAMES = (
    "candidate_nlls",
    "final_states",
)


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


def _peaked_gmm_output(
    peaks: torch.Tensor,
    history_length: torch.Tensor,
    sequence_length: int = 5,
) -> ReorderSizeGMMOutput:
    batch_size = peaks.size(0)
    means = torch.zeros(batch_size, sequence_length, N_COMPONENTS)
    means[:, :, 0] = peaks.unsqueeze(1)

    log_variances = torch.full(
        (batch_size, sequence_length, N_COMPONENTS),
        10.0,
    )
    log_variances[:, :, 0] = -10.0

    mixing_logits = torch.full(
        (batch_size, sequence_length, N_COMPONENTS),
        -20.0,
    )
    mixing_logits[:, :, 0] = 20.0

    batch_indices = torch.arange(batch_size)
    final_indices = history_length - 1

    return ReorderSizeGMMOutput(
        hidden_states=torch.randn(batch_size, sequence_length, HIDDEN_SIZE),
        means=means,
        log_variances=log_variances,
        mixing_logits=mixing_logits,
        final_means=means[batch_indices, final_indices],
        final_log_variances=log_variances[batch_indices, final_indices],
        final_mixing_logits=mixing_logits[batch_indices, final_indices],
    )


def test_build_gmm_features_returns_named_gmm_tensors():
    output = _gmm_output()
    history_length = torch.tensor([3, 5])

    representation = build_gmm_features(
        output,
        history_length,
        max_candidate=24,
    )

    assert set(representation) == set(GMM_OUTPUT_TENSOR_NAMES)
    assert representation["final_states"].shape == (2, HIDDEN_SIZE)
    assert representation["candidate_nlls"].shape == (2, 25)
    assert torch.equal(
        representation["nll_0"],
        representation["candidate_nlls"][:, 0],
    )

    for name in GMM_OUTPUT_TENSOR_NAMES:
        assert torch.isfinite(representation[name]).all()

        if name in GMM_VECTOR_TENSOR_NAMES:
            continue

        assert representation[name].shape == (2,)

    assert (representation["mode_size"] >= 0).all()
    assert (representation["mode_size"] <= 24).all()


def test_gmm_output_transform_honors_max_candidate():
    output = _gmm_output()
    batch = {"history_length": torch.tensor([3, 5])}

    transform = gmm_output_transform(max_candidate=40)
    tensors = transform(output, batch)

    assert set(tensors) == set(GMM_OUTPUT_TENSOR_NAMES)
    assert tensors["final_states"].shape == (2, HIDDEN_SIZE)
    assert tensors["candidate_nlls"].shape == (2, 41)
    assert (tensors["mode_size"] >= 0).all()
    assert (tensors["mode_size"] <= 40).all()


def test_build_gmm_features_peaked_mixture_matches_candidate_mode():
    history_length = torch.tensor([3, 5])
    peaks = torch.tensor([7.0, 3.0])
    max_candidate = 24
    output = _peaked_gmm_output(peaks, history_length)

    representation = build_gmm_features(
        output,
        history_length,
        max_candidate=max_candidate,
    )

    expected_mode = torch.argmin(representation["candidate_nlls"], dim=-1)

    assert torch.equal(representation["mode_size"], expected_mode)
    assert representation["mode_size"].tolist() == [7, 3]
    assert representation["candidate_expected_size"] == pytest.approx(
        peaks,
        abs=0.25,
    )
