import torch

from instacart_rnn.dataset import create_reorder_size_dataloader
from instacart_rnn.models.reorder_size_gmm_model import ReorderSizeGmmModel
from tests.unit.rnn.test_dataset import _write_reorder_size_training_dataset

HIDDEN_SIZE = 50
N_COMPONENTS = 3


def test_reorder_size_gmm_model_forward_and_backward_on_loader_batch(tmp_path):
    dataset_path = tmp_path / "training"
    _write_reorder_size_training_dataset(dataset_path, [[1, 2]])
    batch = next(
        iter(
            create_reorder_size_dataloader(
                dataset_path,
                batch_size=2,
                read_batch_size=2,
                num_workers=0,
                drop_last=False,
                shuffle=False,
            )
        )
    )

    model = ReorderSizeGmmModel(
        lstm_size=8,
    )

    output = model(batch)

    batch_size = batch["user_id"].shape[0]
    sequence_length = batch["reorder_size_history"].shape[1]
    batch_indices = torch.arange(batch_size)
    final_indices = batch["history_length"] - 1

    assert output.means.shape == (batch_size, sequence_length, N_COMPONENTS)
    assert output.log_variances.shape == (batch_size, sequence_length, N_COMPONENTS)
    assert output.mixing_logits.shape == (batch_size, sequence_length, N_COMPONENTS)
    assert output.hidden_states.shape == (batch_size, sequence_length, HIDDEN_SIZE)
    assert output.final_means.shape == (batch_size, N_COMPONENTS)
    assert output.final_log_variances.shape == (batch_size, N_COMPONENTS)
    assert output.final_mixing_logits.shape == (batch_size, N_COMPONENTS)
    assert torch.equal(
        output.final_means,
        output.means[batch_indices, final_indices],
    )
    assert torch.equal(
        output.final_log_variances,
        output.log_variances[batch_indices, final_indices],
    )
    assert torch.equal(
        output.final_mixing_logits,
        output.mixing_logits[batch_indices, final_indices],
    )

    output.final_means.sum().backward()

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters

    for name, parameter in trainable_parameters:
        assert parameter.grad is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has non-finite gradients"
