import torch

from instacart_rnn.dataset import create_reorder_size_dataloader
from instacart_rnn.models.reorder_size_model import ReorderSizeRNNModel
from tests.unit.rnn.test_dataset import _write_reorder_size_training_dataset

HIDDEN_SIZE = 50


def test_reorder_size_rnn_model_forward_and_backward_on_loader_batch(tmp_path):
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

    model = ReorderSizeRNNModel(
        lstm_size=8,
    )

    output = model(batch)

    batch_size = batch["user_id"].shape[0]
    sequence_length = batch["reorder_size_history"].shape[1]
    batch_indices = torch.arange(batch_size)
    final_indices = batch["history_length"] - 1

    assert output.predictions.shape == (batch_size, sequence_length)
    assert output.final_states.shape == (batch_size, HIDDEN_SIZE)
    assert output.final_predictions.shape == (batch_size,)
    assert torch.equal(
        output.final_predictions,
        output.predictions[batch_indices, final_indices],
    )

    output.final_predictions.sum().backward()

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters

    for name, parameter in trainable_parameters:
        assert parameter.grad is not None, f"{name} did not receive a gradient"
        assert torch.isfinite(parameter.grad).all(), f"{name} has non-finite gradients"
