import torch
from torch import nn

from instacart_rnn.dataset import create_product_dataloader
from instacart_rnn.models.product_input_encoder import ProductInputEncoder
from instacart_rnn.models.product_rnn import ProductRNN
from tests.unit.rnn.test_dataset import _write_training_dataset

HIDDEN_SIZE = 50
LSTM_SIZE = 8


def _loader_batch(tmp_path):
    dataset_path = tmp_path / "training"
    _write_training_dataset(dataset_path, [[1, 2]])
    loader = create_product_dataloader(
        dataset_path,
        batch_size=2,
        read_batch_size=2,
        num_workers=0,
        drop_last=False,
        shuffle=False,
    )
    return next(iter(loader))


def _encoder_and_model():
    encoder = ProductInputEncoder(lstm_size=LSTM_SIZE)
    model = ProductRNN(
        input_size=encoder.output_dim,
        lstm_size=LSTM_SIZE,
        dilations=[1, 2],
        filter_widths=[2, 2],
        skip_channels=3,
        residual_channels=4,
    )
    return encoder, model


def _assert_finite_gradients(module: nn.Module) -> None:
    trainable_parameters = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters

    for name, parameter in trainable_parameters:
        assert parameter.grad is not None, f"{name} did not receive a gradient"

        assert torch.isfinite(parameter.grad).all(), f"{name} has non-finite gradients"


def test_encoder_and_product_rnn_forward_on_loader_batch(tmp_path):
    batch = _loader_batch(tmp_path)
    encoder, model = _encoder_and_model()

    encoded = encoder(batch)
    output = model(encoded, batch["history_length"])

    batch_size = batch["user_id"].shape[0]
    sequence_length = batch["is_ordered_history"].shape[1]
    batch_indices = torch.arange(batch_size)
    final_indices = batch["history_length"] - 1

    assert encoded.shape == (batch_size, sequence_length, encoder.output_dim)
    assert output.logits.shape == (batch_size, sequence_length)
    assert output.final_states.shape == (batch_size, HIDDEN_SIZE)
    assert output.final_logits.shape == (batch_size,)
    assert torch.equal(
        output.final_logits,
        output.logits[batch_indices, final_indices],
    )


def test_encoder_and_product_rnn_backward_on_loader_batch(tmp_path):
    batch = _loader_batch(tmp_path)
    encoder, model = _encoder_and_model()

    encoded = encoder(batch)

    output = model(
        encoded,
        batch["history_length"],
    )

    output.final_logits.sum().backward()

    _assert_finite_gradients(encoder)
    _assert_finite_gradients(model)
