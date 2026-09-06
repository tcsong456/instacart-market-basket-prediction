import torch

from instacart_rnn.models.product_rnn import ProductRNN

HIDDEN_SIZE = 50
INPUT_SIZE = 8


def _product_rnn(input_size: int = INPUT_SIZE) -> ProductRNN:
    return ProductRNN(
        input_size=input_size,
        lstm_size=4,
        dilations=[1, 2],
        filter_widths=[2, 2],
        skip_channels=3,
        residual_channels=4,
    )


def _forward(model, *, batch_size, sequence_length, history_length):
    x = torch.randn(batch_size, sequence_length, INPUT_SIZE)
    return model(x, history_length)


def test_product_rnn_returns_sequence_and_final_outputs_for_mixed_lengths():
    model = _product_rnn()
    history_length = torch.tensor([1, 3, 5])

    output = _forward(
        model,
        batch_size=3,
        sequence_length=5,
        history_length=history_length,
    )

    assert output.logits.shape == (3, 5)
    assert output.final_states.shape == (3, HIDDEN_SIZE)
    assert output.final_logits.shape == (3,)


def test_product_rnn_gathers_final_outputs_at_last_valid_timestep():
    model = _product_rnn()
    history_length = torch.tensor([1, 3, 5])

    output = _forward(
        model,
        batch_size=3,
        sequence_length=5,
        history_length=history_length,
    )

    batch_indices = torch.arange(3)
    final_indices = history_length - 1

    assert torch.equal(
        output.final_logits,
        output.logits[batch_indices, final_indices],
    )


def test_product_rnn_sets_zero_index_final_outputs_for_one_step_history():
    model = _product_rnn()
    history_length = torch.tensor([1, 1])

    output = _forward(
        model,
        batch_size=2,
        sequence_length=4,
        history_length=history_length,
    )

    assert torch.equal(output.final_logits, output.logits[:, 0])


def test_product_rnn_backward_produces_finite_gradients():
    model = _product_rnn()

    output = _forward(
        model,
        batch_size=2,
        sequence_length=4,
        history_length=torch.tensor([2, 4]),
    )

    output.logits.sum().backward()

    trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    assert trainable_parameters

    for name, parameter in trainable_parameters:
        assert parameter.grad is not None, f"{name} did not receive a gradient"

        assert torch.isfinite(parameter.grad).all(), f"{name} has non-finite gradients"
