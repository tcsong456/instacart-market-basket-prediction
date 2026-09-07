import torch

from instacart_rnn.models.encoders import (
    ClampOneHotEncoding,
    DenseProductNameEncoder,
)


def test_clamp_one_hot_encoding_sets_in_range_index():
    encoded = ClampOneHotEncoding(num_classes=20)(torch.tensor([[3, 0]]))

    assert encoded.shape == (1, 2, 20)
    assert encoded[0, 0, 3] == 1
    assert encoded[0, 0].sum() == 1
    assert encoded[0, 1, 0] == 1
    assert encoded[0, 1].sum() == 1


def test_clamp_one_hot_encoding_zeros_out_of_range_indices():
    encoded = ClampOneHotEncoding(num_classes=20)(torch.tensor([[25, -1]]))

    assert encoded.shape == (1, 2, 20)
    assert torch.equal(encoded[0, 0], torch.zeros(20))
    assert torch.equal(encoded[0, 1], torch.zeros(20))


def test_product_name_encoder_returns_fixed_output_dim():
    encoder = DenseProductNameEncoder(vocab_size=8, output_dim=4)
    encoded = encoder(torch.tensor([[1, 3, 1], [2, 0, 2]]))

    assert encoded.shape == (2, 4)


def test_product_name_encoder_ignores_out_of_range_tokens_in_bag():
    encoder = DenseProductNameEncoder(
        vocab_size=8,
        output_dim=4,
    )

    valid_only = torch.tensor([[2]])
    with_invalid = torch.tensor([[2, 10, -1]])

    assert torch.allclose(
        encoder(with_invalid),
        encoder(valid_only),
    )


def test_product_name_encoder_all_out_of_range_tokens_match_empty_bag():
    encoder = DenseProductNameEncoder(vocab_size=8, output_dim=4)
    all_out_of_range = torch.tensor([[10, 11, 12]])
    other_out_of_range = torch.tensor([[20, 21, 22]])

    assert torch.allclose(encoder(all_out_of_range), encoder(other_out_of_range))
