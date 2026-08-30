import pytest

from instacart_etl_rnn.cli.build_product_training_dataset import parse_args


def test_parse_args_parses_required_arguments_and_defaults(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_product_training_data",
            "--input-path",
            "silver",
            "--raw-path",
            "bronze",
            "--output-path",
            "gold",
            "--contract-path",
            "contracts",
            "--mode",
            "train",
        ],
    )

    args = parse_args()

    assert args.input_path == "silver"
    assert args.raw_path == "bronze"
    assert args.output_path == "gold"
    assert args.contract_path == "contracts"
    assert args.mode == "train"

    assert args.min_word_freq == 5
    assert args.product_name_length == 50
    assert args.encode_length == 50


def test_parse_args_parses_optional_arguments(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_product_training_data",
            "--input-path",
            "silver",
            "--raw-path",
            "bronze",
            "--output-path",
            "gold",
            "--contract-path",
            "contracts",
            "--mode",
            "evaluation",
            "--min-word-freq",
            "10",
            "--product-name-length",
            "40",
            "--encode-length",
            "60",
        ],
    )

    args = parse_args()

    assert args.mode == "evaluation"
    assert args.min_word_freq == 10
    assert args.product_name_length == 40
    assert args.encode_length == 60


def test_parse_args_raises_when_required_argument_is_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_product_training_data",
            "--raw-path",
            "bronze",
            "--output-path",
            "gold",
            "--contract-path",
            "contracts",
            "--mode",
            "train",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
