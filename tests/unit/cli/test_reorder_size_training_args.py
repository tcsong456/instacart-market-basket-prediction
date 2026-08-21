import pytest

from instacart_etl_rnn.cli.build_reorder_size_training_dataset import parse_args


def test_parse_args_parses_required_arguments_and_default_pad_length(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_reorder_size_training_data ",
            "--input-path",
            "gold",
            "--output-path",
            "training",
            "--contract-path",
            "contracts",
        ],
    )

    args = parse_args()

    assert args.input_path == "gold"
    assert args.output_path == "training"
    assert args.contract_path == "contracts"
    assert args.pad_length == 30


def test_parse_args_parses_custom_pad_length(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_reorder_size_training_data ",
            "--input-path",
            "gold",
            "--output-path",
            "training",
            "--contract-path",
            "contracts",
            "--pad-length",
            "50",
        ],
    )

    args = parse_args()

    assert args.pad_length == 50


def test_parse_args_raises_when_required_argument_is_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_reorder_size_training_data ",
            "--input-path",
            "gold",
            "--output-path",
            "training",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
