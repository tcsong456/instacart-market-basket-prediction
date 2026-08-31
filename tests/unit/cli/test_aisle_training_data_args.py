import pytest

from instacart_etl_rnn.cli.build_aisle_training_dataset import parse_args


def test_parse_args_parses_required_arguments_and_default_pad_length(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_aisle_training_data",
            "--input-path",
            "gold",
            "--output-path",
            "training",
            "--contract-path",
            "contracts",
            "--mode",
            "train",
        ],
    )

    args = parse_args()

    assert args.input_path == "gold"
    assert args.output_path == "training"
    assert args.contract_path == "contracts"
    assert args.pad_length == 30
    assert args.mode == "train"


def test_parse_args_parses_custom_pad_length(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_aisle_training_data",
            "--input-path",
            "gold",
            "--output-path",
            "training",
            "--contract-path",
            "contracts",
            "--mode",
            "validation",
            "--pad-length",
            "50",
        ],
    )

    args = parse_args()

    assert args.pad_length == 50
    assert args.mode == "validation"


def test_parse_args_rejects_invalid_mode(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_aisle_training_data",
            "--input-path",
            "gold",
            "--output-path",
            "training",
            "--contract-path",
            "contracts",
            "--mode",
            "stacking_train",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2


def test_parse_args_raises_when_required_argument_is_missing(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_aisle_training_data",
            "--input-path",
            "gold",
            "--output-path",
            "training",
            "--mode",
            "train",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
