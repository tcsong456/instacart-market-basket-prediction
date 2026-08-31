import pytest

from instacart_etl_rnn.cli.build_aisle_history_dataset import parse_args


def test_parse_args_parses_required_arguments_and_mode(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_aisle_history_data",
            "--input-path",
            "silver",
            "--data-path",
            "bronze",
            "--output-path",
            "gold",
            "--contract-path",
            "contracts",
            "--mode",
            "validation",
        ],
    )

    args = parse_args()

    assert args.input_path == "silver"
    assert args.data_path == "bronze"
    assert args.output_path == "gold"
    assert args.contract_path == "contracts"
    assert args.mode == "validation"


@pytest.mark.parametrize(
    "mode",
    ["train", "validation", "evaluation"],
)
def test_parse_args_accepts_allowed_modes(
    monkeypatch,
    mode,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_aisle_history_data",
            "--input-path",
            "silver",
            "--data-path",
            "bronze",
            "--output-path",
            "gold",
            "--contract-path",
            "contracts",
            "--mode",
            mode,
        ],
    )

    args = parse_args()

    assert args.mode == mode


def test_parse_args_rejects_invalid_mode(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_aisle_history_data",
            "--input-path",
            "silver",
            "--data-path",
            "bronze",
            "--output-path",
            "gold",
            "--contract-path",
            "contracts",
            "--mode",
            "base_train",
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
            "build_aisle_history_data",
            "--input-path",
            "silver",
            "--data-path",
            "bronze",
            "--output-path",
            "gold",
            "--contract-path",
            "contracts",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
