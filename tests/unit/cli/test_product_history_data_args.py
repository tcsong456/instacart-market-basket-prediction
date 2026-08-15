import pytest

from instacart_etl_rnn.cli.build_product_history_dataset import (
    parse_args,
)


def test_parse_args_parses_required_arguments(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_product_history_data",
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

    args = parse_args()

    assert args.input_path == "silver"
    assert args.data_path == "bronze"
    assert args.output_path == "gold"
    assert args.contract_path == "contracts"


@pytest.mark.parametrize(
    "argv",
    [
        [
            "build_product_history_data",
            "--data-path",
            "bronze",
            "--output-path",
            "gold",
            "--contract-path",
            "contracts",
        ],
        [
            "build_product_history_data",
            "--input-path",
            "silver",
            "--output-path",
            "gold",
            "--contract-path",
            "contracts",
        ],
        [
            "build_product_history_data",
            "--input-path",
            "silver",
            "--data-path",
            "bronze",
            "--contract-path",
            "contracts",
        ],
        [
            "build_product_history_data",
            "--input-path",
            "silver",
            "--data-path",
            "bronze",
            "--output-path",
            "gold",
        ],
    ],
)
def test_parse_args_requires_all_arguments(
    monkeypatch,
    argv,
):
    monkeypatch.setattr(
        "sys.argv",
        argv,
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
