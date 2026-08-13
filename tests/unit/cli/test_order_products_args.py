import pytest

from instacart_etl_rnn.cli.build_order_products_dataset import (
    parse_args,
)


def test_parse_args_parses_required_arguments(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_order_products",
            "--input-path",
            "gs://bucket/bronze",
            "--contract-path",
            "gs://bucket/contracts",
            "--output-path",
            "gs://bucket/silver/order_products",
        ],
    )

    args = parse_args()

    assert args.input_path == "gs://bucket/bronze"
    assert args.contract_path == "gs://bucket/contracts"
    assert args.output_path == "gs://bucket/silver/order_products"


@pytest.mark.parametrize(
    "argv",
    [
        [
            "build_order_products",
            "--contract-path",
            "contracts",
            "--output-path",
            "silver",
        ],
        [
            "build_order_products",
            "--input-path",
            "bronze",
            "--output-path",
            "silver",
        ],
        [
            "build_order_products",
            "--input-path",
            "bronze",
            "--contract-path",
            "contracts",
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
