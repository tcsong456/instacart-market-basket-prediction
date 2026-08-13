import pytest

from instacart_etl_rnn.cli.build_bronze_parquet_dataset import parse_args


def test_parse_args_parses_required_arguments(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_bronze",
            "--csv-path",
            "gs://raw",
            "--parquet-path",
            "gs://bronze",
            "--contract-path",
            "gs://contracts",
        ],
    )

    args = parse_args()

    assert args.csv_path == "gs://raw"
    assert args.parquet_path == "gs://bronze"
    assert args.contract_path == "gs://contracts"


@pytest.mark.parametrize(
    "argv",
    [
        [
            "build_bronze",
            "--parquet-path",
            "gs://bronze",
            "--contract-path",
            "gs://contracts",
        ],
        [
            "build_bronze",
            "--csv-path",
            "gs://raw",
            "--contract-path",
            "gs://contracts",
        ],
        [
            "build_bronze",
            "--csv-path",
            "gs://raw",
            "--parquet-path",
            "gs://bronze",
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
