import pytest

from instacart_etl_rnn.cli.build_user_dataset import parse_args


def test_parse_args_parses_required_arguments(
    monkeypatch,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_user",
            "--path",
            "gs://bucket/silver",
            "--contract-path",
            "gs://bucket/contracts",
        ],
    )

    args = parse_args()

    assert args.path == "gs://bucket/silver"
    assert args.contract_path == "gs://bucket/contracts"


@pytest.mark.parametrize(
    "argv",
    [
        [
            "build_user",
            "--contract-path",
            "contracts",
        ],
        [
            "build_user",
            "--path",
            "silver",
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
