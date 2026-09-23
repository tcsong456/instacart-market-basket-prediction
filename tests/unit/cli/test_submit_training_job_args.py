import pytest

from instacart_platform.submit_training_job import parse_args

REQUIRED_ARGS = [
    "submit_training_job.py",
    "--image",
    "img:tag",
    "--template-id",
    "tpl-1",
    "--git-commit",
    "abc123",
    "--timeline",
    "t0",
    "--gold-root",
    "gs://gold",
    "--runs-root",
    "gs://runs",
    "--model-name",
    "product",
]


def test_parse_args_parses_required_arguments_and_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", REQUIRED_ARGS)

    args = parse_args()

    assert args.image == "img:tag"
    assert args.template_id == "tpl-1"
    assert args.git_commit == "abc123"
    assert args.timeline == "t0"
    assert args.gold_root == "gs://gold"
    assert args.runs_root == "gs://runs"
    assert args.model_name == "product"
    assert args.mode == "curated"
    assert args.gpu_type == "NVIDIA GeForce RTX 4090"


def test_parse_args_parses_optional_arguments(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            *REQUIRED_ARGS,
            "--mode",
            "base_train",
            "--gpu-type",
            "NVIDIA L4",
        ],
    )

    args = parse_args()

    assert args.mode == "base_train"
    assert args.gpu_type == "NVIDIA L4"


def test_parse_args_raises_when_required_argument_is_missing(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "submit_training_job.py",
            "--image",
            "img:tag",
            "--template-id",
            "tpl-1",
            "--git-commit",
            "abc123",
            "--timeline",
            "t0",
            "--gold-root",
            "gs://gold",
            "--runs-root",
            "gs://runs",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
