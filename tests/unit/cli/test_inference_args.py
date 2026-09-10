import pytest

from instacart_rnn.inference import parse_args

REQUIRED_ARGS = [
    "inference.py",
    "--model",
    "product",
    "--eval-path",
    "eval",
    "--checkpoint-path",
    "ckpt",
    "--output-path",
    "output",
]


def test_parse_args_parses_required_arguments_and_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", REQUIRED_ARGS)

    args = parse_args()

    assert args.model == "product"
    assert args.eval_path == "eval"
    assert args.checkpoint_path == "ckpt"
    assert args.output_path == "output"

    assert args.rows_per_write == 100000
    assert args.batch_size == 512
    assert args.read_batch_size == 4096
    assert args.num_workers == 0
    assert args.amp is False
    assert args.pin_memory is False


def test_parse_args_parses_optional_arguments(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            *REQUIRED_ARGS,
            "--rows-per-write",
            "25000",
            "--batch-size",
            "128",
            "--read-batch-size",
            "2048",
            "--num-workers",
            "4",
            "--amp",
            "--pin-memory",
        ],
    )

    args = parse_args()

    assert args.rows_per_write == 25000
    assert args.batch_size == 128
    assert args.read_batch_size == 2048
    assert args.num_workers == 4
    assert args.amp is True
    assert args.pin_memory is True


def test_parse_args_raises_when_required_argument_is_missing(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "inference.py",
            "--model",
            "product",
            "--checkpoint-path",
            "ckpt",
            "--output-path",
            "output",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
