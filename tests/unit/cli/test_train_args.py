import pytest

from instacart_rnn.train import parse_args

REQUIRED_ARGS = [
    "train.py",
    "--model",
    "product",
    "--train-path",
    "train",
    "--validation-path",
    "val",
    "--checkpoint-path",
    "ckpt",
]


def test_parse_args_parses_required_arguments_and_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", REQUIRED_ARGS)

    args = parse_args()

    assert args.model == "product"
    assert args.train_path == "train"
    assert args.validation_path == "val"
    assert args.checkpoint_path == "ckpt"

    assert args.epochs == 10
    assert args.batch_size == 512
    assert args.read_batch_size == 4096
    assert args.learning_rate == pytest.approx(1e-3)
    assert args.weight_decay == pytest.approx(0.0)
    assert args.num_workers == 0
    assert args.seed == 42
    assert args.early_stopping == 3
    assert args.grad_clip_norm is None
    assert args.warm_start is False
    assert args.amp is False
    assert args.pin_memory is False


def test_parse_args_parses_optional_arguments(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            *REQUIRED_ARGS,
            "--epochs",
            "3",
            "--batch-size",
            "128",
            "--read-batch-size",
            "2048",
            "--learning-rate",
            "0.01",
            "--weight-decay",
            "0.1",
            "--num-workers",
            "4",
            "--seed",
            "7",
            "--early-stopping",
            "5",
            "--grad-clip-norm",
            "1.5",
            "--warm-start",
            "--amp",
            "--pin-memory",
        ],
    )

    args = parse_args()

    assert args.epochs == 3
    assert args.batch_size == 128
    assert args.read_batch_size == 2048
    assert args.learning_rate == pytest.approx(0.01)
    assert args.weight_decay == pytest.approx(0.1)
    assert args.num_workers == 4
    assert args.seed == 7
    assert args.early_stopping == 5
    assert args.grad_clip_norm == pytest.approx(1.5)
    assert args.warm_start is True
    assert args.amp is True
    assert args.pin_memory is True


def test_parse_args_raises_when_required_argument_is_missing(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "train",
            "--model",
            "product",
            "--validation-path",
            "val",
            "--checkpoint-path",
            "ckpt",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
