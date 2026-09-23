import pytest

from instacart_rnn.run import parse_args

REQUIRED_ARGS = [
    "run.py",
    "--model",
    "product",
    "--run-id",
    "run-1",
    "--data-root",
    "gs://gold/training/curated/t0",
    "--runs-root",
    "gs://runs",
]


def test_parse_args_parses_required_arguments_and_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", REQUIRED_ARGS)

    args = parse_args()

    assert args.model == "product"
    assert args.run_id == "run-1"
    assert args.data_root == "gs://gold/training/curated/t0"
    assert args.runs_root == "gs://runs"

    assert args.epochs == 10
    assert args.train_batch_size == 256
    assert args.eval_batch_size == 512
    assert args.read_batch_size == 4096
    assert args.lstm_size == 256
    assert args.max_candidate == 24
    assert args.rows_per_write == 100000
    assert args.learning_rate == pytest.approx(1e-3)
    assert args.weight_decay == pytest.approx(0.0)
    assert args.num_workers == 0
    assert args.seed == 42
    assert args.early_stopping == 2
    assert args.grad_clip_norm is None
    assert args.git_commit == ""
    assert args.image == ""
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
            "--train-batch-size",
            "128",
            "--eval-batch-size",
            "64",
            "--read-batch-size",
            "2048",
            "--lstm-size",
            "64",
            "--max-candidate",
            "40",
            "--rows-per-write",
            "25000",
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
            "--git-commit",
            "abc123",
            "--image",
            "img:tag",
            "--warm-start",
            "--amp",
            "--pin-memory",
        ],
    )

    args = parse_args()

    assert args.epochs == 3
    assert args.train_batch_size == 128
    assert args.eval_batch_size == 64
    assert args.read_batch_size == 2048
    assert args.lstm_size == 64
    assert args.max_candidate == 40
    assert args.rows_per_write == 25000
    assert args.learning_rate == pytest.approx(0.01)
    assert args.weight_decay == pytest.approx(0.1)
    assert args.num_workers == 4
    assert args.seed == 7
    assert args.early_stopping == 5
    assert args.grad_clip_norm == pytest.approx(1.5)
    assert args.git_commit == "abc123"
    assert args.image == "img:tag"
    assert args.warm_start is True
    assert args.amp is True
    assert args.pin_memory is True


def test_parse_args_raises_when_required_argument_is_missing(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "--model",
            "product",
            "--data-root",
            "gs://gold/training/curated/t0",
            "--runs-root",
            "gs://runs",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
