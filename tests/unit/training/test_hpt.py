import json
from pathlib import Path

import optuna
import pytest

from instacart_rnn.training.hpt import (
    objective,
    parse_args,
    run_hpt,
    sqlite_storage_url,
)
from instacart_rnn.training.runner import TrainingRunResult

REQUIRED_ARGS = [
    "hpt.py",
    "--train-path",
    "train",
    "--validation-path",
    "val",
    "--output-path",
    "out",
    "--model",
    "product",
]


def _objective_kwargs(tmp_path, **overrides):
    values = {
        "model_name": "product",
        "train_path": "train",
        "validation_path": "val",
        "output_path": tmp_path,
        "epochs": 3,
        "seed": 7,
        "amp": False,
        "pin_memory": False,
        "early_stopping": 2,
    }
    values.update(overrides)
    return values


def test_sqlite_storage_url_uses_posix_path_without_backslashes(tmp_path):
    db_path = tmp_path / "nested" / "optuna.db"
    db_path.parent.mkdir()

    url = sqlite_storage_url(db_path)

    assert url.startswith("sqlite:///")
    assert "\\" not in url
    assert url.endswith("optuna.db")

    study = optuna.create_study(storage=url)
    study.optimize(lambda trial: 0.0, n_trials=1)

    assert db_path.exists()


def test_objective_returns_best_validation_loss_from_training_run(
    tmp_path,
    mocker,
):
    run_training = mocker.patch(
        "instacart_rnn.training.hpt.run_training",
        return_value=TrainingRunResult(
            best_validation_loss=0.37,
            best_epoch=2,
            checkpoint_path="ckpt",
        ),
    )

    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    loss = objective(trial, **_objective_kwargs(tmp_path))

    config = run_training.call_args.args[0]

    assert loss == pytest.approx(0.37)
    assert config.model_name == "product"
    assert config.train_path == "train"
    assert config.validation_path == "val"
    assert config.epochs == 3
    assert config.lstm_size in {64, 128, 256}
    assert config.batch_size in {256, 512, 1024}
    assert config.amp is False
    assert config.pin_memory is False
    assert config.early_stopping == 2
    assert config.on_validation_end is not None
    assert Path(run_training.call_args.kwargs["paths"].root) == (
        tmp_path / "trial_0000"
    )


def test_objective_prunes_trial_when_optuna_requests_it(tmp_path, mocker):
    def fake_training(config, paths):
        config.on_validation_end(0, 0.8)
        return TrainingRunResult(
            best_validation_loss=0.8,
            best_epoch=0,
            checkpoint_path=paths.best_checkpoint,
        )

    mocker.patch(
        "instacart_rnn.training.hpt.run_training",
        side_effect=fake_training,
    )

    trial = mocker.Mock()
    trial.number = 4
    trial.params = {"learning_rate": 1e-3}
    trial.suggest_float.side_effect = [1e-3, 1e-5]
    trial.suggest_categorical.side_effect = [512, 2.0, 128]
    trial.should_prune.return_value = True

    with pytest.raises(optuna.TrialPruned, match="Trial 4 pruned at epoch 0"):
        objective(trial, **_objective_kwargs(tmp_path))

    trial.report.assert_called_once_with(0.8, step=0)


def test_run_hpt_writes_best_params_and_sqlite_storage(tmp_path, mocker):
    run_training = mocker.patch(
        "instacart_rnn.training.hpt.run_training",
        return_value=TrainingRunResult(
            best_validation_loss=0.2,
            best_epoch=0,
            checkpoint_path="ckpt",
        ),
    )

    study = run_hpt(
        model_name="product",
        train_path="train",
        validation_path="val",
        output_path=str(tmp_path),
        n_trials=1,
        epochs=1,
        seed=0,
        amp=True,
        pin_memory=True,
        early_stopping=5,
    )

    config = run_training.call_args.args[0]
    payload = json.loads((tmp_path / "best_params.json").read_text())

    assert config.amp is True
    assert config.pin_memory is True
    assert config.early_stopping == 5
    assert (tmp_path / "optuna.db").exists()
    assert study.best_value == pytest.approx(0.2)
    assert payload["validation_loss"] == pytest.approx(0.2)
    assert payload["trial_number"] == study.best_trial.number
    assert payload["parameters"] == study.best_params


def test_objective_forwards_amp_pin_memory_and_early_stopping(tmp_path, mocker):
    run_training = mocker.patch(
        "instacart_rnn.training.hpt.run_training",
        return_value=TrainingRunResult(
            best_validation_loss=0.1,
            best_epoch=0,
            checkpoint_path="ckpt",
        ),
    )

    study = optuna.create_study(direction="minimize")
    trial = study.ask()

    objective(
        trial,
        **_objective_kwargs(
            tmp_path,
            amp=True,
            pin_memory=True,
            early_stopping=5,
        ),
    )

    config = run_training.call_args.args[0]

    assert config.amp is True
    assert config.pin_memory is True
    assert config.early_stopping == 5


def test_parse_args_parses_required_arguments_and_defaults(monkeypatch):
    monkeypatch.setattr("sys.argv", REQUIRED_ARGS)

    args = parse_args()

    assert args.train_path == "train"
    assert args.validation_path == "val"
    assert args.output_path == "out"
    assert args.model == "product"
    assert args.trials == 20
    assert args.epochs == 5
    assert args.seed == 42
    assert args.early_stopping == 2
    assert args.amp is False
    assert args.pin_memory is False


def test_parse_args_parses_optional_arguments(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            *REQUIRED_ARGS,
            "--trials",
            "8",
            "--epochs",
            "3",
            "--seed",
            "7",
            "--early-stopping",
            "5",
            "--amp",
            "--pin-memory",
        ],
    )

    args = parse_args()

    assert args.trials == 8
    assert args.epochs == 3
    assert args.seed == 7
    assert args.early_stopping == 5
    assert args.amp is True
    assert args.pin_memory is True


def test_parse_args_raises_when_required_argument_is_missing(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "hpt.py",
            "--train-path",
            "train",
            "--validation-path",
            "val",
            "--model",
            "product",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        parse_args()

    assert exc_info.value.code == 2
