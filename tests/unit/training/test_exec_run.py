import json
from pathlib import Path

import pytest

from instacart_rnn.training.exec_run import execute_run
from instacart_rnn.training.runner import (
    InferenceRunConfig,
    TrainingRunConfig,
    TrainingRunResult,
)


def _training_config(**overrides):
    values = {
        "model_name": "product",
        "train_path": "train.parquet",
        "validation_path": "val.parquet",
        "epochs": 2,
        "batch_size": 8,
        "read_batch_size": 16,
        "lstm_size": 32,
        "learning_rate": 0.01,
    }
    values.update(overrides)
    return TrainingRunConfig(**values)


def _inference_config(**overrides):
    values = {
        "model_name": "product",
        "eval_path": "eval.parquet",
        "checkpoint_path": "",
        "output_path": "",
        "lstm_size": 32,
    }
    values.update(overrides)
    return InferenceRunConfig(**values)


def _execute_run(tmp_path, **overrides):
    values = {
        "run_id": "run-1",
        "runs_root": str(tmp_path),
        "training_config": _training_config(),
        "inference_config": _inference_config(),
        "git_commit": "abc123",
        "image": "img:tag",
    }
    values.update(overrides)
    return execute_run(**values)


def test_execute_run_writes_completed_status_metrics_and_success_marker(
    tmp_path,
    mocker,
):
    mocker.patch(
        "instacart_rnn.training.exec_run.utc_now",
        side_effect=[
            "2026-01-01T00:00:00+00:00",
            "2026-01-01T01:00:00+00:00",
        ],
    )
    run_training = mocker.patch(
        "instacart_rnn.training.exec_run.run_training",
        return_value=TrainingRunResult(
            best_validation_loss=0.25,
            best_epoch=1,
            checkpoint_path="unused",
        ),
    )
    run_inference = mocker.patch("instacart_rnn.training.exec_run.run_inference")

    returned_run_id = _execute_run(tmp_path)

    run_root = tmp_path / "product" / "run-1"
    run_payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_root / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_root / "config.json").read_text(encoding="utf-8"))
    paths = run_training.call_args.kwargs["paths"]
    inference_config = run_inference.call_args.args[0]

    assert returned_run_id == "run-1"
    assert run_payload == {
        "completed_at": "2026-01-01T01:00:00+00:00",
        "git_commit": "abc123",
        "image": "img:tag",
        "model_name": "product",
        "run_id": "run-1",
        "started_at": "2026-01-01T00:00:00+00:00",
        "status": "completed",
    }
    assert metrics == {
        "best_epoch": 1,
        "best_validation_loss": 0.25,
    }
    assert config["model_name"] == "product"
    assert "on_validation_end" not in config
    assert (run_root / "_SUCCESS").exists()
    assert Path(paths.root) == run_root
    assert Path(inference_config.checkpoint_path) == (
        run_root / "checkpoints" / "best.pt"
    )
    assert Path(inference_config.output_path) == run_root / "artifacts"
    assert inference_config.eval_path == "eval.parquet"


def test_execute_run_marks_failed_and_skips_success_when_training_raises(
    tmp_path,
    mocker,
):
    mocker.patch(
        "instacart_rnn.training.exec_run.run_training",
        side_effect=RuntimeError("boom"),
    )
    run_inference = mocker.patch("instacart_rnn.training.exec_run.run_inference")

    with pytest.raises(RuntimeError, match="boom"):
        _execute_run(tmp_path)

    run_root = tmp_path / "product" / "run-1"
    run_payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))

    assert run_payload["status"] == "failed"
    assert run_payload["completed_at"] is not None
    assert run_payload["run_id"] == "run-1"
    assert not (run_root / "_SUCCESS").exists()
    assert not (run_root / "metrics.json").exists()
    run_inference.assert_not_called()


def test_execute_run_marks_failed_when_inference_raises(tmp_path, mocker):
    mocker.patch(
        "instacart_rnn.training.exec_run.run_training",
        return_value=TrainingRunResult(
            best_validation_loss=0.25,
            best_epoch=1,
            checkpoint_path="unused",
        ),
    )
    mocker.patch(
        "instacart_rnn.training.exec_run.run_inference",
        side_effect=RuntimeError("export failed"),
    )

    with pytest.raises(RuntimeError, match="export failed"):
        _execute_run(tmp_path)

    run_root = tmp_path / "product" / "run-1"
    run_payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))

    assert run_payload["status"] == "failed"
    assert not (run_root / "_SUCCESS").exists()
    assert not (run_root / "metrics.json").exists()
