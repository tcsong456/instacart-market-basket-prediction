import re

import pytest

from instacart_platform.models import (
    JobStatus,
    TrainingJobHandle,
    TrainingResult,
)
from instacart_platform.rnn_submit_training_job import generate_run_id, main

REQUIRED_ARGS = [
    "submit_training_job.py",
    "--image",
    "img:tag",
    "--template-id",
    "tpl-1",
    "--git-commit",
    "deadbeef",
    "--timeline",
    "t0",
    "--gold-root",
    "gs://gold",
    "--runs-root",
    "gs://runs",
    "--model-name",
    "product",
]


def _command_value(command, flag):
    return command[command.index(flag) + 1]


def test_generate_run_id_uses_utc_timestamp_and_hex_suffix():
    run_id = generate_run_id()

    assert re.fullmatch(r"\d{8}T\d{6}Z_[0-9a-f]{6}", run_id)


def test_main_builds_gold_data_root_and_run_command(mocker, monkeypatch):
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
    mocker.patch(
        "instacart_platform.rnn_submit_training_job.generate_run_id",
        return_value="run-1",
    )
    backend_cls = mocker.patch(
        "instacart_platform.rnn_submit_training_job.RunpodTrainingBackend",
    )
    backend_cls.return_value.submit.return_value = TrainingJobHandle(
        job_id="pod-1",
        run_id="run-1",
        runs_root="gs://runs",
        model_name="product",
    )
    backend_cls.return_value.wait.return_value = TrainingResult(
        run_id="run-1",
        status=JobStatus.SUCCEEDED,
    )

    main()

    backend_cls.assert_called_once_with(template_id="tpl-1")
    job = backend_cls.return_value.submit.call_args.args[0]
    command = list(job.command)

    assert job.run_id == "run-1"
    assert job.image == "img:tag"
    assert job.runs_root == "gs://runs"
    assert job.model_name == "product"
    assert job.gpu_type == "NVIDIA L4"
    assert job.gpu_count == 1
    assert command[:2] == ["-m", "instacart_rnn.run"]
    assert _command_value(command, "--run-id") == "run-1"
    assert _command_value(command, "--runs-root") == "gs://runs"
    assert _command_value(command, "--data-root") == (
        "gs://gold/training/base_train/t0"
    )
    assert _command_value(command, "--model") == "product"
    assert _command_value(command, "--git-commit") == "deadbeef"
    assert _command_value(command, "--image") == "img:tag"
    assert "--amp" in command
    assert "--pin-memory" in command
    backend_cls.return_value.terminate.assert_called_once()


def test_main_raises_when_wait_returns_failed_status(mocker, monkeypatch):
    monkeypatch.setattr("sys.argv", REQUIRED_ARGS)
    mocker.patch(
        "instacart_platform.rnn_submit_training_job.generate_run_id",
        return_value="run-1",
    )
    backend_cls = mocker.patch(
        "instacart_platform.rnn_submit_training_job.RunpodTrainingBackend",
    )
    backend_cls.return_value.submit.return_value = TrainingJobHandle(
        job_id="pod-1",
        run_id="run-1",
        runs_root="gs://runs",
        model_name="product",
    )
    backend_cls.return_value.wait.return_value = TrainingResult(
        run_id="run-1",
        status=JobStatus.FAILED,
    )

    with pytest.raises(RuntimeError, match="failed with status failed"):
        main()

    backend_cls.return_value.terminate.assert_called_once()


def test_main_raises_when_wait_times_out_and_still_terminates(
    mocker,
    monkeypatch,
):
    monkeypatch.setattr("sys.argv", REQUIRED_ARGS)
    mocker.patch(
        "instacart_platform.rnn_submit_training_job.generate_run_id",
        return_value="run-1",
    )
    backend_cls = mocker.patch(
        "instacart_platform.rnn_submit_training_job.RunpodTrainingBackend",
    )
    backend_cls.return_value.submit.return_value = TrainingJobHandle(
        job_id="pod-1",
        run_id="run-1",
        runs_root="gs://runs",
        model_name="product",
    )
    backend_cls.return_value.wait.side_effect = TimeoutError("timed out")

    with pytest.raises(TimeoutError, match="timed out"):
        main()

    backend_cls.return_value.terminate.assert_called_once()


def test_main_prefers_training_error_when_terminate_also_fails(
    mocker,
    monkeypatch,
):
    monkeypatch.setattr("sys.argv", REQUIRED_ARGS)
    mocker.patch(
        "instacart_platform.rnn_submit_training_job.generate_run_id",
        return_value="run-1",
    )
    backend_cls = mocker.patch(
        "instacart_platform.rnn_submit_training_job.RunpodTrainingBackend",
    )
    backend_cls.return_value.submit.return_value = TrainingJobHandle(
        job_id="pod-1",
        run_id="run-1",
        runs_root="gs://runs",
        model_name="product",
    )
    backend_cls.return_value.wait.return_value = TrainingResult(
        run_id="run-1",
        status=JobStatus.FAILED,
    )
    backend_cls.return_value.terminate.side_effect = RuntimeError(
        "unable to terminate",
    )

    with pytest.raises(RuntimeError, match="failed with status failed"):
        main()


def test_main_raises_when_terminate_fails_after_success(mocker, monkeypatch):
    monkeypatch.setattr("sys.argv", REQUIRED_ARGS)
    mocker.patch(
        "instacart_platform.rnn_submit_training_job.generate_run_id",
        return_value="run-1",
    )
    backend_cls = mocker.patch(
        "instacart_platform.rnn_submit_training_job.RunpodTrainingBackend",
    )
    backend_cls.return_value.submit.return_value = TrainingJobHandle(
        job_id="pod-1",
        run_id="run-1",
        runs_root="gs://runs",
        model_name="product",
    )
    backend_cls.return_value.wait.return_value = TrainingResult(
        run_id="run-1",
        status=JobStatus.SUCCEEDED,
    )
    backend_cls.return_value.terminate.side_effect = RuntimeError(
        "unable to terminate",
    )

    with pytest.raises(RuntimeError, match="unable to terminate"):
        main()
