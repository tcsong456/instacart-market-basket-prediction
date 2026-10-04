import pytest

from instacart_platform.models import JobStatus, TrainingJobHandle, TrainingResult
from instacart_platform.stacking_submit_training_job import (
    STACKING_CPU_FLAVORS,
    STACKING_MODEL_NAME,
    STACKING_VCPU_COUNT,
    main,
    stack_directory,
    training_selection_args,
)

REQUIRED_ARGS = [
    "stacking_submit_training_job.py",
    "--image",
    "img:tag",
    "--template-id",
    "tpl-1",
    "--git-commit",
    "deadbeef",
    "--mode",
    "curated",
    "--timeline",
    "initial",
    "--runs-root",
    "gs://runs",
]


def _command_value(command, flag):
    return command[command.index(flag) + 1]


def test_stack_directory_appends_the_stack_id():
    assert stack_directory("gs://runs/", "stack-1") == "gs://runs/stack/stack-1"


def test_stack_directory_rejects_a_full_uri():
    with pytest.raises(ValueError, match="directory name"):
        stack_directory("gs://runs", "gs://runs/stack/stack-1")


def test_main_submits_the_stacking_train_module(mocker, monkeypatch):
    monkeypatch.setattr("sys.argv", REQUIRED_ARGS)
    mocker.patch(
        "instacart_platform.stacking_submit_training_job.generate_run_id",
        return_value="run-1",
    )
    backend_cls = mocker.patch(
        "instacart_platform.stacking_submit_training_job.RunpodTrainingBackend",
    )
    backend_cls.return_value.submit.return_value = TrainingJobHandle(
        job_id="pod-1",
        run_id="run-1",
        runs_root="gs://runs",
        model_name=STACKING_MODEL_NAME,
    )
    backend_cls.return_value.wait.return_value = TrainingResult(
        run_id="run-1",
        status=JobStatus.SUCCEEDED,
    )

    main()

    job = backend_cls.return_value.submit.call_args.args[0]
    command = list(job.command)

    assert job.model_name == STACKING_MODEL_NAME
    assert job.compute_type == "CPU"
    assert job.gpu_count == 0
    assert job.cpu_flavor_ids == STACKING_CPU_FLAVORS
    assert job.vcpu_count == STACKING_VCPU_COUNT
    assert command[:2] == ["-m", "instacart_rnn.stacking.stacking_train"]
    assert _command_value(command, "--mode") == "curated"
    assert _command_value(command, "--timeline") == "initial"
    assert _command_value(command, "--n-folds") == "5"
    assert "--stack-dir" not in command
    assert _command_value(command, "--run-id") == "run-1"
    assert _command_value(command, "--runs-root") == "gs://runs"
    assert _command_value(command, "--model") == STACKING_MODEL_NAME
    assert _command_value(command, "--git-commit") == "deadbeef"
    assert _command_value(command, "--image") == "img:tag"
    backend_cls.return_value.terminate.assert_called_once()


def test_training_selection_args_reuses_a_stack_id():
    args = type(
        "Args",
        (),
        {
            "stack_id": "stack-1",
            "mode": None,
            "timeline": None,
            "runs_root": "gs://runs",
            "n_folds": 5,
        },
    )()

    assert training_selection_args(args) == (
        "--stack-dir",
        "gs://runs/stack/stack-1",
    )


def test_training_selection_args_rejects_both_sources():
    args = type(
        "Args",
        (),
        {
            "stack_id": "stack-1",
            "mode": "curated",
            "timeline": None,
            "runs_root": "gs://runs",
            "n_folds": 5,
        },
    )()

    with pytest.raises(ValueError, match="either --stack-id"):
        training_selection_args(args)


def test_main_raises_when_wait_returns_failed_status(mocker, monkeypatch):
    monkeypatch.setattr("sys.argv", REQUIRED_ARGS)
    mocker.patch(
        "instacart_platform.stacking_submit_training_job.generate_run_id",
        return_value="run-1",
    )
    backend_cls = mocker.patch(
        "instacart_platform.stacking_submit_training_job.RunpodTrainingBackend",
    )
    backend_cls.return_value.submit.return_value = TrainingJobHandle(
        job_id="pod-1",
        run_id="run-1",
        runs_root="gs://runs",
        model_name=STACKING_MODEL_NAME,
    )
    backend_cls.return_value.wait.return_value = TrainingResult(
        run_id="run-1",
        status=JobStatus.FAILED,
    )

    with pytest.raises(RuntimeError, match="failed with status failed"):
        main()

    backend_cls.return_value.terminate.assert_called_once()
