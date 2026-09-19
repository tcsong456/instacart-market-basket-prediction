from dataclasses import FrozenInstanceError

import pytest

from instacart_platform.models import (
    JobStatus,
    TrainingJob,
    TrainingJobHandle,
    TrainingResult,
)


def _training_job(**overrides) -> TrainingJob:
    payload = {
        "run_id": "run-1",
        "image": "image:tag",
        "command": ["-m", "instacart_rnn.train"],
        "gpu_type": "rtx_4090",
    }
    payload.update(overrides)
    return TrainingJob(**payload)


def test_training_job_uses_default_gpu_count():
    job = _training_job()

    assert job.run_id == "run-1"
    assert job.image == "image:tag"
    assert job.command == ["-m", "instacart_rnn.train"]
    assert job.gpu_type == "rtx_4090"
    assert job.gpu_count == 1


def test_training_job_keeps_explicit_gpu_count():
    job = _training_job(gpu_count=2)

    assert job.gpu_count == 2


def test_training_job_is_frozen():
    job = _training_job()

    with pytest.raises(FrozenInstanceError):
        job.run_id = "run-2"


def test_job_status_values():
    assert JobStatus.PENDING == "pending"
    assert JobStatus.RUNNING == "running"
    assert JobStatus.SUCCEEDED == "succeeded"
    assert JobStatus.FAILED == "failed"


def test_training_result_records_status():
    handle = TrainingJobHandle(job_id="job-1", run_id="run-1")
    result = TrainingResult(run_id=handle.run_id, status=JobStatus.SUCCEEDED)

    assert handle.job_id == "job-1"
    assert result.run_id == "run-1"
    assert result.status is JobStatus.SUCCEEDED


def test_training_job_handle_is_frozen():
    handle = TrainingJobHandle(job_id="job-1", run_id="run-1")

    with pytest.raises(FrozenInstanceError):
        handle.job_id = "job-2"
