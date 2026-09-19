from instacart_platform.backend import TrainingBackend
from instacart_platform.models import (
    JobStatus,
    TrainingJob,
    TrainingJobHandle,
    TrainingResult,
)


class _InMemoryBackend:
    def __init__(self):
        self.terminated: list[str] = []

    def submit(self, job: TrainingJob) -> TrainingJobHandle:
        return TrainingJobHandle(
            job_id=f"job-{job.run_id}",
            run_id=job.run_id,
        )

    def wait(self, handle: TrainingJobHandle) -> TrainingResult:
        return TrainingResult(
            run_id=handle.run_id,
            status=JobStatus.SUCCEEDED,
        )

    def terminate(self, handle: TrainingJobHandle) -> None:
        self.terminated.append(handle.job_id)


def test_in_memory_backend_follows_training_contract():
    backend = _InMemoryBackend()
    contract: TrainingBackend = backend
    job = TrainingJob(
        run_id="run-1",
        image="image:tag",
        command=["-m", "instacart_rnn.train"],
        gpu_type="rtx_4090",
    )

    handle = contract.submit(job)
    result = contract.wait(handle)
    contract.terminate(handle)

    assert handle.job_id == "job-run-1"
    assert handle.run_id == "run-1"
    assert result.status is JobStatus.SUCCEEDED
    assert result.run_id == "run-1"
    assert backend.terminated == ["job-run-1"]
