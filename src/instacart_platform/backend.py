from typing import Protocol

from instacart_platform.models import (
    TrainingJob,
    TrainingJobHandle,
    TrainingResult,
)


class TrainingBackend(Protocol):
    def submit(self, job: TrainingJob) -> TrainingJobHandle: ...

    def wait(self, handle: TrainingJobHandle) -> TrainingResult: ...

    def terminate(self, handle: TrainingJobHandle) -> None: ...
