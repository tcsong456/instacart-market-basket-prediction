from dataclasses import dataclass
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CleanupReason(str, Enum):
    RUN_FINISHED = "run_finished"
    POD_EXPIRED = "pod_expired"


@dataclass(frozen=True)
class ManagedPod:
    model_name: str
    run_id: str


@dataclass(frozen=True)
class TrainingJob:
    run_id: str
    image: str
    command: tuple
    gpu_type: str
    runs_root: str
    model_name: str
    gpu_count: int = 1


@dataclass(frozen=True)
class TrainingJobHandle:
    job_id: str
    run_id: str
    runs_root: str
    model_name: str


@dataclass(frozen=True)
class TrainingResult:
    run_id: str
    status: JobStatus
