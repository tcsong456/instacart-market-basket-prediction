import json
import logging
import os
import time
from typing import Any

import gcsfs
import requests

from instacart_platform.models import (
    JobStatus,
    TrainingJob,
    TrainingJobHandle,
    TrainingResult,
)
from instacart_platform.runs import build_run_paths

logger = logging.getLogger(__name__)


RUNPOD_API_URL = "https://rest.runpod.io/v1"
# v2 GET includes lifecycle status and runtime; v1 GET does not.
RUNPOD_V2_API_URL = "https://api.runpod.io/v2"
_POD_DEAD_STATUSES = frozenset({"EXITED", "ERROR", "TERMINATED"})


class RunpodTrainingBackend:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        template_id: str,
        timeout_seconds: float = 30.0,
        poll_interval_seconds: float = 30.0,
        max_wait_seconds: float = 6 * 60 * 60,
        completion_grace_seconds: float = 120.0,
        startup_grace_seconds: float = 15 * 60,
    ) -> None:
        self._api_key = api_key or os.environ.get("RUNPOD_API_KEY")

        if not self._api_key:
            raise ValueError(
                "Runpod API key must be provided either through "
                "'api_key' or the RUNPOD_API_KEY environment variable"
            )

        if not template_id.strip():
            raise ValueError("template_id must not be empty")

        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")

        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be greater than 0")

        if max_wait_seconds <= 0:
            raise ValueError("max_wait_seconds must be greater than 0")

        if completion_grace_seconds <= 0:
            raise ValueError("completion_grace_seconds must be greater than 0")

        self._template_id = template_id
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._max_wait_seconds = max_wait_seconds
        self._gcs = gcsfs.GCSFileSystem()
        self._completion_grace_seconds = completion_grace_seconds
        self._startup_grace_seconds = startup_grace_seconds

        self._completed_seen_at: dict[str, float] = {}
        self._runtime_seen: set[str] = set()
        self._startup_seen_at: dict[str, float] = {}

    def submit(
        self,
        job: TrainingJob,
    ) -> TrainingJobHandle:
        gpu_types = self._ordered_gpu_types(job.gpu_type)
        failed_gpu_attempts: list[tuple[str, int, str]] = []

        for gpu_type in gpu_types:
            payload = self._build_create_pod_payload(
                job,
                gpu_type=gpu_type,
            )

            logger.info(
                "Attempting RunPod pod creation with GPU %s",
                gpu_type,
            )

            response = requests.post(
                f"{RUNPOD_API_URL}/pods",
                headers=self._headers(),
                json=payload,
                timeout=self._timeout_seconds,
            )

            if response.ok:
                response_data = response.json()
                pod_id = response_data.get("id")

                if not isinstance(pod_id, str) or not pod_id:
                    raise RuntimeError(
                        "RunPod create Pod response did not contain a valid Pod ID"
                    )

                logger.info(
                    "RunPod created pod %s with GPU %s",
                    pod_id,
                    gpu_type,
                )

                return TrainingJobHandle(
                    job_id=pod_id,
                    run_id=job.run_id,
                    runs_root=job.runs_root,
                    model_name=job.model_name,
                )

            if response.status_code == 500:
                failed_gpu_attempts.append(
                    (gpu_type, response.status_code, response.text)
                )
                logger.warning(
                    "RunPod pod creation returned HTTP 500 for GPU %s: %s",
                    gpu_type,
                    response.text,
                )
                continue

            logger.error(
                "RunPod pod creation failed for GPU %s: status=%s body=%s",
                gpu_type,
                response.status_code,
                response.text,
            )

            response.raise_for_status()

        failures = "; ".join(
            f"{gpu_type} (HTTP {status_code}): {body}"
            for gpu_type, status_code, body in failed_gpu_attempts
        )
        raise RuntimeError(
            f"RunPod pod creation failed for every configured GPU type: {failures}"
        )

    def get_status(
        self,
        handle: TrainingJobHandle,
    ) -> JobStatus:
        run_status = self._get_run_status(handle)

        if run_status == "completed":
            if self._success_exists(handle):
                self._completed_seen_at.pop(handle.run_id, None)
                return JobStatus.SUCCEEDED

            now = time.monotonic()

            completed_seen_at = self._completed_seen_at.setdefault(
                handle.run_id,
                now,
            )

            if now - completed_seen_at < self._completion_grace_seconds:
                return JobStatus.RUNNING

            return JobStatus.FAILED

        if run_status == "failed":
            return JobStatus.FAILED

        pod = self._get_pod(handle.job_id)
        runtime = pod.get("runtime")
        pod_status = pod.get("status")
        logger.info("RunPod pod response: %s", pod)

        if runtime is not None:
            self._runtime_seen.add(handle.job_id)

        if pod_status in _POD_DEAD_STATUSES:
            return JobStatus.FAILED

        # Application successfully reached running state.
        if run_status == "running":
            self._startup_seen_at.pop(handle.job_id, None)

            if handle.job_id in self._runtime_seen and runtime is None:
                return JobStatus.FAILED

            return JobStatus.RUNNING

        # No run.json yet, but we previously saw the container alive
        # and now it has disappeared.
        if handle.job_id in self._runtime_seen and runtime is None:
            return JobStatus.FAILED

        # Container is currently alive but application hasn't written
        # run.json yet.
        if runtime is not None:
            self._startup_seen_at.pop(handle.job_id, None)
            return JobStatus.PENDING

        # Container has not successfully started yet.
        now = time.monotonic()

        startup_seen_at = self._startup_seen_at.setdefault(
            handle.job_id,
            now,
        )

        if now - startup_seen_at < self._startup_grace_seconds:
            return JobStatus.PENDING

        return JobStatus.FAILED

    def wait(
        self,
        handle: TrainingJobHandle,
    ) -> TrainingResult:
        deadline = time.monotonic() + self._max_wait_seconds

        try:
            while True:
                status = self.get_status(handle)

                logger.info(
                    "Run %s: job=%s, status=%s",
                    handle.run_id,
                    handle.job_id,
                    status.value,
                )

                if status in {JobStatus.SUCCEEDED, JobStatus.FAILED}:
                    return TrainingResult(
                        run_id=handle.run_id,
                        status=status,
                    )

                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Training run {handle.run_id} did not "
                        f"complete within "
                        f"{self._max_wait_seconds} seconds"
                    )

                time.sleep(self._poll_interval_seconds)
        finally:
            self._completed_seen_at.pop(handle.run_id, None)

    def terminate(
        self,
        handle: TrainingJobHandle,
        *,
        max_attempts: int = 5,
        retry_delay_seconds: float = 15.0,
    ) -> None:
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = requests.delete(
                    f"{RUNPOD_API_URL}/pods/{handle.job_id}",
                    headers=self._headers(),
                    timeout=self._timeout_seconds,
                )

                if response.status_code == 404:
                    return

                response.raise_for_status()

                return

            except requests.RequestException as exc:
                last_error = exc

                logger.warning(
                    "Failed to terminate RunPod job %s (attempt %s/%s)",
                    handle.job_id,
                    attempt,
                    max_attempts,
                    exc_info=True,
                )

                if attempt < max_attempts:
                    time.sleep(retry_delay_seconds)

        raise RuntimeError(
            f"CRITICAL: unable to terminate RunPod job "
            f"{handle.job_id} after {max_attempts} attempts. "
            f"Pod may still be running and accumulating charges."
        ) from last_error

    def list_pods(self) -> list[dict]:
        pods: list[dict] = []
        cursor: str | None = None

        while True:
            params = {}
            if cursor is not None:
                params["cursor"] = cursor

            response = requests.get(
                f"{RUNPOD_V2_API_URL}/pods",
                headers=self._headers(),
                params=params,
                timeout=self._timeout_seconds,
            )

            if not response.ok:
                logger.error(
                    "Failed to list RunPod pods: status=%s body=%s",
                    response.status_code,
                    response.text,
                )

            response.raise_for_status()

            payload = response.json()

            if not isinstance(payload, dict):
                raise RuntimeError(f"Unexpected RunPod pod-list response: {payload!r}")

            page_pods = payload.get("pods")
            pagination = payload.get("pagination")

            if not isinstance(page_pods, list):
                raise RuntimeError(f"Unexpected RunPod pods response: {payload!r}")

            if not isinstance(pagination, dict):
                raise RuntimeError(
                    f"Unexpected RunPod pagination response: {payload!r}"
                )

            pods.extend(page_pods)

            if not pagination.get("hasNextPage", False):
                break

            cursor = pagination.get("nextCursor")

            if not isinstance(cursor, str) or not cursor:
                raise RuntimeError(
                    "RunPod reported hasNextPage=true without a valid nextCursor"
                )

        return pods

    def delete_pod(self, pod_id: str) -> None:
        response = requests.delete(
            f"{RUNPOD_API_URL}/pods/{pod_id}",
            headers=self._headers(),
            timeout=self._timeout_seconds,
        )

        if response.status_code == 404:
            logger.info(
                "RunPod pod %s no longer exists",
                pod_id,
            )
            return

        if not response.ok:
            logger.error(
                "Failed to delete RunPod pod %s: status=%s body=%s",
                pod_id,
                response.status_code,
                response.text,
            )

        response.raise_for_status()

        logger.info(
            "Deleted RunPod pod %s",
            pod_id,
        )

    def _get_pod(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        response = requests.get(
            f"{RUNPOD_V2_API_URL}/pods/{job_id}",
            headers=self._headers(),
            timeout=self._timeout_seconds,
        )
        if response.status_code == 404:
            return {"status": "TERMINATED", "runtime": None}

        response.raise_for_status()

        data = response.json()

        if not isinstance(data, dict):
            raise RuntimeError("Runpod get Pod response was not a JSON object")

        return data

    def _run_paths(self, handle: TrainingJobHandle):
        return build_run_paths(
            runs_root=handle.runs_root,
            model_name=handle.model_name,
            run_id=handle.run_id,
        )

    def _get_run_status(
        self,
        handle: TrainingJobHandle,
    ) -> str | None:
        path = self._run_paths(handle).run_json
        try:
            self._gcs.invalidate_cache(path)

            with self._gcs.open(path, "r") as file:
                payload = json.load(file)
        except FileNotFoundError:
            return None

        status = payload.get("status")

        if not isinstance(status, str):
            return None

        return status

    def _success_exists(
        self,
        handle: TrainingJobHandle,
    ) -> bool:
        path = self._run_paths(handle).success_marker

        self._gcs.invalidate_cache(path)

        return self._gcs.exists(path)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _ordered_gpu_types(preferred: str) -> tuple[str, ...]:
        SUPPORTED_GPU_TYPES = (
            "NVIDIA L4",
            "NVIDIA RTX A5000",
            "NVIDIA GeForce RTX 4090",
            "NVIDIA A40",
            "NVIDIA L40S",
        )

        if preferred not in SUPPORTED_GPU_TYPES:
            raise ValueError(
                f"Unsupported GPU type: {preferred!r}. "
                f"Supported GPU types: {SUPPORTED_GPU_TYPES}"
            )

        return (
            preferred,
            *(gpu for gpu in SUPPORTED_GPU_TYPES if gpu != preferred),
        )

    def _build_create_pod_payload(
        self, job: TrainingJob, gpu_type: str
    ) -> dict[str, Any]:
        return {
            "name": f"instacart-{job.model_name}-{job.run_id}",
            "imageName": job.image,
            "gpuTypeIds": [gpu_type],
            "gpuCount": job.gpu_count,
            "templateId": self._template_id,
            "dockerStartCmd": job.command,
        }
