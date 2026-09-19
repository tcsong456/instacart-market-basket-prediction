import os
from typing import Any

import requests

from instacart_platform.models import (
    TrainingJob,
    TrainingJobHandle,
)

RUNPOD_API_URL = "https://rest.runpod.io/v1"


class RunpodTrainingBackend:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        template_id: str,
        timeout_seconds: float = 30.0,
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

        self._template_id = template_id
        self._timeout_seconds = timeout_seconds

    def submit(self, job: TrainingJob) -> TrainingJobHandle:
        payload = self._build_create_pod_payload(job)

        response = requests.post(
            f"{RUNPOD_API_URL}/pods",
            headers=self._headers(),
            json=payload,
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()

        response_data = response.json()

        pod_id = response_data.get("id")
        if not isinstance(pod_id, str) or not pod_id:
            raise RuntimeError(
                "Runpod create Pod response did not contain a valid Pod ID"
            )

        return TrainingJobHandle(
            job_id=pod_id,
            run_id=job.run_id,
        )

    def terminate(self, handle: TrainingJobHandle) -> None:
        response = requests.delete(
            f"{RUNPOD_API_URL}/pods/{handle.job_id}",
            headers=self._headers(),
            timeout=self._timeout_seconds,
        )
        response.raise_for_status()

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _build_create_pod_payload(
        self,
        job: TrainingJob,
    ) -> dict[str, Any]:
        return {
            "name": f"instacart-{job.run_id}",
            "imageName": job.image,
            "gpuTypeIds": [job.gpu_type],
            "gpuCount": job.gpu_count,
            "templateId": self._template_id,
            "dockerStartCmd": job.command,
        }
