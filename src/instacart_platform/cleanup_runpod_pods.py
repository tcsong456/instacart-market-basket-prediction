import argparse
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import gcsfs

from instacart_platform.models import CleanupReason, ManagedPod
from instacart_platform.runpod_backend import RunpodTrainingBackend

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--max-pod-age", type=float, default=6.5)
    parser.add_argument("--template-id", required=True)

    return parser.parse_args()


def cleanup_reason(
    *,
    run_status: str | None,
    pod_created_at: datetime,
    now: datetime,
    max_pod_age: timedelta,
) -> CleanupReason | None:
    if run_status in {"completed", "failed"}:
        return CleanupReason.RUN_FINISHED

    if now - pod_created_at > max_pod_age:
        return CleanupReason.POD_EXPIRED

    return None


def parse_managed_pod(pod_name: str) -> ManagedPod | None:
    prefix = "instacart-"

    if not pod_name.startswith(prefix):
        return None

    remainder = pod_name.removeprefix(prefix)

    try:
        model_name, run_id = remainder.split("-", maxsplit=1)
    except ValueError:
        return None

    if not model_name or not run_id:
        return None

    return ManagedPod(
        model_name=model_name,
        run_id=run_id,
    )


def read_run_status(
    *,
    runs_root: str,
    model_name: str,
    run_id: str,
) -> str | None:
    run_path = f"{runs_root.rstrip('/')}/{model_name}/{run_id}/run.json"

    fs = gcsfs.GCSFileSystem()

    try:
        with fs.open(run_path, "r") as f:
            payload = json.load(f)
    except FileNotFoundError:
        return None

    status = payload.get("status")

    if not isinstance(status, str):
        raise RuntimeError(
            f"Invalid run.json for run {run_id!r}: missing or invalid 'status'"
        )

    if status not in ["running", "completed", "failed"]:
        raise RuntimeError(
            f"Received status: {status} is invalid, "
            "only 'running', 'completed', 'failed' are valid states"
        )

    return status


def parse_created_at(pod: dict[str, Any]) -> datetime:
    value = pod.get("createdAt")

    if not isinstance(value, str) or not value:
        raise RuntimeError(
            f"RunPod pod {pod.get('id')!r} has invalid createdAt: {value!r}"
        )

    try:
        created_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(
            f"RunPod pod {pod.get('id')!r} has invalid createdAt: {value!r}"
        ) from exc

    if created_at.tzinfo is None:
        raise RuntimeError(
            f"RunPod pod {pod.get('id')!r} createdAt has no timezone: {value!r}"
        )

    return created_at.astimezone(timezone.utc)


def cleanup_runpod_pods(
    *,
    backend: RunpodTrainingBackend,
    runs_root: str,
    max_pod_age: float,
) -> None:
    if max_pod_age <= 0:
        raise RuntimeError(
            f"max_pod_age must be a positive number, but received {max_pod_age}"
        )

    now = datetime.now(timezone.utc)

    max_pod_age = timedelta(hours=max_pod_age)

    for pod in backend.list_pods():
        pod_id = pod["id"]
        pod_name = pod["name"]

        managed_pod = parse_managed_pod(pod_name)

        if managed_pod is None:
            logger.debug(
                "Ignoring unmanaged RunPod pod %s (%s)",
                pod_id,
                pod_name,
            )
            continue

        run_status = read_run_status(
            runs_root=runs_root,
            model_name=managed_pod.model_name,
            run_id=managed_pod.run_id,
        )

        reason = cleanup_reason(
            run_status=run_status,
            pod_created_at=parse_created_at(pod),
            now=now,
            max_pod_age=max_pod_age,
        )

        if reason is None:
            logger.info(
                "Keeping RunPod pod %s: run=%s status=%s",
                pod_id,
                managed_pod.run_id,
                run_status,
            )
            continue

        logger.warning(
            "Cleaning up RunPod pod %s: run=%s reason=%s",
            pod_id,
            managed_pod.run_id,
            reason.value,
        )

        backend.delete_pod(pod_id)


if __name__ == "__main__":
    args = parse_args()
    backend = RunpodTrainingBackend(template_id=args.template_id)
    cleanup_runpod_pods(
        backend=backend, runs_root=args.runs_root, max_pod_age=args.max_pod_age
    )
