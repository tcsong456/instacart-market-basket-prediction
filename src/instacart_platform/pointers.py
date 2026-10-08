import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone

import gcsfs

from instacart_etl_rnn.common.paths import is_gcs_url, join_path
from instacart_platform.runs import (
    build_run_paths,
    success_marker_exists,
    write_json,
)
from instacart_rnn.stacking.feature_selection import (
    REQUIRED_MODELS,
    load_selected_artifacts,
)
from instacart_rnn.stacking.folds import stack_output_dir

POINTER_ROLES = ("champion", "challenger")


@dataclass(frozen=True)
class ResolvedModel:
    """One base run named by a pointer."""

    model_name: str
    run_id: str
    artifact_path: str
    best_checkpoint: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class ResolvedPointer:
    """A stack and the four base runs a pointer names."""

    role: str
    mode: str
    timeline: str
    stack_id: str
    stack_dir: str
    model_path: str
    previous_stack_id: str | None
    promoted_at: str
    model_sha256: str
    models: dict[str, ResolvedModel]


def pointer_path(
    runs_root: str,
    mode: str,
    timeline: str,
    role: str,
) -> str:
    """Return the JSON path for one role, mode, and timeline.

    Args:
        runs_root: Root that contains ``pointers`` and ``stack``.
        mode: Gold dataset mode, such as ``sample`` or ``curated``.
        timeline: Gold timeline, such as ``t1``.
        role: ``champion`` or ``challenger``.

    Returns:
        ``{runs_root}/pointers/{mode}/{timeline}/{role}.json``.
    """

    _require_role(role)
    _require_segment(mode, "mode")
    _require_segment(timeline, "timeline")
    return str(
        join_path(
            join_path(
                join_path(join_path(runs_root.rstrip("/"), "pointers"), mode),
                timeline,
            ),
            f"{role}.json",
        )
    )


def resolve_pointer(
    *,
    runs_root: str,
    mode: str,
    timeline: str,
    role: str,
) -> ResolvedPointer:
    """Return the stack and base runs named by one pointer.

    A missing pointer raises ``FileNotFoundError``. The latest completed run
    is never substituted for the run id stored in the pointer.

    Args:
        runs_root: Root that contains ``pointers`` and ``stack``.
        mode: Gold dataset mode requested by the caller.
        timeline: Gold timeline requested by the caller.
        role: ``champion`` or ``challenger``.

    Returns:
        The resolved pointer, including each model's artifact directory and
        ``best.pt`` path.
    """

    path = pointer_path(runs_root, mode, timeline, role)
    payload = _read_json(path)
    return _resolve_payload(
        runs_root=runs_root,
        path=path,
        payload=payload,
        mode=mode,
        timeline=timeline,
        role=role,
    )


def publish_pointer(
    *,
    runs_root: str,
    stack_dir: str,
    role: str,
) -> ResolvedPointer:
    """Record an existing stack as ``champion`` or ``challenger``.

    The pointer copies ids from ``selected_artifacts.json`` and the model
    path from ``stacking_train.json``. It records the SHA-256 of ``model.txt``
    and of each ``best.pt``. When a pointer already exists, its stack id is
    stored as ``previous_stack_id``. Metrics are not compared.

    Args:
        runs_root: Root that contains ``pointers`` and the completed runs.
        stack_dir: Directory that holds the stack manifest and ``model.txt``.
        role: ``champion`` or ``challenger``.

    Returns:
        The pointer after it has been read back from the written file.
    """

    _require_role(role)
    record = _load_stack_record(stack_dir)
    _require_completed_models(runs_root, record.run_ids)
    path = pointer_path(runs_root, record.mode, record.timeline, role)
    payload = {
        "role": role,
        "mode": record.mode,
        "timeline": record.timeline,
        "stack_id": record.stack_id,
        "model_path": record.model_path,
        "model_sha256": _file_sha256(record.model_path),
        "previous_stack_id": _previous_stack_id(path),
        "promoted_at": _utc_now(),
        "models": {
            name: {
                "run_id": run_id,
                "checkpoint_sha256": _checkpoint_sha256(runs_root, name, run_id),
            }
            for name, run_id in record.run_ids.items()
        },
    }
    write_json(path, payload)
    return resolve_pointer(
        runs_root=runs_root,
        mode=record.mode,
        timeline=record.timeline,
        role=role,
    )


def _resolve_payload(
    *,
    runs_root: str,
    path: str,
    payload: dict,
    mode: str,
    timeline: str,
    role: str,
) -> ResolvedPointer:
    _require_equal(payload.get("role"), role, f"Pointer {path} role")
    _require_equal(payload.get("mode"), mode, f"Pointer {path} mode")
    _require_equal(payload.get("timeline"), timeline, f"Pointer {path} timeline")
    stack_id = _require_segment(payload.get("stack_id"), "stack_id")
    promoted_at = payload.get("promoted_at")
    if not isinstance(promoted_at, str) or not promoted_at.strip():
        raise ValueError(f"Pointer {path} has no promoted_at")
    previous_stack_id = _optional_stack_id(payload.get("previous_stack_id"), path)
    pointer_runs = _pointer_run_ids(payload.get("models"), path)
    stack_dir = stack_output_dir(runs_root, stack_id)
    model_path = str(join_path(stack_dir, "model.txt"))
    _require_equal(
        payload.get("model_path"),
        model_path,
        f"Pointer {path} model_path",
    )
    _require_file(model_path)
    manifest_path = str(join_path(stack_dir, "selected_artifacts.json"))
    _require_file(manifest_path)
    manifest = load_selected_artifacts(manifest_path)
    _require_equal(
        manifest.get("stack_id"),
        stack_id,
        f"Stack {stack_dir} selected_artifacts.json stack_id",
    )
    _require_equal(
        manifest.get("mode"),
        mode,
        f"Stack {stack_dir} selected_artifacts.json mode",
    )
    _require_equal(
        manifest.get("timeline"),
        timeline,
        f"Stack {stack_dir} selected_artifacts.json timeline",
    )
    models = _resolve_models(
        runs_root=runs_root,
        path=path,
        manifest_models=manifest["models"],
        pointer_models=payload["models"],
        pointer_runs=pointer_runs,
    )
    model_sha256 = _file_sha256(model_path)
    _require_fingerprint(
        payload.get("model_sha256"),
        model_sha256,
        f"Pointer {path} model_sha256",
    )
    return ResolvedPointer(
        role=role,
        mode=mode,
        timeline=timeline,
        stack_id=stack_id,
        stack_dir=stack_dir,
        model_path=model_path,
        previous_stack_id=previous_stack_id,
        promoted_at=promoted_at,
        model_sha256=model_sha256,
        models=models,
    )


@dataclass(frozen=True)
class _StackRecord:
    stack_id: str
    mode: str
    timeline: str
    model_path: str
    run_ids: dict[str, str]


def _load_stack_record(stack_dir: str) -> _StackRecord:
    manifest_path = str(join_path(stack_dir, "selected_artifacts.json"))
    manifest = load_selected_artifacts(manifest_path)
    stack_id = _require_segment(manifest.get("stack_id"), "stack_id")
    if _directory_name(stack_dir) != stack_id:
        raise ValueError(
            f"Stack directory {stack_dir} does not end with stack_id {stack_id}"
        )
    mode = _require_segment(manifest.get("mode"), "mode")
    timeline = _require_segment(manifest.get("timeline"), "timeline")
    training_path = str(join_path(stack_dir, "stacking_train.json"))
    training = _read_json(training_path)
    model_path = str(join_path(stack_dir, "model.txt"))
    _require_equal(
        training.get("model_path"),
        model_path,
        "stacking_train.json model_path",
    )
    _require_file(model_path)
    run_ids = {}
    for name in REQUIRED_MODELS:
        run_id = manifest["models"][name].get("run_id")
        run_ids[name] = _require_segment(run_id, f"{name} run_id")
    return _StackRecord(
        stack_id=stack_id,
        mode=mode,
        timeline=timeline,
        model_path=model_path,
        run_ids=run_ids,
    )


def _resolve_models(
    *,
    runs_root: str,
    path: str,
    manifest_models: dict,
    pointer_models: dict,
    pointer_runs: dict[str, str],
) -> dict[str, ResolvedModel]:
    resolved = {}
    for name in REQUIRED_MODELS:
        manifest_run = manifest_models[name].get("run_id")
        pointer_run = pointer_runs[name]
        _require_equal(
            manifest_run,
            pointer_run,
            f"Pointer {path} run_id for {name}",
        )
        run_paths = _require_completed_run(runs_root, name, pointer_run)
        artifact_path = str(join_path(run_paths.artifacts, "stacking_train"))
        _require_equal(
            manifest_models[name].get("artifact_path"),
            artifact_path,
            f"Pointer {path} artifact_path for {name}",
        )
        _require_file(artifact_path)
        best_checkpoint = str(run_paths.best_checkpoint)
        checkpoint_sha256 = _file_sha256(best_checkpoint)
        entry = pointer_models[name]
        _require_fingerprint(
            entry.get("checkpoint_sha256") if isinstance(entry, dict) else None,
            checkpoint_sha256,
            f"Pointer {path} checkpoint_sha256 for {name}",
        )
        resolved[name] = ResolvedModel(
            model_name=name,
            run_id=pointer_run,
            artifact_path=artifact_path,
            best_checkpoint=best_checkpoint,
            checkpoint_sha256=checkpoint_sha256,
        )
    return resolved


def _require_completed_models(runs_root: str, run_ids: dict[str, str]) -> None:
    for name, run_id in run_ids.items():
        _require_completed_run(runs_root, name, run_id)


def _require_completed_run(runs_root: str, model_name: str, run_id: str):
    paths = build_run_paths(
        runs_root=runs_root,
        model_name=model_name,
        run_id=run_id,
    )
    if not success_marker_exists(str(paths.success_marker)):
        raise FileNotFoundError(f"Run {model_name}/{run_id} has no _SUCCESS")
    payload = _read_json(str(paths.run_json))
    if payload.get("status") != "completed":
        raise ValueError(
            f"Run {model_name}/{run_id} status is {payload.get('status')!r}"
        )
    return paths


def _pointer_run_ids(models: object, path: str) -> dict[str, str]:
    if not isinstance(models, dict):
        raise ValueError(f"Pointer {path} has no models")
    missing = [name for name in REQUIRED_MODELS if name not in models]
    if missing:
        raise ValueError(f"Pointer {path} is missing {', '.join(missing)}")
    run_ids = {}
    for name in REQUIRED_MODELS:
        entry = models[name]
        if not isinstance(entry, dict):
            raise ValueError(f"Pointer {path} model {name} is not an object")
        run_ids[name] = _require_segment(entry.get("run_id"), f"{name} run_id")
    return run_ids


def _previous_stack_id(path: str) -> str | None:
    if not success_marker_exists(path):
        return None
    payload = _read_json(path)
    stack_id = payload.get("stack_id")
    if not isinstance(stack_id, str) or not stack_id.strip():
        raise ValueError(f"Pointer {path} has no stack_id")
    return stack_id


def _optional_stack_id(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or "/" in value:
        raise ValueError(f"Pointer {path} previous_stack_id is invalid")
    return value


def _checkpoint_sha256(runs_root: str, model_name: str, run_id: str) -> str:
    checkpoint = build_run_paths(
        runs_root=runs_root,
        model_name=model_name,
        run_id=run_id,
    ).best_checkpoint
    return _file_sha256(str(checkpoint))


def _require_fingerprint(recorded: object, actual: str, label: str) -> None:
    if recorded != actual:
        raise ValueError(f"{label} is {recorded!r}, file sha256 is {actual}")


def _file_sha256(path: str) -> str:
    """Return the SHA-256 hex digest of a local file or ``gs://`` object."""

    if not success_marker_exists(path):
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    if is_gcs_url(path):
        filesystem = gcsfs.GCSFileSystem()
        filesystem.invalidate_cache(path)
        with filesystem.open(path, "rb") as file:
            _update_digest(file, digest)
    else:
        with open(path, "rb") as file:
            _update_digest(file, digest)
    return digest.hexdigest()


def _update_digest(file, digest) -> None:
    while True:
        chunk = file.read(1024 * 1024)
        if not chunk:
            return
        digest.update(chunk)


def _require_file(path: str) -> None:
    if not success_marker_exists(path):
        raise FileNotFoundError(path)


def _require_equal(actual: object, expected: str, label: str) -> None:
    if not _same_path(actual, expected):
        raise ValueError(f"{label} is {actual!r}, expected {expected}")


def _same_path(actual: object, expected: str) -> bool:
    if not isinstance(actual, str):
        return False
    return _normalize(actual) == _normalize(expected)


def _normalize(path: str) -> str:
    return path.rstrip("/\\").replace("\\", "/")


def _directory_name(path: str) -> str:
    return _normalize(path).rsplit("/", 1)[-1]


def _require_role(role: str) -> None:
    if role not in POINTER_ROLES:
        raise ValueError(f"role must be one of {POINTER_ROLES}, received {role!r}")


def _require_segment(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "/" in value
        or "\\" in value
        or value.startswith("gs:")
    ):
        raise ValueError(f"{name} must be a single path segment, received {value!r}")
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: str) -> dict:
    if not success_marker_exists(path):
        raise FileNotFoundError(path)
    if is_gcs_url(path):
        filesystem = gcsfs.GCSFileSystem()
        filesystem.invalidate_cache(path)
        with filesystem.open(path, "r") as file:
            payload = json.load(file)
    else:
        with open(path, encoding="utf-8") as file:
            payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload
