import hashlib
import json
import logging
import os
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

logger = logging.getLogger(__name__)

POINTER_ROLES = ("champion", "challenger")


class PointerConflictError(RuntimeError):
    """The champion pointer changed during the compare-and-publish section."""


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
    role: str,
    timeline: str | None = None,
) -> str:
    """Return the JSON path for one pointer.

    The champion is one pointer for the mode. A challenger stays on the
    timeline whose stack was just trained.

    Args:
        runs_root: Root that contains ``pointers`` and ``stack``.
        mode: Gold dataset mode, such as ``sample`` or ``curated``.
        role: ``champion`` or ``challenger``.
        timeline: Gold timeline, such as ``t1``. Required for a challenger.

    Returns:
        ``{runs_root}/pointers/{mode}/champion.json``, or
        ``{runs_root}/pointers/{mode}/{timeline}/challenger.json``.
    """

    _require_role(role)
    _require_segment(mode, "mode")
    mode_dir = join_path(join_path(runs_root.rstrip("/"), "pointers"), mode)
    if role == "champion":
        return str(join_path(mode_dir, "champion.json"))
    if timeline is None:
        raise ValueError("timeline is required for a challenger pointer")
    _require_segment(timeline, "timeline")
    return str(join_path(join_path(mode_dir, timeline), f"{role}.json"))


def resolve_pointer(
    *,
    runs_root: str,
    mode: str,
    role: str,
    timeline: str | None = None,
) -> ResolvedPointer:
    """Return the stack and base runs named by one pointer.

    A missing pointer raises ``FileNotFoundError``. The latest completed run
    is never substituted for the run id stored in the pointer. The champion
    lookup is the mode pointer; its timeline is the timeline that stack was
    trained on.

    Args:
        runs_root: Root that contains ``pointers`` and ``stack``.
        mode: Gold dataset mode requested by the caller.
        role: ``champion`` or ``challenger``.
        timeline: Gold timeline requested for a challenger.

    Returns:
        The resolved pointer, including each model's artifact directory and
        ``best.pt`` path.
    """

    path = pointer_path(runs_root, mode, role, timeline)
    payload = _read_json(path)
    if role == "champion":
        timeline = _require_segment(payload.get("timeline"), "timeline")
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
    path = pointer_path(runs_root, record.mode, role, record.timeline)
    previous = _existing_pointer(path)
    payload = _pointer_payload(
        record,
        role=role,
        runs_root=runs_root,
        previous_stack_id=None if previous is None else previous["stack_id"],
        previous_model_sha256=None if previous is None else previous["model_sha256"],
        previous_models=None if previous is None else previous["models"],
    )
    write_json(path, payload)
    return resolve_pointer(
        runs_root=runs_root,
        mode=record.mode,
        role=role,
        timeline=None if role == "champion" else record.timeline,
    )


def pointer_generation(path: str) -> int | None:
    """Return the current pointer generation.

    Args:
        path: Pointer JSON path returned by :func:`pointer_path`.

    Returns:
        ``None`` when the file is absent. Otherwise the GCS object generation
        or the local modification token.
    """

    if is_gcs_url(path):
        return _gcs_generation(path)
    if not os.path.exists(path):
        return None
    return os.stat(path).st_mtime_ns


def publish_champion_if_current(
    *,
    runs_root: str,
    stack_dir: str,
    expected_generation: int | None,
    previous_stack_id: str | None,
) -> ResolvedPointer:
    """Publish ``champion.json`` only when its generation is still current.

    ``expected_generation=None`` creates the file only if it is still absent.
    A changed or newly created pointer raises :class:`PointerConflictError`.

    Args:
        runs_root: Root that contains ``pointers`` and the completed runs.
        stack_dir: Directory that holds the stack manifest and ``model.txt``.
        expected_generation: Generation observed before the comparison.
        previous_stack_id: Champion stack id at that generation, if one existed.

    Returns:
        The champion pointer after it has been read back.
    """

    record = _load_stack_record(stack_dir)
    _require_completed_models(runs_root, record.run_ids)
    path = pointer_path(runs_root, record.mode, "champion")
    previous = _existing_pointer(path) if previous_stack_id is not None else None
    if previous is not None:
        _require_equal(
            previous.get("stack_id"),
            previous_stack_id,
            f"Pointer {path} stack_id",
        )
    payload = _pointer_payload(
        record,
        role="champion",
        runs_root=runs_root,
        previous_stack_id=previous_stack_id,
        previous_model_sha256=None if previous is None else previous["model_sha256"],
        previous_models=None if previous is None else previous["models"],
    )
    _write_json_if_generation(path, payload, expected_generation)
    return resolve_pointer(
        runs_root=runs_root,
        mode=record.mode,
        role="champion",
    )


def rollback_champion(
    *,
    runs_root: str,
    mode: str,
) -> ResolvedPointer:
    """Restore the previous production champion for one mode.

    The current champion stack becomes ``previous_stack_id``, so a second
    rollback returns to the stack that was just left. The previous stack's
    ``model.txt`` and checkpoints are checked. When the pointer stored their
    fingerprints, those hashes must still match. A pointer written before
    those fields existed restores the previous stack after its files and
    completed runs are checked. The current champion's files are left in
    place.

    Args:
        runs_root: Root that contains ``pointers`` and the completed runs.
        mode: Gold dataset mode, such as ``sample`` or ``curated``.

    Returns:
        The champion pointer after the previous stack has been published.

    Raises:
        FileNotFoundError: No champion pointer exists, or the previous stack
            is missing a required file.
        ValueError: The champion has no previous stack, or the previous
            stack does not match its recorded fingerprints.
        PointerConflictError: The champion changed before it was published.
    """

    path = pointer_path(runs_root, mode, "champion")
    generation = pointer_generation(path)
    if generation is None:
        raise FileNotFoundError(path)
    payload = _read_json(path)
    _require_equal(payload.get("role"), "champion", f"Pointer {path} role")
    _require_equal(payload.get("mode"), mode, f"Pointer {path} mode")
    current_stack_id = _require_segment(payload.get("stack_id"), "stack_id")
    previous_stack_id = _optional_stack_id(payload.get("previous_stack_id"), path)
    if previous_stack_id is None:
        raise ValueError(
            f"Champion {current_stack_id} has no previous stack to restore"
        )
    previous_dir = stack_output_dir(runs_root, previous_stack_id)
    record = _load_stack_record(previous_dir)
    _require_equal(record.mode, mode, f"Stack {previous_dir} mode")
    if _previous_fingerprints_recorded(payload):
        _require_previous_fingerprints(
            path=path,
            payload=payload,
            record=record,
            runs_root=runs_root,
        )
    else:
        logger.warning(
            "Pointer %s has no stored previous fingerprints; "
            "restoring %s after checking its files",
            path,
            previous_stack_id,
        )
    return publish_champion_if_current(
        runs_root=runs_root,
        stack_dir=previous_dir,
        expected_generation=generation,
        previous_stack_id=current_stack_id,
    )


def _pointer_payload(
    record,
    *,
    role: str,
    runs_root: str,
    previous_stack_id: str | None,
    previous_model_sha256: str | None = None,
    previous_models: dict | None = None,
) -> dict:
    return {
        "role": role,
        "mode": record.mode,
        "timeline": record.timeline,
        "stack_id": record.stack_id,
        "model_path": record.model_path,
        "model_sha256": _file_sha256(record.model_path),
        "previous_stack_id": previous_stack_id,
        "previous_model_sha256": previous_model_sha256,
        "previous_models": previous_models,
        "promoted_at": _utc_now(),
        "models": {
            name: {
                "run_id": run_id,
                "checkpoint_sha256": _checkpoint_sha256(runs_root, name, run_id),
            }
            for name, run_id in record.run_ids.items()
        },
    }


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


def _existing_pointer(path: str) -> dict | None:
    if not success_marker_exists(path):
        return None
    payload = _read_json(path)
    stack_id = payload.get("stack_id")
    if not isinstance(stack_id, str) or not stack_id.strip():
        raise ValueError(f"Pointer {path} has no stack_id")
    model_sha256 = payload.get("model_sha256")
    models = payload.get("models")
    if not isinstance(model_sha256, str) or not model_sha256:
        raise ValueError(f"Pointer {path} has no model_sha256")
    if not isinstance(models, dict):
        raise ValueError(f"Pointer {path} has no models")
    return {
        "stack_id": stack_id,
        "model_sha256": model_sha256,
        "models": models,
    }


def _previous_fingerprints_recorded(payload: dict) -> bool:
    recorded_model = payload.get("previous_model_sha256")
    previous_models = payload.get("previous_models")
    has_model = isinstance(recorded_model, str) and bool(recorded_model)
    has_models = isinstance(previous_models, dict)
    return has_model or has_models


def _require_previous_fingerprints(
    *,
    path: str,
    payload: dict,
    record,
    runs_root: str,
) -> None:
    recorded_model = payload.get("previous_model_sha256")
    if not isinstance(recorded_model, str) or not recorded_model:
        raise ValueError(f"Pointer {path} has no previous_model_sha256")
    _require_fingerprint(
        recorded_model,
        _file_sha256(record.model_path),
        f"Pointer {path} previous_model_sha256",
    )
    previous_models = payload.get("previous_models")
    if not isinstance(previous_models, dict):
        raise ValueError(f"Pointer {path} has no previous_models")
    for name, run_id in record.run_ids.items():
        entry = previous_models.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Pointer {path} previous model {name} is not an object")
        _require_equal(
            entry.get("run_id"),
            run_id,
            f"Pointer {path} previous run_id for {name}",
        )
        _require_fingerprint(
            entry.get("checkpoint_sha256"),
            _checkpoint_sha256(runs_root, name, run_id),
            f"Pointer {path} previous checkpoint_sha256 for {name}",
        )


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


def _write_json_if_generation(
    path: str,
    payload: dict,
    expected_generation: int | None,
) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=str)
    if is_gcs_url(path):
        _gcs_upload_if_generation(path, text, expected_generation)
        return
    if pointer_generation(path) != expected_generation:
        raise PointerConflictError(f"Pointer {path} changed before it was published")
    write_json(path, payload)


def _gcs_generation(path: str) -> int | None:
    from google.cloud import storage

    bucket_name, blob_name = _split_gcs(path)
    blob = storage.Client().bucket(bucket_name).get_blob(blob_name)
    if blob is None:
        return None
    return int(blob.generation)


def _gcs_upload_if_generation(
    path: str,
    text: str,
    expected_generation: int | None,
) -> None:
    from google.api_core.exceptions import PreconditionFailed
    from google.cloud import storage

    bucket_name, blob_name = _split_gcs(path)
    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    match = 0 if expected_generation is None else expected_generation
    try:
        blob.upload_from_string(
            text,
            content_type="application/json",
            if_generation_match=match,
        )
    except PreconditionFailed as exc:
        raise PointerConflictError(
            f"Pointer {path} changed before it was published"
        ) from exc


def _split_gcs(path: str) -> tuple[str, str]:
    without_scheme = path[len("gs://") :]
    bucket_name, _, blob_name = without_scheme.partition("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid GCS path {path}")
    return bucket_name, blob_name


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
