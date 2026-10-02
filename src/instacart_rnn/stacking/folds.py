import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import gcsfs
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from instacart_etl_rnn.common.paths import is_gcs_url, join_path
from instacart_etl_rnn.common.setup_logging import configure_logging
from instacart_platform.runs import build_run_paths, write_json

logger = logging.getLogger(__name__)

STACKING_MODELS = (
    "product",
    "aisle",
    "reorder_size_gmm",
    "reorder_size_rnn",
)
FOLD_CHOICES = (4, 5)


@dataclass(frozen=True)
class SelectedArtifact:
    """One completed model's stacking-train inference export."""

    model_name: str
    run_id: str
    artifact_path: str
    completed_at: str
    train_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select completed stacking-train artifacts and write "
            "deterministic user folds for GBM feature selection."
        )
    )
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument(
        "--timeline",
        required=True,
        help=(
            "Gold timeline whose stacking-train export should be used. "
            "Use t1 for N-1 feature selection."
        ),
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        choices=FOLD_CHOICES,
        default=4,
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def generate_stack_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:6]
    return f"{timestamp}_{suffix}"


def stack_output_dir(runs_root: str, stack_id: str) -> str:
    """Return the immutable directory for one stack candidate.

    Args:
        runs_root: Root that contains one directory per model.
        stack_id: Id of the stack candidate being written.

    Returns:
        ``{runs_root}/stack/{stack_id}``.
    """

    if not stack_id.strip():
        raise ValueError("stack_id must not be empty")

    return str(join_path(join_path(runs_root.rstrip("/"), "stack"), stack_id))


def train_path_matches(train_path: str, *, mode: str, timeline: str) -> bool:
    return f"/training/{mode}/{timeline}/" in train_path


def select_stacking_artifacts(
    *,
    runs_root: str,
    mode: str,
    timeline: str,
) -> dict[str, SelectedArtifact]:
    """Select the latest completed stacking-train artifact for every model.

    A run qualifies when ``run.json`` says ``completed``, ``_SUCCESS`` exists,
    ``config.json`` ``train_path`` contains ``/training/{mode}/{timeline}/``,
    and ``artifacts/stacking_train`` exists. The latest ``completed_at`` wins.
    Equal timestamps use the greater ``run_id``.

    Args:
        runs_root: Root that contains one directory per model.
        mode: Gold dataset mode, such as ``sample`` or ``curated``.
        timeline: Gold timeline, such as ``t1``.

    Returns:
        Selected artifact for each stacking model, keyed by model name.
    """

    return {
        model_name: select_model_artifact(
            runs_root=runs_root,
            model_name=model_name,
            mode=mode,
            timeline=timeline,
        )
        for model_name in STACKING_MODELS
    }


def select_model_artifact(
    *,
    runs_root: str,
    model_name: str,
    mode: str,
    timeline: str,
) -> SelectedArtifact:
    candidates: list[SelectedArtifact] = []
    for run_id in list_run_ids(runs_root, model_name):
        artifact = _candidate_artifact(
            runs_root=runs_root,
            model_name=model_name,
            run_id=run_id,
            mode=mode,
            timeline=timeline,
        )
        if artifact is not None:
            candidates.append(artifact)

    if not candidates:
        raise FileNotFoundError(
            f"No completed stacking_train artifact for model {model_name} "
            f"with train_path containing /training/{mode}/{timeline}/"
        )

    return max(candidates, key=lambda item: (item.completed_at, item.run_id))


def assign_user_folds(
    user_ids: list[int],
    *,
    n_folds: int,
    seed: int,
) -> dict[int, int]:
    """Assign each user to a stable fold.

    The fold is a hash of the seed and user id, so row order cannot change it.
    Every product row for a user stays in that user's fold.

    Args:
        user_ids: Distinct user identifiers.
        n_folds: Number of folds, either 4 or 5.
        seed: Salt that keeps the assignment repeatable.

    Returns:
        Mapping of user id to a fold in ``0 .. n_folds - 1``.
    """

    if n_folds not in FOLD_CHOICES:
        raise ValueError(f"n_folds must be one of {FOLD_CHOICES}, received {n_folds}")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if not user_ids:
        raise ValueError("user_ids must not be empty")

    return {user_id: _fold_for_user(user_id, n_folds, seed) for user_id in user_ids}


def read_user_ids(artifact_path: str) -> list[int]:
    """Return the sorted distinct ``user_id`` values in an inference export."""

    table = pads.dataset(artifact_path, format="parquet").to_table(columns=["user_id"])
    user_ids = sorted({int(value) for value in table.column("user_id").to_pylist()})
    if not user_ids:
        raise ValueError(f"No user_id values found in {artifact_path}")
    return user_ids


def write_user_folds(
    output_path: str,
    folds: dict[int, int],
) -> None:
    """Write ``user_id`` and ``fold`` to a Parquet file, sorted by user id."""

    ordered_ids = sorted(folds)
    table = pa.table(
        {
            "user_id": pa.array(ordered_ids, type=pa.int64()),
            "fold": pa.array(
                [folds[user_id] for user_id in ordered_ids], type=pa.int8()
            ),
        }
    )
    sink = _open_binary(output_path)
    try:
        pq.write_table(table, sink)
    finally:
        if hasattr(sink, "close"):
            sink.close()


def list_run_ids(runs_root: str, model_name: str) -> list[str]:
    model_root = join_path(runs_root.rstrip("/"), model_name)
    return _child_names(str(model_root))


def main() -> None:
    """Select artifacts, assign product-user folds, and write both outputs.

    Logs go to stderr. The resolved stack directory is printed alone on
    stdout so a shell script can pass it to feature selection.
    """

    root_logger = logging.getLogger()
    previous_handlers = list(root_logger.handlers)
    previous_level = root_logger.level
    try:
        configure_logging()
        _send_logs_to_stderr()
        _write_stack(parse_args())
    finally:
        _restore_logging(root_logger, previous_handlers, previous_level)


def _write_stack(args: argparse.Namespace) -> None:
    artifacts = select_stacking_artifacts(
        runs_root=args.runs_root,
        mode=args.mode,
        timeline=args.timeline,
    )
    for artifact in artifacts.values():
        logger.info(
            "Selected %s run %s completed_at=%s artifact=%s",
            artifact.model_name,
            artifact.run_id,
            artifact.completed_at,
            artifact.artifact_path,
        )

    user_ids = read_user_ids(artifacts["product"].artifact_path)
    folds = assign_user_folds(user_ids, n_folds=args.n_folds, seed=args.seed)
    stack_id = generate_stack_id()
    output_dir = stack_output_dir(args.runs_root, stack_id)
    write_json(
        str(join_path(output_dir, "selected_artifacts.json")),
        {
            "stack_id": stack_id,
            "mode": args.mode,
            "timeline": args.timeline,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "models": {name: asdict(artifact) for name, artifact in artifacts.items()},
        },
    )
    write_user_folds(str(join_path(output_dir, "user_folds.parquet")), folds)
    logger.info(
        "Wrote stack_id=%s with %d user folds across %d folds to %s",
        stack_id,
        len(folds),
        args.n_folds,
        output_dir,
    )
    print(output_dir, flush=True)


def _restore_logging(
    root_logger: logging.Logger,
    previous_handlers: list[logging.Handler],
    previous_level: int,
) -> None:
    root_logger.handlers.clear()
    for handler in previous_handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(previous_level)


def _send_logs_to_stderr() -> None:
    for handler in logging.getLogger().handlers:
        if getattr(handler, "stream", None) is sys.stdout:
            handler.setStream(sys.stderr)


def _candidate_artifact(
    *,
    runs_root: str,
    model_name: str,
    run_id: str,
    mode: str,
    timeline: str,
) -> SelectedArtifact | None:
    paths = build_run_paths(
        runs_root=runs_root,
        model_name=model_name,
        run_id=run_id,
    )
    if not _exists(paths.success_marker):
        return None

    run_payload = _read_json(paths.run_json)
    if run_payload.get("status") != "completed":
        return None

    config_payload = _read_json(paths.config_json)
    train_path = config_payload.get("train_path")
    if not isinstance(train_path, str) or not train_path_matches(
        train_path,
        mode=mode,
        timeline=timeline,
    ):
        return None

    artifact_path = str(join_path(paths.artifacts, "stacking_train"))
    if not _exists(artifact_path):
        return None

    completed_at = run_payload.get("completed_at")
    if not isinstance(completed_at, str) or not completed_at:
        return None

    return SelectedArtifact(
        model_name=model_name,
        run_id=run_id,
        artifact_path=artifact_path,
        completed_at=completed_at,
        train_path=train_path,
    )


def _fold_for_user(user_id: int, n_folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % n_folds


def _child_names(root: str) -> list[str]:
    root = root.rstrip("/")
    if is_gcs_url(root):
        filesystem = gcsfs.GCSFileSystem()
        filesystem.invalidate_cache(root)
        if not filesystem.exists(root):
            return []
        root_key = root.removeprefix("gs://").rstrip("/")
        names: list[str] = []
        for entry in filesystem.ls(root, detail=False):
            entry_key = str(entry).removeprefix("gs://").rstrip("/")
            if entry_key == root_key:
                continue
            names.append(entry_key.split("/")[-1])
        return names

    path = Path(root)
    if not path.is_dir():
        return []
    return [child.name for child in path.iterdir() if child.is_dir()]


def _exists(path: str) -> bool:
    if is_gcs_url(path):
        filesystem = gcsfs.GCSFileSystem()
        filesystem.invalidate_cache(path)
        return bool(filesystem.exists(path))
    return Path(path).exists()


def _read_json(path: str) -> dict:
    if is_gcs_url(path):
        filesystem = gcsfs.GCSFileSystem()
        filesystem.invalidate_cache(path)
        with filesystem.open(path, "r") as file:
            payload = json.load(file)
    else:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


def _open_binary(path: str):
    if is_gcs_url(path):
        return gcsfs.GCSFileSystem().open(path, "wb")

    parent = Path(path).parent
    parent.mkdir(parents=True, exist_ok=True)
    return path


if __name__ == "__main__":
    main()
