import hashlib
import json
from pathlib import Path

import pytest

from instacart_etl_rnn.common.paths import join_path
from instacart_platform.pointers import (
    publish_pointer,
    resolve_pointer,
)
from instacart_platform.runs import build_run_paths, write_json, write_success_marker
from instacart_rnn.stacking.feature_selection import REQUIRED_MODELS

NAMED_RUN = "20260101T000000Z_named"
NEWER_RUN = "20260201T000000Z_newer"


def _artifact_path(runs_root: Path, model_name: str, run_id: str) -> str:
    paths = build_run_paths(
        runs_root=str(runs_root),
        model_name=model_name,
        run_id=run_id,
    )
    return str(join_path(paths.artifacts, "stacking_train"))


def _write_run(
    runs_root: Path,
    *,
    model_name: str,
    run_id: str,
    success: bool = True,
    status: str = "completed",
) -> None:
    paths = build_run_paths(
        runs_root=str(runs_root),
        model_name=model_name,
        run_id=run_id,
    )
    write_json(
        str(paths.run_json),
        {"status": status, "completed_at": "2026-01-01T00:00:00+00:00"},
    )
    if success:
        write_success_marker(str(paths.success_marker))
    Path(_artifact_path(runs_root, model_name, run_id)).mkdir(parents=True)
    checkpoint = Path(paths.best_checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(f"{model_name}:{run_id}".encode())


def _write_stack(
    runs_root: Path,
    *,
    stack_id: str,
    run_id: str,
    mode: str = "curated",
    timeline: str = "t1",
) -> str:
    stack_dir = runs_root / "stack" / stack_id
    stack_dir.mkdir(parents=True)
    model_path = str(join_path(stack_dir, "model.txt"))
    Path(model_path).write_text("tree\n", encoding="utf-8")
    write_json(
        str(join_path(stack_dir, "stacking_train.json")),
        {"model_path": model_path},
    )
    write_json(
        str(join_path(stack_dir, "selected_artifacts.json")),
        {
            "stack_id": stack_id,
            "mode": mode,
            "timeline": timeline,
            "models": {
                name: {
                    "run_id": run_id,
                    "artifact_path": _artifact_path(runs_root, name, run_id),
                }
                for name in REQUIRED_MODELS
            },
        },
    )
    return str(stack_dir)


def _write_named_stack(runs_root: Path, stack_id: str = "stack-named") -> str:
    for name in REQUIRED_MODELS:
        _write_run(runs_root, model_name=name, run_id=NAMED_RUN)
        _write_run(runs_root, model_name=name, run_id=NEWER_RUN)
    return _write_stack(runs_root, stack_id=stack_id, run_id=NAMED_RUN)


def test_resolve_pointer_raises_when_the_pointer_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="champion.json"):
        resolve_pointer(
            runs_root=str(tmp_path),
            mode="curated",
            timeline="t1",
            role="champion",
        )


def test_resolve_pointer_rejects_a_timeline_mismatch(tmp_path):
    _write_named_stack(tmp_path)
    publish_pointer(
        runs_root=str(tmp_path),
        stack_dir=str(tmp_path / "stack" / "stack-named"),
        role="champion",
    )
    pointer = tmp_path / "pointers" / "curated" / "t1" / "champion.json"
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    payload["timeline"] = "t2"
    pointer.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="timeline"):
        resolve_pointer(
            runs_root=str(tmp_path),
            mode="curated",
            timeline="t1",
            role="champion",
        )


def test_resolve_pointer_rejects_a_stack_id_that_disagrees_with_the_manifest(
    tmp_path,
):
    stack_dir = Path(_write_named_stack(tmp_path))
    manifest_path = stack_dir / "selected_artifacts.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["stack_id"] = "stack-other"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    write_json(
        str(tmp_path / "pointers" / "curated" / "t1" / "champion.json"),
        {
            "role": "champion",
            "mode": "curated",
            "timeline": "t1",
            "stack_id": "stack-named",
            "model_path": str(stack_dir / "model.txt"),
            "previous_stack_id": None,
            "promoted_at": "2026-03-01T00:00:00+00:00",
            "models": {name: {"run_id": NAMED_RUN} for name in REQUIRED_MODELS},
        },
    )

    with pytest.raises(ValueError, match="stack_id"):
        resolve_pointer(
            runs_root=str(tmp_path),
            mode="curated",
            timeline="t1",
            role="champion",
        )


def test_resolve_pointer_rejects_a_named_run_without_success(tmp_path):
    for name in REQUIRED_MODELS:
        _write_run(runs_root=tmp_path, model_name=name, run_id=NAMED_RUN, success=False)
    _write_stack(tmp_path, stack_id="stack-named", run_id=NAMED_RUN)
    write_json(
        str(tmp_path / "pointers" / "curated" / "t1" / "champion.json"),
        {
            "role": "champion",
            "mode": "curated",
            "timeline": "t1",
            "stack_id": "stack-named",
            "model_path": str(tmp_path / "stack" / "stack-named" / "model.txt"),
            "previous_stack_id": None,
            "promoted_at": "2026-03-01T00:00:00+00:00",
            "models": {name: {"run_id": NAMED_RUN} for name in REQUIRED_MODELS},
        },
    )

    with pytest.raises(FileNotFoundError, match="_SUCCESS"):
        resolve_pointer(
            runs_root=str(tmp_path),
            mode="curated",
            timeline="t1",
            role="champion",
        )


def test_resolve_pointer_returns_the_named_runs_when_a_newer_run_exists(tmp_path):
    stack_dir = _write_named_stack(tmp_path)

    resolved = publish_pointer(
        runs_root=str(tmp_path),
        stack_dir=stack_dir,
        role="champion",
    )

    assert resolved.stack_id == "stack-named"
    assert resolved.model_path == str(join_path(stack_dir, "model.txt"))
    assert resolved.previous_stack_id is None
    assert set(resolved.models) == set(REQUIRED_MODELS)
    product = resolved.models["product"]
    assert product.run_id == NAMED_RUN
    assert product.artifact_path == _artifact_path(tmp_path, "product", NAMED_RUN)
    assert product.best_checkpoint == str(
        build_run_paths(
            runs_root=str(tmp_path),
            model_name="product",
            run_id=NAMED_RUN,
        ).best_checkpoint
    )
    assert NEWER_RUN not in product.artifact_path
    assert resolved.model_sha256 == _sha256(Path(resolved.model_path))
    assert product.checkpoint_sha256 == _sha256(Path(product.best_checkpoint))


def test_publish_pointer_records_the_previous_stack_id(tmp_path):
    first = _write_named_stack(tmp_path, stack_id="stack-first")
    publish_pointer(runs_root=str(tmp_path), stack_dir=first, role="champion")
    second = _write_stack(tmp_path, stack_id="stack-second", run_id=NAMED_RUN)

    resolved = publish_pointer(
        runs_root=str(tmp_path),
        stack_dir=second,
        role="champion",
    )

    assert resolved.stack_id == "stack-second"
    assert resolved.previous_stack_id == "stack-first"


def test_resolve_pointer_rejects_a_replaced_model_file(tmp_path):
    stack_dir = _write_named_stack(tmp_path)
    publish_pointer(
        runs_root=str(tmp_path),
        stack_dir=stack_dir,
        role="champion",
    )
    Path(join_path(stack_dir, "model.txt")).write_text("other\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model_sha256"):
        resolve_pointer(
            runs_root=str(tmp_path),
            mode="curated",
            timeline="t1",
            role="champion",
        )


def test_resolve_pointer_rejects_a_replaced_checkpoint(tmp_path):
    stack_dir = _write_named_stack(tmp_path)
    publish_pointer(
        runs_root=str(tmp_path),
        stack_dir=stack_dir,
        role="champion",
    )
    checkpoint = build_run_paths(
        runs_root=str(tmp_path),
        model_name="product",
        run_id=NAMED_RUN,
    ).best_checkpoint
    Path(checkpoint).write_bytes(b"replaced")

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        resolve_pointer(
            runs_root=str(tmp_path),
            mode="curated",
            timeline="t1",
            role="champion",
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
