import hashlib
import json
import os
from pathlib import Path

import pytest

from instacart_etl_rnn.common.paths import join_path
from instacart_platform.pointers import (
    PointerConflictError,
    pointer_generation,
    publish_champion_if_current,
    publish_pointer,
    resolve_pointer,
    rollback_champion,
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
    pointer = tmp_path / "pointers" / "curated" / "champion.json"
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
        str(tmp_path / "pointers" / "curated" / "champion.json"),
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
        str(tmp_path / "pointers" / "curated" / "champion.json"),
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


def _champion_file(runs_root: Path) -> Path:
    return runs_root / "pointers" / "curated" / "champion.json"


def test_publish_champion_if_current_creates_a_missing_champion(tmp_path):
    stack_dir = _write_named_stack(tmp_path)
    champion_file = _champion_file(tmp_path)
    assert pointer_generation(str(champion_file)) is None

    resolved = publish_champion_if_current(
        runs_root=str(tmp_path),
        stack_dir=stack_dir,
        expected_generation=None,
        previous_stack_id=None,
    )

    assert resolved.role == "champion"
    assert resolved.stack_id == "stack-named"
    assert resolved.previous_stack_id is None
    assert pointer_generation(str(champion_file)) is not None


def test_publish_champion_if_current_rejects_a_stale_generation(tmp_path):
    stack_dir = _write_named_stack(tmp_path, stack_id="stack-first")
    publish_champion_if_current(
        runs_root=str(tmp_path),
        stack_dir=stack_dir,
        expected_generation=None,
        previous_stack_id=None,
    )
    champion_file = _champion_file(tmp_path)
    observed = pointer_generation(str(champion_file))
    os.utime(champion_file, ns=(1, 1))
    replacement = _write_stack(tmp_path, stack_id="stack-second", run_id=NAMED_RUN)

    with pytest.raises(PointerConflictError, match="changed"):
        publish_champion_if_current(
            runs_root=str(tmp_path),
            stack_dir=replacement,
            expected_generation=observed,
            previous_stack_id="stack-first",
        )

    payload = json.loads(champion_file.read_text(encoding="utf-8"))
    assert payload["stack_id"] == "stack-first"


def test_publish_champion_if_current_rejects_create_when_the_file_exists(tmp_path):
    stack_dir = _write_named_stack(tmp_path, stack_id="stack-first")
    publish_champion_if_current(
        runs_root=str(tmp_path),
        stack_dir=stack_dir,
        expected_generation=None,
        previous_stack_id=None,
    )
    replacement = _write_stack(tmp_path, stack_id="stack-second", run_id=NAMED_RUN)

    with pytest.raises(PointerConflictError, match="changed"):
        publish_champion_if_current(
            runs_root=str(tmp_path),
            stack_dir=replacement,
            expected_generation=None,
            previous_stack_id=None,
        )

    payload = json.loads(_champion_file(tmp_path).read_text(encoding="utf-8"))
    assert payload["stack_id"] == "stack-first"


def test_champion_pointer_is_shared_across_timelines(tmp_path):
    for name in REQUIRED_MODELS:
        _write_run(tmp_path, model_name=name, run_id=NAMED_RUN)
    initial = _write_stack(
        tmp_path,
        stack_id="stack-initial",
        run_id=NAMED_RUN,
        timeline="initial",
    )
    publish_pointer(runs_root=str(tmp_path), stack_dir=initial, role="champion")
    champion_file = _champion_file(tmp_path)
    assert champion_file.is_file()
    timeline_champion = tmp_path / "pointers" / "curated" / "initial" / "champion.json"
    assert not timeline_champion.exists()

    resolved = resolve_pointer(
        runs_root=str(tmp_path),
        mode="curated",
        role="champion",
    )
    assert resolved.timeline == "initial"
    assert resolved.stack_id == "stack-initial"

    later = _write_stack(
        tmp_path,
        stack_id="stack-t2",
        run_id=NAMED_RUN,
        timeline="t2",
    )
    promoted = publish_champion_if_current(
        runs_root=str(tmp_path),
        stack_dir=later,
        expected_generation=pointer_generation(str(champion_file)),
        previous_stack_id="stack-initial",
    )
    assert promoted.timeline == "t2"
    assert promoted.previous_stack_id == "stack-initial"
    assert not (tmp_path / "pointers" / "curated" / "t2" / "champion.json").exists()

    challenger_dir = _write_stack(
        tmp_path,
        stack_id="stack-t1",
        run_id=NAMED_RUN,
        timeline="t1",
    )
    challenger = publish_pointer(
        runs_root=str(tmp_path),
        stack_dir=challenger_dir,
        role="challenger",
    )
    assert challenger.timeline == "t1"
    challenger_file = tmp_path / "pointers" / "curated" / "t1" / "challenger.json"
    assert challenger_file.is_file()
    payload = json.loads(champion_file.read_text(encoding="utf-8"))
    assert payload["stack_id"] == "stack-t2"


def _publish_two_champions(tmp_path) -> None:
    for name in REQUIRED_MODELS:
        _write_run(tmp_path, model_name=name, run_id=NAMED_RUN)
    first = _write_stack(
        tmp_path,
        stack_id="stack-first",
        run_id=NAMED_RUN,
        timeline="initial",
    )
    second = _write_stack(
        tmp_path,
        stack_id="stack-second",
        run_id=NAMED_RUN,
        timeline="t1",
    )
    publish_pointer(runs_root=str(tmp_path), stack_dir=first, role="champion")
    publish_pointer(runs_root=str(tmp_path), stack_dir=second, role="champion")


def test_rollback_champion_restores_the_previous_stack(tmp_path):
    _publish_two_champions(tmp_path)

    restored = rollback_champion(runs_root=str(tmp_path), mode="curated")

    assert restored.stack_id == "stack-first"
    assert restored.previous_stack_id == "stack-second"
    assert restored.timeline == "initial"

    returned = rollback_champion(runs_root=str(tmp_path), mode="curated")

    assert returned.stack_id == "stack-second"
    assert returned.previous_stack_id == "stack-first"


def test_rollback_champion_rejects_a_champion_with_no_history(tmp_path):
    stack_dir = _write_named_stack(tmp_path)
    publish_pointer(runs_root=str(tmp_path), stack_dir=stack_dir, role="champion")

    with pytest.raises(ValueError, match="no previous stack"):
        rollback_champion(runs_root=str(tmp_path), mode="curated")


def test_rollback_champion_raises_when_the_pointer_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="champion.json"):
        rollback_champion(runs_root=str(tmp_path), mode="curated")


def test_rollback_champion_keeps_the_current_stack_when_the_previous_file_changed(
    tmp_path,
):
    _publish_two_champions(tmp_path)
    model_path = tmp_path / "stack" / "stack-first" / "model.txt"
    model_path.write_text("other\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model_sha256"):
        rollback_champion(runs_root=str(tmp_path), mode="curated")

    payload = json.loads(_champion_file(tmp_path).read_text(encoding="utf-8"))
    assert payload["stack_id"] == "stack-second"


def test_rollback_champion_restores_when_the_current_model_file_changed(tmp_path):
    _publish_two_champions(tmp_path)
    model_path = tmp_path / "stack" / "stack-second" / "model.txt"
    model_path.write_text("other\n", encoding="utf-8")

    restored = rollback_champion(runs_root=str(tmp_path), mode="curated")

    assert restored.stack_id == "stack-first"
    assert restored.previous_stack_id == "stack-second"


def test_rollback_champion_rejects_a_stale_generation(tmp_path, mocker):
    _publish_two_champions(tmp_path)
    champion_file = _champion_file(tmp_path)
    observed = pointer_generation(str(champion_file))
    mocker.patch(
        "instacart_platform.pointers.pointer_generation",
        side_effect=[observed, observed + 1],
    )

    with pytest.raises(PointerConflictError, match="changed"):
        rollback_champion(runs_root=str(tmp_path), mode="curated")

    payload = json.loads(champion_file.read_text(encoding="utf-8"))
    assert payload["stack_id"] == "stack-second"


def test_gcs_champion_upload_matches_the_observed_generation(mocker):
    blob = mocker.Mock()
    bucket = mocker.Mock()
    bucket.blob.return_value = blob
    client = mocker.Mock()
    client.bucket.return_value = bucket
    mocker.patch("google.cloud.storage.Client", return_value=client)

    from instacart_platform.pointers import _gcs_upload_if_generation

    _gcs_upload_if_generation("gs://bucket/pointers/champion.json", "{}", 17)

    blob.upload_from_string.assert_called_once_with(
        "{}",
        content_type="application/json",
        if_generation_match=17,
    )


def test_gcs_champion_upload_reports_a_precondition_failure(mocker):
    from google.api_core.exceptions import PreconditionFailed

    blob = mocker.Mock()
    blob.upload_from_string.side_effect = PreconditionFailed("changed")
    bucket = mocker.Mock()
    bucket.blob.return_value = blob
    client = mocker.Mock()
    client.bucket.return_value = bucket
    mocker.patch("google.cloud.storage.Client", return_value=client)

    from instacart_platform.pointers import _gcs_upload_if_generation

    with pytest.raises(PointerConflictError, match="changed"):
        _gcs_upload_if_generation("gs://bucket/pointers/champion.json", "{}", 0)
