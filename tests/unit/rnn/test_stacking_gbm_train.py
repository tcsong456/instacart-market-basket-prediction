import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from instacart_platform.pointers import (
    PointerConflictError,
    ResolvedModel,
    ResolvedPointer,
)
from instacart_rnn.stacking.feature_selection import REQUIRED_MODELS
from instacart_rnn.stacking.stacking_train import (
    NONE_TOKEN,
    _evaluation_input_paths,
    basket_f1,
    choose_prefix,
    evaluation_artifact_path,
    evaluation_label_path,
    execute_backend_run,
    join_labeled_features,
    load_champion_evaluation,
    mean_order_f1,
    predicted_basket,
    prediction_directory,
    promote_trained_stack,
    resolve_backend_run,
    resolve_stack_directory,
    save_booster,
    score_labeled_frame,
    stacking_validation_artifact_path,
    stacking_validation_label_path,
    train_booster,
    true_basket,
    write_predictions,
)
from instacart_rnn.stacking.stacking_train import (
    _require_disjoint_users as require_disjoint_users,
)


def test_evaluation_label_path_replaces_the_train_suffix():
    path = evaluation_label_path(
        "gs://gold/training/curated/t1/product_training_data_train"
    )

    assert path == ("gs://gold/training/curated/t1/product_training_data_evaluation")


def test_evaluation_label_path_rejects_a_non_train_file():
    with pytest.raises(ValueError, match="_training_data_train"):
        evaluation_label_path(
            "gs://gold/training/curated/t1/product_training_data_validation"
        )


def test_evaluation_artifact_path_switches_the_split_directory():
    path = evaluation_artifact_path("gs://runs/product/run-1/artifacts/stacking_train")

    assert path == "gs://runs/product/run-1/artifacts/evaluation"


def test_stacking_validation_artifact_path_switches_the_split_directory():
    path = stacking_validation_artifact_path(
        "gs://runs/product/run-1/artifacts/stacking_train"
    )

    assert path == "gs://runs/product/run-1/artifacts/stacking_validation"


def test_stacking_validation_label_path_uses_the_shared_snapshot():
    path = stacking_validation_label_path(
        "gs://gold/training/sample/t1/product_training_data_train"
    )

    assert path == (
        "gs://gold/training/sample/stacking_train/product_training_data_validation"
    )


def test_stacking_validation_label_path_rejects_a_non_train_file():
    with pytest.raises(ValueError, match="_training_data_train"):
        stacking_validation_label_path(
            "gs://gold/training/sample/t1/product_training_data_validation"
        )


def test_prediction_directory_uses_the_run_artifacts_for_both_splits(tmp_path):
    paths = resolve_backend_run(str(tmp_path), "stacking_gbm", "run-1")

    evaluation = prediction_directory(
        paths=paths,
        stack_dir=str(tmp_path / "stack" / "stack-1"),
        split="evaluation",
    )
    stacking_n = prediction_directory(
        paths=paths,
        stack_dir=str(tmp_path / "stack" / "stack-1"),
        split="stacking_validation",
    )

    assert evaluation == str(
        tmp_path / "stacking_gbm" / "run-1" / "artifacts" / "evaluation"
    )
    assert stacking_n == str(
        tmp_path / "stacking_gbm" / "run-1" / "artifacts" / "stacking_validation"
    )


def test_evaluation_artifact_path_rejects_a_different_split():
    with pytest.raises(ValueError, match="stacking_train"):
        evaluation_artifact_path("gs://runs/product/run-1/artifacts/evaluation")


def test_train_booster_stores_the_feature_names(mocker):
    lightgbm = mocker.Mock()
    lightgbm.train.return_value = "booster"
    mocker.patch.dict("sys.modules", {"lightgbm": lightgbm})

    booster = train_booster(
        np.array([[0.1, 0.2]]),
        np.array([1.0]),
        parameters={"objective": "binary"},
        num_boost_round=2,
        seed=7,
        feature_names=["product_logit", "aisle_logit"],
    )

    assert booster == "booster"
    assert lightgbm.Dataset.call_args.kwargs["feature_name"] == [
        "product_logit",
        "aisle_logit",
    ]


def test_train_booster_rejects_a_feature_name_width_mismatch():
    with pytest.raises(ValueError, match="feature names"):
        train_booster(
            np.array([[0.1, 0.2]]),
            np.array([1.0]),
            parameters={"objective": "binary"},
            num_boost_round=2,
            seed=7,
            feature_names=["product_logit"],
        )


def test_save_booster_writes_the_local_model_text(tmp_path):
    destination = tmp_path / "nested" / "model.txt"

    class _Booster:
        def save_model(self, path: str) -> None:
            from pathlib import Path

            Path(path).write_text("tree\n", encoding="utf-8")

    save_booster(_Booster(), str(destination))

    assert destination.read_text(encoding="utf-8") == "tree\n"


def test_save_booster_uploads_the_model_bytes(mocker):
    from contextlib import contextmanager

    written = {}

    @contextmanager
    def open_object(path, mode):
        class _File:
            def write(self, payload):
                written["path"] = path
                written["mode"] = mode
                written["payload"] = payload

        yield _File()

    filesystem = mocker.Mock()
    filesystem.open.side_effect = open_object
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.gcsfs.GCSFileSystem",
        return_value=filesystem,
    )

    class _Booster:
        def save_model(self, path: str) -> None:
            from pathlib import Path

            Path(path).write_bytes(b"tree\n")

    save_booster(_Booster(), "gs://runs/stack/stack-1/model.txt")

    assert written == {
        "path": "gs://runs/stack/stack-1/model.txt",
        "mode": "wb",
        "payload": b"tree\n",
    }


def test_resolve_stack_directory_reuses_an_existing_directory(mocker):
    write_stack = mocker.patch("instacart_rnn.stacking.stacking_train.write_stack")

    stack_dir = resolve_stack_directory(
        stack_dir="gs://runs/stack/stack-1/",
        runs_root=None,
        mode=None,
        timeline=None,
        n_folds=5,
        seed=42,
    )

    assert stack_dir == "gs://runs/stack/stack-1"
    write_stack.assert_not_called()


def test_resolve_stack_directory_selects_the_latest_runs(mocker):
    write_stack = mocker.patch(
        "instacart_rnn.stacking.stacking_train.write_stack",
        return_value="gs://runs/stack/new-stack",
    )

    stack_dir = resolve_stack_directory(
        stack_dir=None,
        runs_root="gs://runs",
        mode="curated",
        timeline="initial",
        n_folds=5,
        seed=42,
    )

    assert stack_dir == "gs://runs/stack/new-stack"
    write_stack.assert_called_once_with(
        runs_root="gs://runs",
        mode="curated",
        timeline="initial",
        n_folds=5,
        seed=42,
    )


def test_resolve_stack_directory_prefers_an_existing_directory(mocker):
    write_stack = mocker.patch("instacart_rnn.stacking.stacking_train.write_stack")

    stack_dir = resolve_stack_directory(
        stack_dir="gs://runs/stack/stack-1/",
        runs_root="gs://runs",
        mode="curated",
        timeline="t1",
        n_folds=5,
        seed=42,
    )

    assert stack_dir == "gs://runs/stack/stack-1"
    write_stack.assert_not_called()


def test_resolve_backend_run_requires_all_three_arguments():
    assert resolve_backend_run(None, None, None) is None
    with pytest.raises(ValueError, match="must be set together"):
        resolve_backend_run("gs://runs", None, "run-1")


def test_execute_backend_run_writes_completed_status_and_success_marker(
    tmp_path,
    mocker,
):
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.utc_now",
        side_effect=["2026-01-01T00:00:00+00:00", "2026-01-01T01:00:00+00:00"],
    )
    paths = resolve_backend_run(str(tmp_path), "stacking_gbm", "run-1")

    result = execute_backend_run(
        paths,
        run_attributes={
            "run_id": "run-1",
            "model_name": "stacking_gbm",
            "git_commit": "abc123",
            "image": "img:tag",
        },
        config={"stack_dir": "gs://runs/stack/stack-1", "seed": 42},
        work=lambda: {
            "evaluation_log_loss": 0.24,
            "evaluation_f1": 0.37,
            "evaluation_predictions_path": (
                "gs://runs/stacking_gbm/run-1/artifacts/evaluation/predictions.parquet"
            ),
            "stacking_validation_log_loss": 0.31,
            "stacking_validation_f1": 0.28,
            "stacking_validation_predictions_path": (
                "gs://runs/stacking_gbm/run-1/artifacts/"
                "stacking_validation/predictions.parquet"
            ),
            "model_path": "gs://runs/stack/stack-1/model.txt",
        },
    )

    run_root = tmp_path / "stacking_gbm" / "run-1"
    run_payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_root / "metrics.json").read_text(encoding="utf-8"))

    assert result["evaluation_f1"] == 0.37
    assert metrics["stacking_validation_f1"] == 0.28
    assert run_payload["status"] == "completed"
    assert run_payload["started_at"] == "2026-01-01T00:00:00+00:00"
    assert run_payload["completed_at"] == "2026-01-01T01:00:00+00:00"
    assert metrics["model_path"] == "gs://runs/stack/stack-1/model.txt"
    assert (run_root / "_SUCCESS").exists()
    assert (run_root / "config.json").exists()


def test_execute_backend_run_marks_failed_without_a_success_marker(tmp_path, mocker):
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.utc_now",
        side_effect=["2026-01-01T00:00:00+00:00", "2026-01-01T01:00:00+00:00"],
    )
    paths = resolve_backend_run(str(tmp_path), "stacking_gbm", "run-1")

    def fail():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        execute_backend_run(
            paths,
            run_attributes={
                "run_id": "run-1",
                "model_name": "stacking_gbm",
                "git_commit": "abc123",
                "image": "img:tag",
            },
            config={"stack_dir": "gs://runs/stack/stack-1"},
            work=fail,
        )

    run_root = tmp_path / "stacking_gbm" / "run-1"
    run_payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))

    assert run_payload["status"] == "failed"
    assert not (run_root / "_SUCCESS").exists()
    assert not (run_root / "metrics.json").exists()


def test_execute_backend_run_skips_work_when_success_marker_exists(tmp_path):
    run_root = tmp_path / "stacking_gbm" / "run-1"
    run_root.mkdir(parents=True)
    (run_root / "_SUCCESS").write_text("", encoding="utf-8")
    paths = resolve_backend_run(str(tmp_path), "stacking_gbm", "run-1")
    called = {"work": False}

    def work():
        called["work"] = True
        return {}

    result = execute_backend_run(
        paths,
        run_attributes={
            "run_id": "run-1",
            "model_name": "stacking_gbm",
            "git_commit": "",
            "image": "",
        },
        config={},
        work=work,
    )

    assert result is None
    assert called["work"] is False


def test_basket_f1_scores_the_empty_order_as_a_match():
    assert basket_f1({NONE_TOKEN}, {NONE_TOKEN}) == 1.0
    assert basket_f1({NONE_TOKEN}, {10}) == 0.0
    assert basket_f1({10, 11}, {10}) == pytest.approx(2.0 / 3.0)


def test_choose_prefix_keeps_a_high_probability_product():
    prefix, include_none = choose_prefix(np.array([0.9]))

    assert prefix == 1
    assert include_none is False


def test_choose_prefix_uses_none_when_every_product_is_unlikely():
    prefix, include_none = choose_prefix(np.array([0.05, 0.04]))

    assert include_none is True
    assert prefix == 0


def test_predicted_basket_keeps_the_highest_probability_products():
    basket = predicted_basket(
        np.array([10, 11, 12]),
        np.array([0.2, 0.95, 0.1]),
    )

    assert basket == {11}


def test_true_basket_uses_none_when_no_label_is_positive():
    assert true_basket(np.array([10, 11]), np.array([0, 0])) == {NONE_TOKEN}
    assert true_basket(np.array([10, 11]), np.array([1, 0])) == {10}


def test_mean_order_f1_averages_users_rather_than_rows():
    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "product_id": [10, 11, 20],
            "label": [1, 0, 0],
        }
    )

    score = mean_order_f1(frame, np.array([0.95, 0.05, 0.02]))

    assert score == 1.0


def test_disjoint_users_rejects_an_evaluation_user_seen_in_training():
    train = pd.DataFrame({"user_id": [1, 1, 2]})
    evaluation = pd.DataFrame({"user_id": [2, 3]})

    with pytest.raises(ValueError, match="overlap contains 1"):
        require_disjoint_users(train, evaluation)


def test_score_labeled_frame_returns_probabilities_and_basket_f1():
    frame = pd.DataFrame(
        {
            "user_id": [1, 1],
            "product_id": [10, 11],
            "label": [1.0, 0.0],
            "product_logit": [0.2, 0.1],
        }
    )

    class _Booster:
        def predict(self, rows):
            return np.array([0.95, 0.05])

    scored = score_labeled_frame(_Booster(), frame, ["product_logit"])

    assert list(scored["probabilities"]) == [0.95, 0.05]
    assert scored["n_users"] == 1
    assert scored["f1"] == 1.0


def test_write_predictions_uses_the_evaluation_columns(tmp_path):
    frame = pd.DataFrame(
        {
            "user_id": [7, 7],
            "product_id": [10, 11],
            "label": [1.0, 0.0],
        }
    )

    path = write_predictions(str(tmp_path / "evaluation"), frame, np.array([0.8, 0.2]))
    table = pq.read_table(path)

    assert path == str(tmp_path / "evaluation" / "predictions.parquet")
    assert table.column_names == ["user_id", "product_id", "label", "probability"]
    assert table.column("probability").to_pylist() == [0.8, 0.2]


def test_join_labeled_features_keeps_labeled_product_rows():
    product = pd.DataFrame(
        {
            "user_id": [1, 2],
            "product_id": [10, 20],
            "aisle_id": [100, 200],
            "product_logit": [0.1, 0.2],
        }
    )
    aisle = pd.DataFrame(
        {"user_id": [1, 2], "aisle_id": [100, 200], "aisle_logit": [1.0, 2.0]}
    )
    reorder = pd.DataFrame({"user_id": [1, 2], "reorder_prediction": [3.0, 4.0]})
    gmm = pd.DataFrame({"user_id": [1, 2], "gmm_nll_0": [8.0, 9.0]})
    labels = pd.DataFrame({"user_id": [1, 2], "product_id": [10, 20], "label": [1, 0]})

    frame = join_labeled_features(
        product=product,
        aisle=aisle,
        reorder=reorder,
        gmm=gmm,
        labels=labels,
    )

    assert list(frame["label"]) == [1, 0]
    assert list(frame["aisle_logit"]) == [1.0, 2.0]
    assert "fold" not in frame.columns


def _resolved_pointer(stack_id: str, role: str = "challenger") -> ResolvedPointer:
    return ResolvedPointer(
        role=role,
        mode="curated",
        timeline="t1",
        stack_id=stack_id,
        stack_dir=f"/tmp/stack/{stack_id}",
        model_path=f"/tmp/stack/{stack_id}/model.txt",
        previous_stack_id=None,
        promoted_at="2026-01-01T00:00:00+00:00",
        model_sha256="abc",
        models={},
    )


def _promote(tmp_path, mocker, *, generation, champion_f1=None, champion_error=None):
    stack_dir = tmp_path / "stack" / "stack-new"
    stack_dir.mkdir(parents=True)
    events = []

    def publish_challenger(**kwargs):
        events.append("challenger")
        assert kwargs["role"] == "challenger"
        return _resolved_pointer("stack-new")

    def publish_champion(**kwargs):
        recorded = json.loads((stack_dir / "stacking_train.json").read_text())
        assert recorded["promotion_chosen"] is True
        assert recorded["champion_updated"] is False
        events.append(("champion", kwargs["expected_generation"]))
        return _resolved_pointer("stack-new", role="champion")

    mocker.patch(
        "instacart_rnn.stacking.stacking_train.publish_pointer",
        side_effect=publish_challenger,
    )
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.pointer_generation",
        return_value=generation,
    )
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.publish_champion_if_current",
        side_effect=publish_champion,
    )

    def resolve_champion(**kwargs):
        assert kwargs["role"] == "champion"
        assert kwargs.get("timeline") is None
        if champion_error is not None:
            raise champion_error
        return _resolved_pointer("stack-old", role="champion")

    mocker.patch(
        "instacart_rnn.stacking.stacking_train.resolve_pointer",
        side_effect=resolve_champion,
    )
    mocker.patch(
        "instacart_rnn.stacking.stacking_train._evaluation_input_paths",
        return_value={name: f"{name}.parquet" for name in REQUIRED_MODELS},
    )
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.load_booster",
        return_value=object(),
    )
    champion_frame = pd.DataFrame({"user_id": [9], "aisle_logit": [0.2]})
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.load_champion_evaluation",
        return_value=(champion_frame, ["aisle_logit"]),
    )

    def score(booster, frame, columns):
        events.append(("score", columns, frame["user_id"].tolist()))
        return {"f1": champion_f1}

    mocker.patch(
        "instacart_rnn.stacking.stacking_train.score_labeled_frame",
        side_effect=score,
    )
    return stack_dir, events


def test_promote_trained_stack_makes_the_first_stack_the_champion(tmp_path, mocker):
    stack_dir, events = _promote(tmp_path, mocker, generation=None)

    result = promote_trained_stack(
        runs_root=str(tmp_path),
        stack_dir=str(stack_dir),
        evaluation_frame=pd.DataFrame({"user_id": [1]}),
        evaluation_label_path="labels.parquet",
        challenger_f1=0.41,
        result={"evaluation_f1": 0.41},
    )

    assert events == ["challenger", ("champion", None)]
    assert result["promotion_chosen"] is True
    assert result["champion_updated"] is True
    assert result["champion_f1"] is None
    assert result["champion_timeline"] is None
    assert result["previous_champion_stack_id"] is None


def test_promote_trained_stack_promotes_a_higher_evaluation_f1(tmp_path, mocker):
    stack_dir, events = _promote(tmp_path, mocker, generation=17, champion_f1=0.2)

    result = promote_trained_stack(
        runs_root=str(tmp_path),
        stack_dir=str(stack_dir),
        evaluation_frame=pd.DataFrame({"user_id": [1], "product_logit": [0.1]}),
        evaluation_label_path="labels.parquet",
        challenger_f1=0.41,
        result={"evaluation_f1": 0.41},
    )

    assert events == [
        "challenger",
        ("score", ["aisle_logit"], [9]),
        ("champion", 17),
    ]
    assert result["promotion_chosen"] is True
    assert result["champion_f1"] == 0.2
    assert result["champion_timeline"] == "t1"
    assert result["previous_champion_stack_id"] == "stack-old"
    assert result["champion_updated"] is True


def test_promote_trained_stack_keeps_the_champion_when_f1_is_not_higher(
    tmp_path,
    mocker,
):
    stack_dir, events = _promote(tmp_path, mocker, generation=17, champion_f1=0.41)

    result = promote_trained_stack(
        runs_root=str(tmp_path),
        stack_dir=str(stack_dir),
        evaluation_frame=pd.DataFrame({"user_id": [1], "product_logit": [0.1]}),
        evaluation_label_path="labels.parquet",
        challenger_f1=0.41,
        result={"evaluation_f1": 0.41},
    )

    assert events == ["challenger", ("score", ["aisle_logit"], [9])]
    assert result["promotion_chosen"] is False
    assert result["champion_updated"] is False


def test_promote_trained_stack_does_not_replace_a_broken_champion(tmp_path, mocker):
    stack_dir, events = _promote(
        tmp_path,
        mocker,
        generation=17,
        champion_error=FileNotFoundError("gs://runs/stack/stack-old/model.txt"),
    )
    (stack_dir / "stacking_train.json").write_text('{"kept": true}', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="model.txt"):
        promote_trained_stack(
            runs_root=str(tmp_path),
            stack_dir=str(stack_dir),
            evaluation_frame=pd.DataFrame({"user_id": [1]}),
            evaluation_label_path="labels.parquet",
            challenger_f1=0.41,
            result={"evaluation_f1": 0.41},
        )

    assert events == ["challenger"]
    assert json.loads((stack_dir / "stacking_train.json").read_text()) == {"kept": True}


def test_promote_trained_stack_records_the_decision_before_a_conflict(
    tmp_path,
    mocker,
):
    stack_dir, _events = _promote(tmp_path, mocker, generation=None)
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.publish_champion_if_current",
        side_effect=PointerConflictError("Pointer changed before it was published"),
    )

    with pytest.raises(PointerConflictError, match="changed"):
        promote_trained_stack(
            runs_root=str(tmp_path),
            stack_dir=str(stack_dir),
            evaluation_frame=pd.DataFrame({"user_id": [1]}),
            evaluation_label_path="labels.parquet",
            challenger_f1=0.41,
            result={"evaluation_f1": 0.41},
        )

    recorded = json.loads((stack_dir / "stacking_train.json").read_text())
    assert recorded["promotion_chosen"] is True
    assert recorded["champion_updated"] is False


def test_execute_backend_run_copies_the_promotion_decision(tmp_path, mocker):
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.utc_now",
        side_effect=["2026-01-01T00:00:00+00:00", "2026-01-01T01:00:00+00:00"],
    )
    paths = resolve_backend_run(str(tmp_path), "stacking_gbm", "run-1")

    execute_backend_run(
        paths,
        run_attributes={
            "run_id": "run-1",
            "model_name": "stacking_gbm",
            "git_commit": "abc123",
            "image": "img:tag",
        },
        config={"stack_dir": "gs://runs/stack/stack-1"},
        work=lambda: {
            "evaluation_log_loss": 0.24,
            "evaluation_f1": 0.37,
            "evaluation_predictions_path": "predictions.parquet",
            "stacking_validation_log_loss": 0.31,
            "stacking_validation_f1": 0.28,
            "stacking_validation_predictions_path": "stacking.parquet",
            "model_path": "gs://runs/stack/stack-1/model.txt",
            "challenger_stack_id": "stack-new",
            "previous_champion_stack_id": "stack-old",
            "champion_timeline": "initial",
            "champion_f1": 0.2,
            "promotion_chosen": True,
            "champion_updated": True,
        },
    )

    metrics = json.loads(
        (tmp_path / "stacking_gbm" / "run-1" / "metrics.json").read_text()
    )
    assert metrics["promotion_chosen"] is True
    assert metrics["champion_f1"] == 0.2
    assert metrics["champion_timeline"] == "initial"
    assert metrics["challenger_stack_id"] == "stack-new"


def _champion_pointer(tmp_path, run_id: str) -> ResolvedPointer:
    stack_dir = tmp_path / "stack" / "stack-old"
    stack_dir.mkdir(parents=True)
    (stack_dir / "stacking_train.json").write_text(
        json.dumps({"feature_columns": ["product_logit", "aisle_logit"]}),
        encoding="utf-8",
    )
    models = {
        name: ResolvedModel(
            model_name=name,
            run_id=run_id,
            artifact_path=str(
                tmp_path / name / run_id / "artifacts" / "stacking_train"
            ),
            best_checkpoint=str(tmp_path / name / run_id / "checkpoints" / "best.pt"),
            checkpoint_sha256="abc",
        )
        for name in REQUIRED_MODELS
    }
    return ResolvedPointer(
        role="champion",
        mode="curated",
        timeline="t1",
        stack_id="stack-old",
        stack_dir=str(stack_dir),
        model_path=str(stack_dir / "model.txt"),
        previous_stack_id=None,
        promoted_at="2026-01-01T00:00:00+00:00",
        model_sha256="abc",
        models=models,
    )


def _outputs_for(user_id: int):
    product = pd.DataFrame(
        {
            "user_id": [user_id],
            "product_id": [10],
            "aisle_id": [100],
            "product_logit": [0.4],
        }
    )
    aisle = pd.DataFrame(
        {"user_id": [user_id], "aisle_id": [100], "aisle_logit": [0.3]}
    )
    reorder = pd.DataFrame({"user_id": [user_id], "reorder_prediction": [1.0]})
    gmm = pd.DataFrame({"user_id": [user_id], "gmm_nll_0": [0.5]})
    labels = pd.DataFrame({"user_id": [user_id], "product_id": [10], "label": [1.0]})
    return product, aisle, reorder, gmm, labels


def test_load_champion_evaluation_uses_the_champion_exports(tmp_path, mocker):
    champion = _champion_pointer(tmp_path, "run-old")
    captured = {}

    def read_outputs(paths):
        captured["paths"] = paths
        product, aisle, reorder, gmm, _labels = _outputs_for(4)
        return product, aisle, reorder, gmm

    mocker.patch(
        "instacart_rnn.stacking.stacking_train.read_stacking_outputs",
        side_effect=read_outputs,
    )
    _product, _aisle, _reorder, _gmm, labels = _outputs_for(4)
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.read_frame",
        return_value=labels,
    )

    frame, columns = load_champion_evaluation(
        champion,
        "gs://gold/product_training_data_evaluation",
        np.array([4]),
        runs_root=str(tmp_path),
        evaluation_timeline="t1",
        evaluation_inputs={name: f"{name}.parquet" for name in REQUIRED_MODELS},
    )

    assert columns == ["product_logit", "aisle_logit"]
    assert frame["user_id"].tolist() == [4]
    assert set(captured["paths"]) == set(REQUIRED_MODELS)
    assert all(path.endswith("evaluation") for path in captured["paths"].values())
    assert all("run-old" in path for path in captured["paths"].values())


def test_load_champion_evaluation_rejects_different_users(tmp_path, mocker):
    champion = _champion_pointer(tmp_path, "run-old")
    product, aisle, reorder, gmm, labels = _outputs_for(4)
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.read_stacking_outputs",
        return_value=(product, aisle, reorder, gmm),
    )
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.read_frame",
        return_value=labels,
    )

    with pytest.raises(ValueError, match="evaluation users"):
        load_champion_evaluation(
            champion,
            "gs://gold/product_training_data_evaluation",
            np.array([9]),
            runs_root=str(tmp_path),
            evaluation_timeline="t1",
            evaluation_inputs={name: f"{name}.parquet" for name in REQUIRED_MODELS},
        )


def test_load_champion_evaluation_rescores_an_earlier_timeline(tmp_path, mocker):
    champion = _champion_pointer(tmp_path, "run-old")
    for name in REQUIRED_MODELS:
        config = tmp_path / name / "run-old" / "config.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({"lstm_size": 256}), encoding="utf-8")
    exported = []

    def export(**kwargs):
        exported.append(kwargs)

    mocker.patch(
        "instacart_rnn.stacking.stacking_train._export_checkpoint",
        side_effect=export,
    )

    def read_outputs(paths):
        exported.append(paths)
        product, aisle, reorder, gmm, _labels = _outputs_for(4)
        return product, aisle, reorder, gmm

    mocker.patch(
        "instacart_rnn.stacking.stacking_train.read_stacking_outputs",
        side_effect=read_outputs,
    )
    _product, _aisle, _reorder, _gmm, labels = _outputs_for(4)
    mocker.patch(
        "instacart_rnn.stacking.stacking_train.read_frame",
        return_value=labels,
    )
    inputs = {
        name: f"gs://gold/training/curated/t2/{name}_training_data_evaluation"
        for name in REQUIRED_MODELS
    }

    frame, _columns = load_champion_evaluation(
        champion,
        inputs["product"],
        np.array([4]),
        runs_root=str(tmp_path),
        evaluation_timeline="t2",
        evaluation_inputs=inputs,
    )

    calls = exported[:-1]
    scored_paths = exported[-1]
    assert frame["user_id"].tolist() == [4]
    assert {call["model_name"] for call in calls} == set(REQUIRED_MODELS)
    for call in calls:
        name = call["model_name"]
        output_path = call["output_path"].replace("\\", "/")
        assert call["input_path"] == inputs[name]
        assert call["checkpoint_path"] == champion.models[name].best_checkpoint
        assert call["lstm_size"] == 256
        assert output_path.endswith(f"rescore/t2/stack-old/{name}")
        assert scored_paths[name] == call["output_path"]


def test_evaluation_input_paths_follow_this_timeline(tmp_path):
    stack_dir = tmp_path / "stack" / "stack-new"
    stack_dir.mkdir(parents=True)
    models = {}
    for name in REQUIRED_MODELS:
        filename = (
            "reorder_size_training_data_train"
            if "reorder_size" in name
            else f"{name}_training_data_train"
        )
        models[name] = {
            "train_path": f"gs://gold/training/curated/t2/{filename}",
        }
    (stack_dir / "selected_artifacts.json").write_text(
        json.dumps({"models": models}),
        encoding="utf-8",
    )

    paths = _evaluation_input_paths(
        str(stack_dir),
        "gs://gold/training/curated/t2/product_training_data_evaluation",
    )

    assert paths["product"].endswith("product_training_data_evaluation")
    assert "/t2/" in paths["aisle"]
    assert paths["aisle"].endswith("aisle_training_data_evaluation")
    assert paths["reorder_size_rnn"].endswith("reorder_size_training_data_evaluation")
    assert paths["reorder_size_gmm"].endswith("reorder_size_training_data_evaluation")
