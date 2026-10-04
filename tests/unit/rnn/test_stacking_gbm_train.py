import json

import numpy as np
import pandas as pd
import pytest

from instacart_rnn.stacking.stacking_train import (
    NONE_TOKEN,
    basket_f1,
    choose_prefix,
    evaluation_artifact_path,
    evaluation_label_path,
    execute_backend_run,
    join_labeled_features,
    mean_order_f1,
    predicted_basket,
    resolve_backend_run,
    resolve_stack_directory,
    save_booster,
    train_booster,
    true_basket,
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
            "model_path": "gs://runs/stack/stack-1/model.txt",
        },
    )

    run_root = tmp_path / "stacking_gbm" / "run-1"
    run_payload = json.loads((run_root / "run.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_root / "metrics.json").read_text(encoding="utf-8"))

    assert result["evaluation_f1"] == 0.37
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
