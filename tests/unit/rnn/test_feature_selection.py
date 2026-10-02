import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from instacart_rnn.stacking.feature_selection import (
    binary_log_loss,
    build_feature_frame,
    build_versions,
    choose_version,
    compact_columns,
    load_selected_artifacts,
    out_of_fold_probabilities,
    project_state_block,
    read_model_frame,
    select_feature_set,
    stacking_label_path,
)


def test_stacking_label_path_inserts_the_stacking_train_directory():
    path = stacking_label_path(
        "gs://gold/training/sample/t1/product_training_data_train"
    )

    assert path == (
        "gs://gold/training/sample/t1/stacking_train/product_training_data_train"
    )


def test_stacking_label_path_rejects_a_non_train_file():
    with pytest.raises(ValueError, match="_training_data_train"):
        stacking_label_path(
            "gs://gold/training/sample/t1/product_training_data_validation"
        )


def test_build_feature_frame_keeps_each_user_in_one_fold():
    product = _frame(
        ["user_id", "product_id", "aisle_id", "product_logit"],
        [[1, 10, 100, 0.1], [1, 11, 100, 0.2], [2, 20, 200, 0.3]],
    )
    aisle = _frame(
        ["user_id", "aisle_id", "aisle_logit"],
        [[1, 100, 1.0], [2, 200, 2.0]],
    )
    reorder = _frame(
        ["user_id", "reorder_prediction"],
        [[1, 3.0], [2, 4.0]],
    )
    gmm = _frame(["user_id", "gmm_nll_0"], [[1, 8.0], [2, 9.0]])
    labels = _frame(
        ["user_id", "product_id", "label"],
        [[1, 10, 1], [1, 11, 0], [2, 20, 1]],
    )
    folds = _frame(["user_id", "fold"], [[1, 0], [2, 1]])

    frame = build_feature_frame(
        product=product,
        aisle=aisle,
        reorder=reorder,
        gmm=gmm,
        labels=labels,
        folds=folds,
    )

    user_one = frame.loc[frame["user_id"] == 1]
    assert set(user_one["fold"]) == {0}
    assert set(user_one["aisle_logit"]) == {1.0}
    assert set(user_one["reorder_prediction"]) == {3.0}
    assert set(user_one["gmm_nll_0"]) == {8.0}
    assert list(user_one["label"]) == [1, 0]


def test_build_feature_frame_rejects_a_user_without_a_fold():
    product = _frame(
        ["user_id", "product_id", "aisle_id"],
        [[1, 10, 100]],
    )
    aisle = _frame(["user_id", "aisle_id"], [[1, 100]])
    reorder = _frame(["user_id"], [[1]])
    gmm = _frame(["user_id"], [[1]])
    labels = _frame(["user_id", "product_id", "label"], [[1, 10, 1]])
    folds = _frame(["user_id", "fold"], [[2, 0]])

    with pytest.raises(ValueError, match="missing a user fold"):
        build_feature_frame(
            product=product,
            aisle=aisle,
            reorder=reorder,
            gmm=gmm,
            labels=labels,
            folds=folds,
        )


def test_out_of_fold_training_excludes_the_held_out_fold():
    frame = _frame(
        ["user_id", "fold", "label", "user_code"],
        [[1, 0, 1, 1], [2, 0, 0, 2], [3, 1, 1, 3], [4, 1, 0, 4]],
    )
    seen: list[tuple[set[float], set[float]]] = []

    def matrices(train_mask, test_mask):
        values = frame[["user_code"]].to_numpy(dtype=float)
        return values[train_mask], values[test_mask]

    def fit_predict(train_x, train_y, test_x):
        seen.append((set(train_x[:, 0]), set(test_x[:, 0])))
        return np.full(len(test_x), 0.5)

    out_of_fold_probabilities(frame, matrices, fit_predict)

    assert seen == [({3.0, 4.0}, {1.0, 2.0}), ({1.0, 2.0}, {3.0, 4.0})]


def test_select_feature_set_keeps_version_1_when_scores_match():
    frame = _version_frame(signal=1.0)

    def fit_predict(train_x, train_y, test_x):
        return test_x[:, 0]

    winner, results = select_feature_set(frame, fit_predict)

    assert winner == "version_1"
    assert [item["feature_set"] for item in results] == [
        "version_1",
        "version_2",
        "version_3",
    ]
    assert results[0]["n_features"] < results[2]["n_features"]


@pytest.mark.parametrize(
    ("losses", "expected"),
    [
        ((0.40, 0.399, 0.398), "version_1"),
        ((0.50, 0.395, 0.39), "version_2"),
        ((0.50, 0.40, 0.39), "version_3"),
        ((0.50, 0.42, 0.30), "version_3"),
        ((0.30, 0.30, 0.30), "version_1"),
    ],
)
def test_choose_version_keeps_the_smallest_competitive_set(losses, expected):
    results = [
        {"feature_set": f"version_{index}", "complexity": index, "oof_log_loss": loss}
        for index, loss in enumerate(losses, start=1)
    ]

    assert choose_version(results) == expected


def test_project_state_block_uses_the_training_axis_only():
    values = np.array(
        [
            [2.0, 0.0],
            [4.0, 0.0],
            [0.0, 9.0],
        ]
    )
    keys = np.array([[1], [2], [3]])
    train_mask = np.array([True, True, False])
    test_mask = np.array([False, False, True])

    train_projected, test_projected, count = project_state_block(
        values,
        keys,
        train_mask,
        test_mask,
    )

    assert count == 1
    assert abs(test_projected[0, 0]) == pytest.approx(3.0)
    assert abs(train_projected[0, 0] - train_projected[1, 0]) == pytest.approx(2.0)


def test_project_state_block_fits_each_repeated_key_once():
    values = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [10.0, 0.0],
            [5.0, 0.0],
        ]
    )
    keys = np.array([[1], [1], [1], [2], [3]])
    train_mask = np.array([True, True, True, True, False])
    test_mask = np.array([False, False, False, False, True])

    _, test_projected, _ = project_state_block(
        values,
        keys,
        train_mask,
        test_mask,
    )

    assert test_projected[0, 0] == pytest.approx(0.0)


def test_project_state_block_rejects_a_single_training_key():
    values = np.array([[1.0, 2.0], [1.0, 2.0]])
    keys = np.array([[1], [1]])
    train_mask = np.array([True, True])
    test_mask = np.array([False, False])

    with pytest.raises(ValueError, match="at least two finite"):
        project_state_block(values, keys, train_mask, test_mask)


def test_binary_log_loss_of_a_confident_correct_prediction_is_near_zero():
    loss = binary_log_loss(np.array([1.0, 0.0]), np.array([1.0, 0.0]))

    assert loss == pytest.approx(0.0, abs=1e-4)


def test_read_model_frame_flattens_lists_and_renames_scalars(tmp_path):
    states = pa.FixedSizeListArray.from_arrays(
        pa.array([0.25, 0.75, 1.5, 2.5], type=pa.float32()),
        2,
    )
    pq.write_table(
        pa.table(
            {
                "user_id": pa.array([1, 2], type=pa.int64()),
                "product_id": pa.array([10, 20], type=pa.int64()),
                "final_logits": pa.array([0.1, 0.2], type=pa.float32()),
                "final_states": states,
            }
        ),
        tmp_path / "part.parquet",
    )

    frame = read_model_frame(
        str(tmp_path),
        ["user_id", "product_id"],
        {"final_logits": "product_logit"},
        {"final_states": "product_state_"},
    )

    assert list(frame["product_logit"]) == pytest.approx([0.1, 0.2])
    assert list(frame["product_state_0"]) == [0.25, 1.5]
    assert list(frame["product_state_1"]) == [0.75, 2.5]


def test_version_builders_put_compact_columns_ahead_of_hidden_states():
    frame = _version_frame(signal=0.0)
    versions = build_versions(frame)
    train_mask = np.array([True, True, False, False])
    test_mask = ~train_mask

    version_1 = versions[0][2](train_mask, test_mask)[0]
    version_3 = versions[2][2](train_mask, test_mask)[0]

    assert [name for name, _, _ in versions] == [
        "version_1",
        "version_2",
        "version_3",
    ]
    assert list(compact_columns(frame))[0] == "product_logit"
    np.testing.assert_allclose(version_3[:, : version_1.shape[1]], version_1)
    assert version_3.shape[1] > version_1.shape[1]


def test_load_selected_artifacts_requires_every_model(tmp_path):
    path = tmp_path / "selected_artifacts.json"
    path.write_text(json.dumps({"models": {"product": {}}}), encoding="utf-8")

    with pytest.raises(ValueError, match="aisle"):
        load_selected_artifacts(str(path))


def _frame(columns: list[str], rows: list[list]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=columns)


def _version_frame(signal: float) -> pd.DataFrame:
    rows = []
    for index, (user_id, fold, label) in enumerate(
        ((1, 0, 1), (2, 0, 0), (3, 1, 1), (4, 1, 0))
    ):
        row = {
            "user_id": user_id,
            "product_id": 10 + index,
            "aisle_id": 100 + index,
            "fold": fold,
            "label": label,
            "product_logit": label if signal else 0.5,
            "aisle_logit": 0.5,
            "reorder_prediction": 0.5,
            "gmm_nll_0": 0.5,
            "gmm_mode_size": 0.5,
            "gmm_candidate_expected_size": 0.5,
            "gmm_candidate_entropy": 0.5,
            "gmm_expected_size": 0.5,
            "gmm_candidate_nll_0": 0.5,
            "product_state_0": float(index),
            "aisle_state_0": float(index),
            "reorder_state_0": float(user_id),
            "gmm_state_0": float(user_id),
        }
        rows.append(row)
    return pd.DataFrame(rows)
