import numpy as np
import pandas as pd
import pytest

from instacart_rnn.stacking.stacking_train import (
    NONE_TOKEN,
    basket_f1,
    choose_prefix,
    evaluation_artifact_path,
    evaluation_label_path,
    join_labeled_features,
    mean_order_f1,
    predicted_basket,
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
