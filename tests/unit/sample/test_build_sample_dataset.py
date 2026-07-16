import numpy as np
import pandas as pd
import pytest

from instacart_etl.sample.build_sample_dataset import (
    filter_orders_by_users,
    sample_users,
)


def test_same_seed_produce_same_users(tiny_fake_testset_csv):
    orders = pd.read_csv(tiny_fake_testset_csv / "orders.csv")

    users_1 = sample_users(orders, 2, 17690)
    users_2 = sample_users(orders, 2, 17690)

    assert np.array_equal(users_1, users_2)


def test_sample_users_are_unique(tiny_fake_testset_csv):
    orders = pd.read_csv(tiny_fake_testset_csv / "orders.csv")

    users = sample_users(orders, 3, 17690)

    assert users.is_unique
    assert users.nunique() == 3


def test_sample_n_larger_than_unique_users(tiny_fake_testset_csv):
    orders = pd.read_csv(tiny_fake_testset_csv / "orders.csv")

    with pytest.raises(ValueError, match="is greater than number of unique users"):
        sample_users(orders, 99999)


def test_filter_orders_only_contain_sampled_users(tiny_fake_testset_csv):
    orders = pd.read_csv(tiny_fake_testset_csv / "orders.csv")

    user_ids = pd.Series([10, 30])
    filtered_orders = filter_orders_by_users(orders, user_ids)

    assert set(filtered_orders["user_id"]) == {10, 30}
    assert len(filtered_orders) == 4
    assert filtered_orders["order_id"].tolist() == [1, 2, 5, 6]
