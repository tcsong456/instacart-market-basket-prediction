import numpy as np
import pandas as pd
import pytest

from instacart_etl.sample.build_sample_dataset import (
    LOOKUP_FILES,
    copy_lookup_files,
    filter_orders_by_users,
    sample_users,
    write_filtered_orders,
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
    assert len(users) == 3


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


def test_filter_orders_contain_orders_correctly(tiny_fake_testset_csv, sample_dir):
    order_ids = {1, 5}
    write_filtered_orders(
        input_dir=tiny_fake_testset_csv,
        output_dir=sample_dir,
        chunk_size=2,
        filename="order_products__prior.csv",
        order_ids=order_ids,
    )

    result = pd.read_csv(sample_dir / "order_products__prior.csv")

    assert set(result["order_id"]) == order_ids
    assert len(result) == 2
    assert list(result.columns) == [
        "order_id",
        "product_id",
        "add_to_cart_order",
        "reordered",
    ]


def test_filtered_orders_empty_output(tiny_fake_testset_csv, sample_dir):
    write_filtered_orders(
        input_dir=tiny_fake_testset_csv,
        output_dir=sample_dir,
        chunk_size=1,
        filename="order_products__prior.csv",
        order_ids={999999},
    )

    result = pd.read_csv(sample_dir / "order_products__prior.csv")

    assert len(result) == 0
    assert list(result.columns) == [
        "order_id",
        "product_id",
        "add_to_cart_order",
        "reordered",
    ]


def test_filter_by_larger_chunksize_than_total_row(tiny_fake_testset_csv, sample_dir):
    write_filtered_orders(
        input_dir=tiny_fake_testset_csv,
        output_dir=sample_dir,
        chunk_size=4,
        filename="order_products__prior.csv",
        order_ids={1, 3, 5},
    )

    result = pd.read_csv(sample_dir / "order_products__prior.csv")
    assert set(result["order_id"]) == {1, 3, 5}


def test_copy_lookup_files(tiny_fake_testset_csv, sample_dir):
    copy_lookup_files(tiny_fake_testset_csv, sample_dir)

    for filename in LOOKUP_FILES:
        assert (tiny_fake_testset_csv / filename).read_bytes() == (
            sample_dir / filename
        ).read_bytes(), f"{filename} is not copied correctly"
