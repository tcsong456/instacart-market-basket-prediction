import pandas as pd

from instacart_etl.sample.build_sample_dataset import build_sample_dataset


def _sort_product_order(df: pd.DataFrame):
    return df.sort_values(["order_id", "add_to_cart_order"]).reset_index(drop=True)


def test_sample_dataset_pipeline(tiny_fake_testset_csv, sample_dir):
    sample_n = 2
    build_sample_dataset(
        input_dir=tiny_fake_testset_csv,
        output_dir=sample_dir,
        sample_n=sample_n,
        chunk_size=2,
        seed=10817,
    )

    source_orders = pd.read_csv(tiny_fake_testset_csv / "orders.csv")
    source_prior = pd.read_csv(tiny_fake_testset_csv / "order_products__prior.csv")
    source_train = pd.read_csv(tiny_fake_testset_csv / "order_products__train.csv")
    actual_orders = pd.read_csv(sample_dir / "orders.csv")
    actual_prior = pd.read_csv(sample_dir / "order_products__prior.csv")
    actual_train = pd.read_csv(sample_dir / "order_products__train.csv")

    assert actual_orders["user_id"].nunique() == sample_n

    sampled_order_ids = set(actual_orders["order_id"])
    sampled_user_ids = set(actual_orders["user_id"])

    expected_orders = source_orders[source_orders["user_id"].isin(sampled_user_ids)]
    expected_orders = expected_orders[actual_orders.columns]
    pd.testing.assert_frame_equal(
        expected_orders.sort_values("order_id").reset_index(drop=True),
        actual_orders.sort_values("order_id").reset_index(drop=True),
    )

    expected_prior = source_prior[source_prior["order_id"].isin(sampled_order_ids)]
    expected_prior = expected_prior[actual_prior.columns]
    pd.testing.assert_frame_equal(
        _sort_product_order(expected_prior), _sort_product_order(actual_prior)
    )

    expected_train = source_train[source_train["order_id"].isin(sampled_order_ids)]
    expected_train = expected_train[actual_train.columns]
    pd.testing.assert_frame_equal(
        _sort_product_order(expected_train), _sort_product_order(actual_train)
    )
