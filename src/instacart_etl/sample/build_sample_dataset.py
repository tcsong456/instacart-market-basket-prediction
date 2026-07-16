from pathlib import Path

import pandas as pd


def sample_users(orders: pd.DataFrame, sample_n: int, seed: int = 42):
    unique_users = orders["user_id"].nunique()
    if sample_n > unique_users:
        raise ValueError(
            f"sample_n={sample_n} is greater than number of unique users "
            f"available: {unique_users}"
        )
    return orders["user_id"].drop_duplicates().sample(n=sample_n, random_state=seed)


def filter_orders_by_users(orders: pd.DataFrame, user_ids: pd.Series) -> pd.DataFrame:
    return orders[orders["user_id"].isin(user_ids)]


def build_sample_dataset(
    input_dir: Path, output_dir: Path, sample_n: int, seed: int = 42
) -> None:
    orders = pd.read_csv(input_dir / "orders.csv")

    sampled_user_ids = sample_users(orders, sample_n, seed)
    filtered_orders = filter_orders_by_users(orders, sampled_user_ids)
    filtered_orders.to_csv(output_dir / "orders.csv", index=False)
