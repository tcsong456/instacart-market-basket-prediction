import pandas as pd

def sample_users(orders: pd.DataFrame, sample_n: int, seed: int):
    unique_users = orders['user_id'].nunique()
    if sample_n > unique_users:
        raise ValueError(
            f"sample_n={sample_n} is greater than number of unique users "
            f"available: {unique_users}"
        )
    return orders['user_id'].drop_duplicates().sample(n=sample_n, random_state=seed)


def filter_orders_by_users(
    orders: pd.DataFrame, user_ids: pd.Series
) -> pd.DataFrame:
    return orders[orders['user_id'].isin(user_ids)]

