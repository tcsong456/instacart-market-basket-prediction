import argparse
import shutil
from pathlib import Path

import pandas as pd

LOOKUP_FILES = ["aisles.csv", "departments.csv", "products.csv"]
ORDER_PRODUCT_FILES = ["order_products__prior.csv", "order_products__train.csv"]


def sample_users(orders: pd.DataFrame, sample_n: int, seed: int = 42) -> pd.Series:
    unique_users = orders["user_id"].nunique()
    if sample_n > unique_users:
        raise ValueError(
            f"sample_n={sample_n} is greater than number of unique users "
            f"available: {unique_users}"
        )
    return orders["user_id"].drop_duplicates().sample(n=sample_n, random_state=seed)


def filter_orders_by_users(orders: pd.DataFrame, user_ids: pd.Series) -> pd.DataFrame:
    return orders[orders["user_id"].isin(user_ids)]


def write_filtered_orders(
    input_dir: Path,
    output_dir: Path,
    chunk_size: int,
    filename: str,
    order_ids: set[int],
) -> None:
    first_chunk = True
    write_any_rows = False
    for chunk in pd.read_csv(input_dir / filename, chunksize=chunk_size):
        filtered_chunk = chunk[chunk["order_id"].isin(order_ids)]
        if not filtered_chunk.empty:
            filtered_chunk.to_csv(
                output_dir / filename,
                index=False,
                header=first_chunk,
                mode="w" if first_chunk else "a",
            )
            first_chunk = False
            write_any_rows = True

    if not write_any_rows:
        columns = pd.read_csv(input_dir / filename, nrows=0).columns
        pd.DataFrame(columns=columns).to_csv(output_dir / filename, index=False)


def copy_lookup_files(input_dir: Path, output_dir: Path) -> None:
    for filename in LOOKUP_FILES:
        shutil.copy2(input_dir / filename, output_dir / filename)


def build_sample_dataset(
    input_dir: Path, output_dir: Path, sample_n: int, chunk_size: int, seed: int = 42
) -> None:
    """
    Create sampled datasets for local development and tests. Select a subset
    of users and all its corresponding orders as well as order products, copy
    'look_up_files' to produce a self-contained sample dataset folder

    Args:
        raw_dir: Directory containing the original Instacart CSV files.
        sample_dir: Output directory for the sampled dataset.
        chunk_size: Number of rows to process per chunk when filtering
            large order product files.
        seed: Random seed used for reproducible user sampling.
        sample_n: Number of users to sample.

    Returns:
        None.
    """

    orders = pd.read_csv(input_dir / "orders.csv")

    sampled_user_ids = sample_users(orders, sample_n, seed)
    filtered_orders = filter_orders_by_users(orders, sampled_user_ids)
    filtered_orders.to_csv(output_dir / "orders.csv", index=False)

    sampled_order_ids = set(filtered_orders["order_id"])
    for filename in ORDER_PRODUCT_FILES:
        write_filtered_orders(
            input_dir=input_dir,
            output_dir=output_dir,
            chunk_size=chunk_size,
            filename=filename,
            order_ids=sampled_order_ids,
        )

    copy_lookup_files(input_dir, output_dir)


def parse_args():  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-n", type=int, default=100)
    parser.add_argument("--chunk-sie", type=int, default=1e6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":  # pragma: no cover
    args = parse_args()
    build_sample_dataset(
        raw_dir=args.input_dir,
        sample_dir=args.output_dir,
        chunk_size=args.chunk_size,
        seed=args.seed,
        sample_n=args.sample_n,
    )
