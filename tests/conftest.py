import pandas as pd
import pytest


@pytest.fixture
def raw_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


@pytest.fixture
def sample_dir(tmp_path):
    sample_dir = tmp_path / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


@pytest.fixture
def tiny_fake_testset_csv(raw_dir):
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3, 4, 5, 6],
            "user_id": [10, 10, 20, 20, 30, 30],
            "eval_set": ["prior", "train", "prior", "train", "prior", "test"],
            "order_number": [1, 2, 1, 2, 1, 2],
            "order_dow": [1, 2, 1, 2, 1, 2],
            "order_hour_of_day": [10, 11, 10, 11, 10, 11],
            "days_since_prior_order": [None, 7, None, 5, None, 3],
        }
    )

    prior = pd.DataFrame(
        {
            "order_id": [1, 3, 5],
            "product_id": [101, 201, 301],
            "add_to_cart_order": [1, 1, 1],
            "reordered": [0, 0, 0],
        }
    )

    train = pd.DataFrame(
        {
            "order_id": [2, 4],
            "product_id": [102, 202],
            "add_to_cart_order": [1, 1],
            "reordered": [1, 1],
        }
    )

    products = pd.DataFrame(
        {
            "product_id": [101, 102, 201, 202, 301],
            "product_name": ["a", "b", "c", "d", "e"],
            "aisle_id": [1, 1, 2, 2, 3],
            "department_id": [10, 10, 20, 20, 30],
        }
    )

    aisles = pd.DataFrame(
        {
            "aisle_id": [1, 2, 3],
            "aisle": ["fresh", "dairy", "snacks"],
        }
    )

    departments = pd.DataFrame(
        {
            "department_id": [10, 20, 30],
            "department": ["produce", "frozen", "pantry"],
        }
    )

    orders.to_csv(raw_dir / "orders.csv", index=False)
    prior.to_csv(raw_dir / "order_products__prior.csv", index=False)
    train.to_csv(raw_dir / "order_products__train.csv", index=False)
    products.to_csv(raw_dir / "products.csv", index=False)
    aisles.to_csv(raw_dir / "aisles.csv", index=False)
    departments.to_csv(raw_dir / "departments.csv", index=False)

    return raw_dir
