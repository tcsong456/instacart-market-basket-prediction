import pandas as pd
import numpy as np
from scripts.build_sample_dataset import sample_users

#1111
def test_same_seed_produce_same_users(tiny_fake_testset_csv):
    orders = pd.read_csv(tiny_fake_testset_csv / "orders.csv")

    users_1 = sample_users(orders, 2, 17690)
    users_2 = sample_users(orders, 2, 17690)

    return np.array_equal(users_1, users_2)


def test_sample_users_are_unique(tiny_fake_testset_csv):
    orders = pd.read_csv(tiny_fake_testset_csv / "orders.csv")

    users = sample_users(orders, 2, 17690)

    return users.is_unique
