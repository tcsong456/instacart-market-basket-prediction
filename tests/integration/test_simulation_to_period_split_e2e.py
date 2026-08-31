from pathlib import Path

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.jobs.create_period_split_data_job import (
    run_order_products_split_job,
)
from instacart_etl_rnn.jobs.create_user_split_data_job import (
    run_simulation_split_job,
)
from instacart_etl_rnn.silver.create_order_products import build_order_products

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
)

BASE_USER = 1
STACKING_USER = 41
HOLDOUT_USER = 3
NEW_USER = 8


def _order_id(user_id: int, order_number: int) -> int:
    return user_id * 1000 + order_number


def _write_bronze_inputs(spark, bronze_path):
    order_rows = []
    prior_product_rows = []
    train_product_rows = []

    users = {
        BASE_USER: 6,
        STACKING_USER: 6,
        HOLDOUT_USER: 6,
        NEW_USER: 4,
    }

    product_by_user = {
        BASE_USER: 10,
        STACKING_USER: 20,
        HOLDOUT_USER: 30,
        NEW_USER: 40,
    }

    for user_id, order_history in users.items():
        for order_number in range(1, order_history + 1):
            oid = _order_id(user_id, order_number)
            eval_set = "train" if order_number == order_history else "prior"
            days = None if order_number == 1 else float(order_number)
            order_rows.append(
                (
                    oid,
                    user_id,
                    eval_set,
                    order_number,
                    order_number % 7,
                    10 + order_number,
                    days,
                )
            )
            product_row = (
                oid,
                product_by_user[user_id],
                1,
                0 if order_number == 1 else 1,
            )
            if eval_set == "prior":
                prior_product_rows.append(product_row)
            else:
                train_product_rows.append(product_row)

    orders = spark.createDataFrame(
        order_rows,
        """
        order_id int,
        user_id int,
        eval_set string,
        order_number int,
        order_dow int,
        order_hour_of_day int,
        days_since_prior_order double
        """,
    )
    write_parquet(bronze_path / "orders", orders)

    prior = spark.createDataFrame(
        prior_product_rows,
        """
        order_id int,
        product_id int,
        add_to_cart_order int,
        reordered int
        """,
    )
    train = spark.createDataFrame(train_product_rows, schema=prior.schema)
    write_parquet(bronze_path / "order_products__prior", prior)
    write_parquet(bronze_path / "order_products__train", train)

    products = spark.createDataFrame(
        [
            (10, "Banana", 1, 1),
            (20, "Milk", 2, 2),
            (30, "Bread", 3, 3),
            (40, "Eggs", 4, 4),
        ],
        """
        product_id int,
        product_name string,
        aisle_id int,
        department_id int
        """,
    )
    write_parquet(bronze_path / "products", products)


def _order_ids(df):
    return {row.order_id for row in df.select("order_id").collect()}


def _user_ids(df):
    return {row.user_id for row in df.select("user_id").distinct().collect()}


def test_simulation_silver_period_split_chain_end_to_end(spark, tmp_path):
    bronze_path = tmp_path / "bronze"
    simulation_path = tmp_path / "simulation" / "t2"
    silver_path = tmp_path / "silver"
    split_path = tmp_path / "period_split"

    _write_bronze_inputs(spark, bronze_path)

    run_simulation_split_job(
        spark=spark,
        input_path=str(bronze_path),
        output_path=str(simulation_path),
        contract_path=str(CONTRACT_PATH),
        period="t2",
    )

    available_orders = read_parquet(
        simulation_path / "available_orders",
        spark,
    )
    assert available_orders.count() == 22

    cohorts = {
        row.user_id: (
            row.user_cohort,
            row.development_split,
            row.arrival_period,
        )
        for row in (
            available_orders.select(
                "user_id",
                "user_cohort",
                "development_split",
                "arrival_period",
            )
            .distinct()
            .collect()
        )
    }

    assert cohorts[BASE_USER] == ("established", "base_train", None)
    assert cohorts[STACKING_USER] == ("established", "stacking_train", None)
    assert cohorts[HOLDOUT_USER] == ("final_holdout", None, None)
    assert cohorts[NEW_USER] == ("new_user", None, "t2")

    build_order_products(
        spark=spark,
        input_path=str(bronze_path),
        output_path=str(silver_path),
        contract_path=str(CONTRACT_PATH),
        order_path=str(simulation_path),
    )

    silver = read_parquet(silver_path / "order_products", spark)
    assert silver.count() == 22
    assert _user_ids(silver) == {
        BASE_USER,
        STACKING_USER,
        HOLDOUT_USER,
        NEW_USER,
    }

    run_order_products_split_job(
        spark=spark,
        input_path=str(silver_path),
        output_path=str(split_path),
        mode="base_train",
        period="t2",
        contract_path=str(CONTRACT_PATH),
    )

    base_dir = split_path / "t2"
    base_train = read_parquet(base_dir / "order_products_train", spark)
    base_validation = read_parquet(base_dir / "order_products_validation", spark)
    base_evaluation = read_parquet(base_dir / "order_products_evaluation", spark)

    assert STACKING_USER not in _user_ids(base_train)
    assert STACKING_USER not in _user_ids(base_validation)
    assert STACKING_USER not in _user_ids(base_evaluation)

    assert _order_ids(base_train) == {_order_id(BASE_USER, n) for n in range(1, 6)} | {
        _order_id(NEW_USER, n) for n in range(1, 4)
    }
    assert _order_ids(base_validation) == {
        _order_id(BASE_USER, n) for n in range(1, 7)
    } | {_order_id(NEW_USER, n) for n in range(1, 5)}
    assert _order_ids(base_evaluation) == {
        _order_id(HOLDOUT_USER, n) for n in range(1, 7)
    }
    assert _user_ids(base_evaluation) == {HOLDOUT_USER}

    run_order_products_split_job(
        spark=spark,
        input_path=str(silver_path),
        output_path=str(split_path),
        mode="stacking_train",
        period="t2",
        contract_path=str(CONTRACT_PATH),
    )

    stacking_dir = split_path / "stacking_train"
    stacking_train = read_parquet(
        stacking_dir / "order_products_train",
        spark,
    )
    stacking_validation = read_parquet(
        stacking_dir / "order_products_validation",
        spark,
    )

    assert _user_ids(stacking_train) == {STACKING_USER}
    assert _user_ids(stacking_validation) == {STACKING_USER}
    assert _order_ids(stacking_train) == {
        _order_id(STACKING_USER, n) for n in range(1, 6)
    }
    assert _order_ids(stacking_validation) == {
        _order_id(STACKING_USER, n) for n in range(1, 7)
    }
    assert not (stacking_dir / "order_products_evaluation").exists()
