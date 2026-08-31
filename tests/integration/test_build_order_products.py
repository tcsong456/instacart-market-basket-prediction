from pathlib import Path

import pytest
from pyspark.sql import functions as F

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.jobs.create_user_split_data_job import (
    run_simulation_split_job,
)
from instacart_etl_rnn.silver.create_order_products import build_order_products
from instacart_etl_rnn.validation.exceptions import DataValidationError

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
)


def _write_bronze_inputs(spark, bronze_path):
    """Write bronze orders / order_products / products for the e2e.

    Users:
      101 - 6 orders -> established or final_holdout (hash-dependent)
      202 - 4 orders -> new_user
    """

    orders = spark.createDataFrame(
        [
            (1011, 101, "prior", 1, 0, 10, None),
            (1012, 101, "prior", 2, 1, 11, 5.0),
            (1013, 101, "prior", 3, 2, 12, 6.0),
            (1014, 101, "prior", 4, 3, 13, 7.0),
            (1015, 101, "prior", 5, 4, 14, 8.0),
            (1016, 101, "train", 6, 5, 15, 9.0),
            (2021, 202, "prior", 1, 0, 10, None),
            (2022, 202, "prior", 2, 1, 11, 5.0),
            (2023, 202, "prior", 3, 2, 12, 6.0),
            (2024, 202, "train", 4, 3, 13, 7.0),
        ],
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

    order_products_prior = spark.createDataFrame(
        [
            (1011, 10, 1, 0),
            (1012, 20, 1, 0),
            (1013, 10, 1, 1),
            (1014, 20, 1, 1),
            (1015, 10, 1, 1),
            (2021, 30, 1, 0),
            (2022, 40, 1, 0),
            (2023, 30, 1, 1),
        ],
        """
        order_id int,
        product_id int,
        add_to_cart_order int,
        reordered int
        """,
    )
    order_products_train = spark.createDataFrame(
        [
            (1016, 20, 1, 1),
            (2024, 40, 1, 1),
        ],
        schema=order_products_prior.schema,
    )
    write_parquet(bronze_path / "order_products__prior", order_products_prior)
    write_parquet(bronze_path / "order_products__train", order_products_train)

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


def test_build_order_products_from_real_simulation_output(
    spark,
    tmp_path,
):
    bronze_path = tmp_path / "bronze"
    simulation_path = tmp_path / "simulation" / "t2"
    silver_path = tmp_path / "silver"

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
    assert available_orders.count() == 10

    build_order_products(
        spark=spark,
        input_path=str(bronze_path),
        output_path=str(silver_path),
        contract_path=str(CONTRACT_PATH),
        order_path=str(simulation_path),
    )

    result = read_parquet(silver_path / "order_products", spark)

    actual = {
        (row.order_id, row.product_id): row.asDict(recursive=True)
        for row in result.collect()
    }

    assert set(actual) == {
        (1011, 10),
        (1012, 20),
        (1013, 10),
        (1014, 20),
        (1015, 10),
        (1016, 20),
        (2021, 30),
        (2022, 40),
        (2023, 30),
        (2024, 40),
    }

    assert actual[(1011, 10)]["product_name"] == "Banana"
    assert actual[(1011, 10)]["aisle_id"] == 1
    assert actual[(1011, 10)]["department_id"] == 1
    assert actual[(1011, 10)]["days_since_prior_order"] == -1.0

    assert actual[(2022, 40)]["product_name"] == "Eggs"
    assert actual[(2022, 40)]["aisle_id"] == 4
    assert actual[(2022, 40)]["department_id"] == 4
    assert actual[(2022, 40)]["days_since_prior_order"] == 5.0

    user_101 = result.filter(F.col("user_id") == 101).orderBy("order_number").collect()
    assert len(user_101) == 6

    cohort_101 = user_101[0].user_cohort
    assert cohort_101 in {"established", "final_holdout"}
    assert {row.user_cohort for row in user_101} == {cohort_101}
    assert {row.current_period for row in user_101} == {"t2"}
    assert {row.order_history for row in user_101} == {6}

    flags_101 = {
        row.order_number: (
            row.is_train_available,
            row.is_validation_available,
            row.is_evaluation_available,
        )
        for row in user_101
    }

    if cohort_101 == "established":
        assert {row.development_split for row in user_101} <= {
            "base_train",
            "stacking_train",
        }
        assert {row.arrival_period for row in user_101} == {None}
        assert flags_101 == {
            1: (True, True, False),
            2: (True, True, False),
            3: (True, True, False),
            4: (True, True, False),
            5: (True, True, False),
            6: (False, True, False),
        }
        assert {row.order_number: row.simulation_period for row in user_101} == {
            1: "initial",
            2: "initial",
            3: "initial",
            4: "validation",
            5: "t1",
            6: "t2",
        }
    else:
        assert {row.development_split for row in user_101} == {None}
        assert {row.arrival_period for row in user_101} == {None}
        assert flags_101 == {
            1: (False, False, True),
            2: (False, False, True),
            3: (False, False, True),
            4: (False, False, True),
            5: (False, False, True),
            6: (False, False, True),
        }
        assert {row.simulation_period for row in user_101} == {"final_holdout"}

    user_202 = result.filter(F.col("user_id") == 202).orderBy("order_number").collect()
    assert len(user_202) == 4
    assert {row.user_cohort for row in user_202} == {"new_user"}
    assert {row.development_split for row in user_202} == {None}
    assert {row.current_period for row in user_202} == {"t2"}
    assert {row.order_history for row in user_202} == {4}
    assert {row.simulation_period for row in user_202} == {"new_user_pool"}

    arrival_period = user_202[0].arrival_period
    assert arrival_period in {"t1", "t2"}
    assert {row.arrival_period for row in user_202} == {arrival_period}

    flags_202 = {
        row.order_number: (
            row.is_train_available,
            row.is_validation_available,
            row.is_evaluation_available,
        )
        for row in user_202
    }

    if arrival_period == "t2":
        assert flags_202 == {
            1: (True, True, False),
            2: (True, True, False),
            3: (True, True, False),
            4: (False, True, False),
        }
    else:
        assert flags_202 == {
            1: (True, False, False),
            2: (True, False, False),
            3: (True, False, False),
            4: (True, False, False),
        }


def test_build_order_products_does_not_write_invalid_silver_dataset(
    spark,
    tmp_path,
):
    bronze_path = tmp_path / "bronze"
    order_path = tmp_path / "simulation" / "initial"
    output_path = tmp_path / "silver"
    written_path = output_path / "order_products"

    products_df = spark.createDataFrame(
        [
            (10, "Apple", 1, 1),
            (20, "Banana", 2, 2),
        ],
        """
        product_id INT,
        product_name STRING,
        aisle_id INT,
        department_id INT
        """,
    )

    bad_prior_df = spark.createDataFrame(
        [
            (1, 999, 1, 0),
        ],
        """
        order_id INT,
        product_id INT,
        add_to_cart_order INT,
        reordered INT
        """,
    )
    train_df = spark.createDataFrame(
        [],
        schema=bad_prior_df.schema,
    )
    available_orders_df = spark.createDataFrame(
        [
            (
                1,
                100,
                "prior",
                1,
                0,
                10,
                None,
                6,
                "established",
                "base_train",
                None,
                "initial",
                "initial",
                True,
                True,
                False,
            ),
        ],
        """
        order_id INT,
        user_id INT,
        eval_set STRING,
        order_number INT,
        order_dow INT,
        order_hour_of_day INT,
        days_since_prior_order DOUBLE,
        order_history INT,
        user_cohort STRING,
        development_split STRING,
        arrival_period STRING,
        simulation_period STRING,
        current_period STRING,
        is_train_available BOOLEAN,
        is_validation_available BOOLEAN,
        is_evaluation_available BOOLEAN
        """,
    )
    products_df.write.parquet(str(bronze_path / "products"))
    bad_prior_df.write.parquet(str(bronze_path / "order_products__prior"))
    train_df.write.parquet(str(bronze_path / "order_products__train"))
    available_orders_df.write.parquet(str(order_path / "available_orders"))

    with pytest.raises(DataValidationError):
        build_order_products(
            spark=spark,
            input_path=str(bronze_path),
            output_path=str(output_path),
            order_path=str(order_path),
            contract_path=str(CONTRACT_PATH),
        )

    assert not written_path.exists()
