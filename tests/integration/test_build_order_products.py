from pathlib import Path

import pytest

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.silver.create_order_products import build_order_products
from instacart_etl_rnn.validation.exceptions import DataValidationError

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
)


def test_build_order_products_preserves_simulation_columns(
    spark,
    tmp_path,
):
    order_products_prior = spark.createDataFrame(
        [
            (100, 10, 1, 0),
            (101, 20, 2, 1),
        ],
        """
        order_id int,
        product_id int,
        add_to_cart_order int,
        reordered int
        """,
    )
    write_parquet(tmp_path / "order_products__prior", order_products_prior)

    order_products_train = spark.createDataFrame(
        [
            (102, 10, 1, 1),
        ],
        """
        order_id int,
        product_id int,
        add_to_cart_order int,
        reordered int
        """,
    )
    write_parquet(tmp_path / "order_products__train", order_products_train)

    orders = spark.createDataFrame(
        [
            (
                100,
                1,
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
            (
                101,
                1,
                "prior",
                2,
                1,
                11,
                7.0,
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
            (
                102,
                1,
                "train",
                3,
                2,
                12,
                5.0,
                6,
                "established",
                "base_train",
                None,
                "validation",
                "initial",
                True,
                True,
                False,
            ),
        ],
        """
        order_id int,
        user_id int,
        eval_set string,
        order_number int,
        order_dow int,
        order_hour_of_day int,
        days_since_prior_order double,
        order_history int,
        user_cohort string,
        development_split string,
        arrival_period string,
        simulation_period string,
        current_period string,
        is_train_available boolean,
        is_validation_available boolean,
        is_evaluation_available boolean
        """,
    )
    write_parquet(tmp_path / "orders" / "available_orders", orders)

    products = spark.createDataFrame(
        [
            (10, "Banana", 1, 1),
            (20, "Milk", 2, 2),
        ],
        """
        product_id int,
        product_name string,
        aisle_id int,
        department_id int
        """,
    )
    write_parquet(tmp_path / "products", products)

    build_order_products(
        spark=spark,
        input_path=tmp_path,
        output_path=tmp_path / "silver",
        contract_path=CONTRACT_PATH,
        order_path=tmp_path / "orders",
    )

    output_df = read_parquet(tmp_path / "silver" / "order_products", spark)

    actual = {
        (row.order_id, row.product_id): row.asDict(recursive=True)
        for row in output_df.collect()
    }

    assert actual[(100, 10)]["user_id"] == 1
    assert actual[(100, 10)]["order_history"] == 6
    assert actual[(100, 10)]["user_cohort"] == "established"
    assert actual[(100, 10)]["development_split"] == "base_train"
    assert actual[(100, 10)]["arrival_period"] is None
    assert actual[(100, 10)]["simulation_period"] == "initial"
    assert actual[(100, 10)]["current_period"] == "initial"
    assert actual[(100, 10)]["is_train_available"] is True
    assert actual[(100, 10)]["is_validation_available"] is True
    assert actual[(100, 10)]["is_evaluation_available"] is False

    assert actual[(101, 20)]["is_train_available"] is True
    assert actual[(101, 20)]["is_validation_available"] is True
    assert actual[(101, 20)]["is_evaluation_available"] is False

    assert actual[(102, 10)]["is_train_available"] is True
    assert actual[(102, 10)]["is_validation_available"] is True
    assert actual[(102, 10)]["is_evaluation_available"] is False
    assert actual[(102, 10)]["simulation_period"] == "validation"

    assert actual[(100, 10)]["days_since_prior_order"] == -1.0

    assert actual[(100, 10)]["product_name"] == "Banana"
    assert actual[(100, 10)]["aisle_id"] == 1
    assert actual[(100, 10)]["department_id"] == 1


def test_build_order_products_preserves_new_user_metadata(
    spark,
    tmp_path,
):
    prior = spark.createDataFrame(
        [
            (200, 30, 1, 0),
            (201, 30, 1, 1),
            (202, 40, 2, 1),
        ],
        """
        order_id int,
        product_id int,
        add_to_cart_order int,
        reordered int
        """,
    )
    write_parquet(tmp_path / "order_products__prior", prior)

    train = spark.createDataFrame(
        [],
        schema=prior.schema,
    )
    write_parquet(tmp_path / "order_products__train", train)

    orders = spark.createDataFrame(
        [
            (
                200,
                50,
                "prior",
                1,
                0,
                10,
                None,
                3,
                "new_user",
                None,
                "t1",
                "new_user_pool",
                "t1",
                True,
                True,
                False,
            ),
            (
                201,
                50,
                "prior",
                2,
                1,
                11,
                7.0,
                3,
                "new_user",
                None,
                "t1",
                "new_user_pool",
                "t1",
                True,
                True,
                False,
            ),
            (
                202,
                50,
                "prior",
                3,
                2,
                12,
                8.0,
                3,
                "new_user",
                None,
                "t1",
                "new_user_pool",
                "t1",
                False,
                True,
                False,
            ),
        ],
        """
        order_id int,
        user_id int,
        eval_set string,
        order_number int,
        order_dow int,
        order_hour_of_day int,
        days_since_prior_order double,
        order_history int,
        user_cohort string,
        development_split string,
        arrival_period string,
        simulation_period string,
        current_period string,
        is_train_available boolean,
        is_validation_available boolean,
        is_evaluation_available boolean
        """,
    )
    write_parquet(tmp_path / "orders" / "available_orders", orders)

    products = spark.createDataFrame(
        [
            (30, "Milk", 2, 2),
            (40, "Bread", 3, 3),
        ],
        """
        product_id int,
        product_name string,
        aisle_id int,
        department_id int
        """,
    )
    write_parquet(tmp_path / "products", products)

    build_order_products(
        spark=spark,
        input_path=tmp_path,
        output_path=tmp_path / "silver",
        contract_path=CONTRACT_PATH,
        order_path=tmp_path / "orders",
    )

    result = read_parquet(tmp_path / "silver" / "order_products", spark)

    actual = {row.order_number: row.asDict(recursive=True) for row in result.collect()}

    assert actual[1]["user_cohort"] == "new_user"
    assert actual[1]["development_split"] is None
    assert actual[1]["arrival_period"] == "t1"
    assert actual[1]["is_train_available"] is True
    assert actual[1]["is_validation_available"] is True
    assert actual[1]["is_evaluation_available"] is False

    assert actual[2]["is_train_available"] is True
    assert actual[2]["is_validation_available"] is True
    assert actual[2]["is_evaluation_available"] is False

    assert actual[3]["is_train_available"] is False
    assert actual[3]["is_validation_available"] is True
    assert actual[3]["is_evaluation_available"] is False


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
