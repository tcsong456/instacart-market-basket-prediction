from pathlib import Path

from pyspark.sql import Row

from instacart_etl_rnn.common.io import read_parquet, write_parquet
from instacart_etl_rnn.jobs.create_period_split_data_job import (
    COLUMNS,
    run_order_products_split_job,
)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
)

_DEFAULTS = {
    "add_to_cart_order": 1,
    "reordered": 0,
    "eval_set": "prior",
    "order_dow": 0,
    "order_hour_of_day": 10,
    "days_since_prior_order": -1.0,
    "product_name": "Banana",
    "aisle_id": 1,
    "department_id": 1,
}


def _row(**fields) -> Row:
    return Row(**{**_DEFAULTS, **fields})


def _order_ids(df):
    return {row.order_id for row in df.select("order_id").collect()}


def test_run_order_products_split_job_base_train_end_to_end(spark, tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"

    order_products = spark.createDataFrame(
        [
            _row(
                product_id=10,
                order_id=101,
                user_id=1,
                order_number=1,
                order_history=3,
                user_cohort="established",
                development_split="base_train",
                is_train_available=True,
                is_validation_available=True,
                is_evaluation_available=False,
            ),
            _row(
                product_id=10,
                order_id=102,
                user_id=1,
                order_number=2,
                order_history=3,
                user_cohort="established",
                development_split="base_train",
                is_train_available=True,
                is_validation_available=True,
                is_evaluation_available=False,
                days_since_prior_order=5.0,
                reordered=1,
            ),
            _row(
                product_id=20,
                order_id=103,
                user_id=1,
                order_number=3,
                order_history=3,
                user_cohort="established",
                development_split="base_train",
                is_train_available=False,
                is_validation_available=True,
                is_evaluation_available=False,
                days_since_prior_order=6.0,
                eval_set="train",
                product_name="Milk",
                aisle_id=2,
                department_id=2,
            ),
            _row(
                product_id=30,
                order_id=201,
                user_id=2,
                order_number=1,
                order_history=3,
                user_cohort="final_holdout",
                development_split=None,
                is_train_available=False,
                is_validation_available=False,
                is_evaluation_available=True,
                product_name="Bread",
                aisle_id=3,
                department_id=3,
            ),
            _row(
                product_id=30,
                order_id=202,
                user_id=2,
                order_number=2,
                order_history=3,
                user_cohort="final_holdout",
                development_split=None,
                is_train_available=False,
                is_validation_available=False,
                is_evaluation_available=True,
                days_since_prior_order=7.0,
                product_name="Bread",
                aisle_id=3,
                department_id=3,
                reordered=1,
            ),
            _row(
                product_id=40,
                order_id=203,
                user_id=2,
                order_number=3,
                order_history=3,
                user_cohort="final_holdout",
                development_split=None,
                is_train_available=False,
                is_validation_available=False,
                is_evaluation_available=False,
                days_since_prior_order=8.0,
                eval_set="train",
                product_name="Eggs",
                aisle_id=4,
                department_id=4,
            ),
            _row(
                product_id=50,
                order_id=301,
                user_id=3,
                order_number=1,
                order_history=2,
                user_cohort="established",
                development_split="stacking_train",
                is_train_available=True,
                is_validation_available=True,
                is_evaluation_available=False,
                product_name="Apple",
                aisle_id=5,
                department_id=5,
            ),
        ]
    )

    write_parquet(input_path / "order_products", order_products)

    run_order_products_split_job(
        spark=spark,
        input_path=str(input_path),
        output_path=str(output_path),
        mode="base_train",
        period="t1",
        contract_path=str(CONTRACT_PATH),
    )

    period_dir = output_path / "t1"
    train = read_parquet(period_dir / "order_products_train", spark)
    validation = read_parquet(period_dir / "order_products_validation", spark)
    evaluation = read_parquet(period_dir / "order_products_evaluation", spark)

    assert _order_ids(train) == {101, 102}
    assert _order_ids(validation) == {101, 102, 103}
    assert _order_ids(evaluation) == {201, 202}

    assert set(train.columns) == set(COLUMNS)
    assert set(validation.columns) == set(COLUMNS)
    assert set(evaluation.columns) == set(COLUMNS)

    assert not (output_path / "stacking_train").exists()


def test_run_order_products_split_job_stacking_train_end_to_end(spark, tmp_path):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"

    order_products = spark.createDataFrame(
        [
            _row(
                product_id=10,
                order_id=1001,
                user_id=10,
                order_number=1,
                order_history=3,
                user_cohort="established",
                development_split="stacking_train",
                is_train_available=False,
                is_validation_available=False,
                is_evaluation_available=False,
            ),
            _row(
                product_id=20,
                order_id=1002,
                user_id=10,
                order_number=2,
                order_history=3,
                user_cohort="established",
                development_split="stacking_train",
                is_train_available=False,
                is_validation_available=False,
                is_evaluation_available=False,
                days_since_prior_order=5.0,
                product_name="Milk",
                aisle_id=2,
                department_id=2,
                reordered=1,
            ),
            _row(
                product_id=30,
                order_id=1003,
                user_id=10,
                order_number=3,
                order_history=3,
                user_cohort="established",
                development_split="stacking_train",
                is_train_available=False,
                is_validation_available=False,
                is_evaluation_available=False,
                days_since_prior_order=6.0,
                eval_set="train",
                product_name="Bread",
                aisle_id=3,
                department_id=3,
            ),
            _row(
                product_id=40,
                order_id=1101,
                user_id=11,
                order_number=1,
                order_history=2,
                user_cohort="established",
                development_split="base_train",
                is_train_available=True,
                is_validation_available=True,
                is_evaluation_available=False,
                product_name="Eggs",
                aisle_id=4,
                department_id=4,
            ),
        ]
    )

    write_parquet(input_path / "order_products", order_products)

    run_order_products_split_job(
        spark=spark,
        input_path=str(input_path),
        output_path=str(output_path),
        mode="stacking_train",
        period="t1",
        contract_path=str(CONTRACT_PATH),
    )

    stacking_dir = output_path / "stacking_train"
    train = read_parquet(stacking_dir / "order_products_train", spark)
    validation = read_parquet(stacking_dir / "order_products_validation", spark)

    assert _order_ids(train) == {1001, 1002}
    assert _order_ids(validation) == {1001, 1002, 1003}

    assert set(train.columns) == set(COLUMNS)
    assert set(validation.columns) == set(COLUMNS)

    assert not (stacking_dir / "order_products_evaluation").exists()
    assert not (output_path / "t1").exists()
