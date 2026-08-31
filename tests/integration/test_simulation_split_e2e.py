from pathlib import Path

from pyspark.sql import functions as F

from instacart_etl_rnn.common.io import write_parquet
from instacart_etl_rnn.jobs.create_user_split_data_job import (
    run_simulation_split_job,
)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "instacart_etl_rnn" / "contracts"
)


def test_run_simulation_split_job_end_to_end(
    spark,
    tmp_path,
):
    input_path = tmp_path / "input"
    output_path = tmp_path / "output"

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
            (3031, 303, "prior", 1, 0, 10, None),
            (3032, 303, "train", 2, 1, 11, 5.0),
            (4041, 404, "test", 1, 0, 10, None),
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

    write_parquet(input_path / "orders", orders)

    run_simulation_split_job(
        spark=spark,
        input_path=str(input_path),
        output_path=str(output_path),
        contract_path=str(CONTRACT_PATH),
        period="initial",
    )

    result = spark.read.parquet(str(output_path / "available_orders"))

    assert result.count() == 12

    assert {
        "user_id",
        "order_id",
        "order_number",
        "eval_set",
        "order_dow",
        "order_hour_of_day",
        "days_since_prior_order",
        "order_history",
        "user_cohort",
        "development_split",
        "arrival_period",
        "simulation_period",
        "current_period",
        "is_train_available",
        "is_validation_available",
        "is_evaluation_available",
    }.issubset(result.columns)

    assert {row.user_id for row in result.select("user_id").distinct().collect()} == {
        101,
        202,
        303,
    }

    assert {
        row.current_period
        for row in result.select("current_period").distinct().collect()
    } == {
        "initial",
    }

    excluded = result.filter(F.col("user_id") == 303).orderBy("order_number").collect()

    assert excluded

    for row in excluded:
        assert row.user_cohort == "excluded"
        assert row.development_split is None
        assert row.arrival_period is None
        assert row.is_train_available is False
        assert row.is_validation_available is False
        assert row.is_evaluation_available is False

    new_user = result.filter(F.col("user_id") == 202).orderBy("order_number").collect()

    assert new_user
    assert all(row.user_cohort == "new_user" for row in new_user)
    assert all(row.development_split is None for row in new_user)
    assert {row.arrival_period for row in new_user} <= {"t1", "t2"}

    user_101 = result.filter(F.col("user_id") == 101).orderBy("order_number").collect()

    assert user_101

    cohort = user_101[0].user_cohort

    assert cohort in {
        "established",
        "final_holdout",
    }

    assert all(row.user_cohort == cohort for row in user_101)

    if cohort == "established":
        assert all(
            row.development_split
            in {
                "base_train",
                "stacking_train",
            }
            for row in user_101
        )

        actual = {
            row.order_number: (
                row.is_train_available,
                row.is_validation_available,
                row.is_evaluation_available,
            )
            for row in user_101
        }

        assert actual == {
            1: (True, True, False),
            2: (True, True, False),
            3: (True, True, False),
            4: (False, True, False),
            5: (False, False, False),
            6: (False, False, False),
        }

    else:
        assert all(row.development_split is None for row in user_101)

        actual = {
            row.order_number: (
                row.is_train_available,
                row.is_validation_available,
                row.is_evaluation_available,
            )
            for row in user_101
        }

        assert actual == {
            1: (False, False, True),
            2: (False, False, True),
            3: (False, False, True),
            4: (False, False, True),
            5: (False, False, False),
            6: (False, False, False),
        }
