from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_user_simulation_split(
    orders: DataFrame,
) -> DataFrame:
    """
    Assign users to deterministic simulation cohorts.

    Users are grouped by order history and assigned to final holdout,
    established, new-user, or excluded cohorts. Established users are
    further split into base or stacking training, while new users are
    assigned an arrival period.

    Args:
        orders: Order-level DataFrame containing user_id and order_number.

    Returns:
        One row per user with cohort, development split, and arrival period.
    """

    users = (
        orders.groupBy("user_id")
        .agg(F.max("order_number").cast("int").alias("order_history"))
        .withColumn(
            "_holdout_bucket",
            F.pmod(
                F.xxhash64(
                    F.col("user_id"),
                    F.lit("final_holdout"),
                ),
                F.lit(10),
            ),
        )
    )

    users = users.withColumn(
        "user_cohort",
        F.when(
            (F.col("order_history") >= 6) & (F.col("_holdout_bucket") == 0),
            F.lit("final_holdout"),
        )
        .when(
            F.col("order_history") >= 6,
            F.lit("established"),
        )
        .when(
            F.col("order_history").between(3, 5),
            F.lit("new_user"),
        )
        .otherwise(F.lit("excluded")),
    )

    users = users.withColumn(
        "_development_bucket",
        F.pmod(
            F.xxhash64(
                F.col("user_id"),
                F.lit("development_split"),
            ),
            F.lit(10),
        ),
    )

    users = users.withColumn(
        "development_split",
        F.when(
            F.col("user_cohort") != "established",
            F.lit(None).cast("string"),
        )
        .when(
            F.col("_development_bucket") == 0,
            F.lit("stacking_train"),
        )
        .otherwise(F.lit("base_train")),
    )

    users = users.withColumn(
        "arrival_period",
        F.when(
            F.col("user_cohort") != "new_user",
            F.lit(None).cast("string"),
        ).otherwise(
            F.when(
                F.pmod(
                    F.xxhash64(
                        F.col("user_id"),
                        F.lit("new_user_arrival"),
                    ),
                    F.lit(2),
                )
                == 0,
                F.lit("t1"),
            ).otherwise(F.lit("t2"))
        ),
    )

    return users.select(
        "user_id",
        "order_history",
        "user_cohort",
        "development_split",
        "arrival_period",
    )


def build_order_simulation_split(
    orders: DataFrame,
    user_split: DataFrame,
) -> DataFrame:
    """Assign each order its role in the simulated model lifecycle."""

    df = orders.join(
        user_split,
        on="user_id",
        how="inner",
    )

    return df.withColumn(
        "simulation_period",
        F.when(
            F.col("user_cohort") == "final_holdout",
            F.lit("final_holdout"),
        )
        .when(
            F.col("user_cohort") == "excluded",
            F.lit("excluded"),
        )
        .when(
            F.col("user_cohort") == "new_user",
            F.lit("new_user_pool"),
        )
        .when(
            F.col("user_cohort") == "established",
            F.when(
                F.col("order_number") <= F.col("order_history") - 3,
                F.lit("initial"),
            )
            .when(
                F.col("order_number") == F.col("order_history") - 2,
                F.lit("validation"),
            )
            .when(
                F.col("order_number") == F.col("order_history") - 1,
                F.lit("t1"),
            )
            .when(
                F.col("order_number") == F.col("order_history"),
                F.lit("t2"),
            ),
        ),
    )


def add_order_role(
    df: DataFrame,
    period: str,
) -> DataFrame:
    """
    Add data-availability flags for a simulation period.

    Marks whether each order is available for training, validation,
    or final-holdout evaluation based on cohort, arrival period,
    order history, and the current simulation period.

    Args:
        df: Order-level DataFrame containing cohort and order metadata.
        period: Simulation period: initial, t1, or t2.

    Returns:
        DataFrame with current_period and availability flag columns.

    Raises:
        ValueError: If the simulation period is unsupported.
    """

    period_order = {
        "initial": 0,
        "t1": 1,
        "t2": 2,
    }

    if period not in period_order:
        raise ValueError(f"Unsupported simulation period: {period}")

    current_period = period_order[period]

    train_offset = {
        "initial": 3,
        "t1": 2,
        "t2": 1,
    }[period]

    validation_offset = {
        "initial": 2,
        "t1": 1,
        "t2": 0,
    }[period]

    evaluation_offset = {
        "initial": 2,
        "t1": 1,
        "t2": 0,
    }[period]

    train_order = F.col("order_history") - F.lit(train_offset)

    validation_order = F.col("order_history") - F.lit(validation_offset)

    evaluation_order = F.col("order_history") - F.lit(evaluation_offset)

    is_established = F.col("user_cohort") == "established"

    is_new_user = F.col("user_cohort") == "new_user"

    is_final_holdout = F.col("user_cohort") == "final_holdout"

    arrival_period_value = F.when(
        F.col("arrival_period") == "t1",
        F.lit(1),
    ).when(
        F.col("arrival_period") == "t2",
        F.lit(2),
    )

    is_current_new_user = is_new_user & (arrival_period_value == F.lit(current_period))

    is_previous_new_user = is_new_user & (arrival_period_value < F.lit(current_period))

    is_established_train = is_established & (F.col("order_number") <= train_order)

    is_established_validation = is_established & (
        F.col("order_number") <= validation_order
    )

    is_current_new_train = is_current_new_user & (
        F.col("order_number") <= F.col("order_history") - 1
    )

    is_current_new_validation = is_current_new_user & (
        F.col("order_number") <= F.col("order_history")
    )

    is_previous_new_train = is_previous_new_user & (
        F.col("order_number") <= F.col("order_history")
    )

    is_holdout_evaluation = is_final_holdout & (
        F.col("order_number") <= evaluation_order
    )

    return (
        df.withColumn(
            "current_period",
            F.lit(period),
        )
        .withColumn(
            "is_train_available",
            (is_established_train | is_current_new_train | is_previous_new_train),
        )
        .withColumn(
            "is_validation_available",
            (is_established_validation | is_current_new_validation),
        )
        .withColumn(
            "is_evaluation_available",
            is_holdout_evaluation,
        )
    )
