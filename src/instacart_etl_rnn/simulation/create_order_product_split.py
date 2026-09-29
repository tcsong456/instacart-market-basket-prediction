from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def select_base_model_users(
    order_products: DataFrame,
) -> DataFrame:
    """Select users used by the base-model training and evaluation pipeline.

    Includes:
    - established users assigned to the ``base_train`` development split;
    - all new users;
    - final holdout users used for champion/challenger evaluation.

    Established users assigned to ``stacking_train`` and excluded users are
    removed.

    Args:
        order_products: Order-product DataFrame containing user cohort and
            development split metadata.

    Returns:
        DataFrame containing users eligible for base-model processing.
    """

    return order_products.filter(
        (
            (F.col("user_cohort") == "established")
            & (F.col("development_split") == "base_train")
        )
        | (F.col("user_cohort") == "new_user")
        | (F.col("user_cohort") == "final_holdout")
    )


# Latest order included in the stacking train snapshot. The gold builder
# treats that order as the label, so the example is history through H-offset
# and label H-offset: initial is H-3/H-2, t1 is H-2/H-1, and t2 is H-1/H.
STACKING_TRAIN_OFFSET = {
    "initial": 2,
    "t1": 1,
    "t2": 0,
}


def select_stacking_model_users(
    order_products: DataFrame,
    period: str,
) -> DataFrame:
    """Select stacking users and assign a timeline-specific train snapshot.

    Keeps only established users assigned to the ``stacking_train``
    development split. The user set does not depend on ``period``.

    Train availability follows that period's prediction example. The snapshot
    includes the label order, which the gold builder uses as the target:

    - ``initial``: orders through H-2, so the example is H-3 / H-2
    - ``t1``: orders through H-1, so the example is H-2 / H-1
    - ``t2``: orders through H, so the example is H-1 / H

    Validation availability stays orders through H for every period. That
    shared snapshot is the final-order example.

    Args:
        order_products: Order-product DataFrame containing user cohort,
            development split, order number, and order history metadata.
        period: Simulation period. Must be ``initial``, ``t1``, or ``t2``.

    Returns:
        DataFrame containing stacking users with train and validation
        availability flags recalculated for stacking-model training.

    Raises:
        ValueError: If ``period`` is unsupported.
    """

    if period not in STACKING_TRAIN_OFFSET:
        raise ValueError(f"Unsupported simulation period: {period}")

    order_products = order_products.filter(
        (F.col("user_cohort") == "established")
        & (F.col("development_split") == "stacking_train")
    )

    train_order = F.col("order_history") - F.lit(STACKING_TRAIN_OFFSET[period])

    return order_products.withColumn(
        "is_train_available",
        F.col("order_number") <= train_order,
    ).withColumn(
        "is_validation_available",
        F.col("order_number") <= F.col("order_history"),
    )


def split_order_products_by_role(
    order_products: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame]:

    train_history = order_products.filter(F.col("is_train_available"))

    evaluation_history = order_products.filter(F.col("is_evaluation_available"))

    validation_history = order_products.filter(F.col("is_validation_available"))

    return train_history, evaluation_history, validation_history
