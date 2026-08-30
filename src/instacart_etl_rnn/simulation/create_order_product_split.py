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


def select_stacking_model_users(
    order_products: DataFrame,
) -> DataFrame:
    """Select established users reserved for stacking-model training.

    Keeps only established users assigned to the ``stacking_train``
    development split.

    For stacking users, all orders except the final order are available for
    training, while all orders are available for validation. This allows the
    final order to act as the stacking validation target.

    Args:
        order_products: Order-product DataFrame containing user cohort,
            development split, order number, and order history metadata.

    Returns:
        DataFrame containing stacking users with train and validation
        availability flags recalculated for stacking-model training.
    """

    order_products = order_products.filter(
        (F.col("user_cohort") == "established")
        & (F.col("development_split") == "stacking_train")
    )

    return order_products.withColumn(
        "is_train_available",
        F.col("order_number") <= F.col("order_history") - 1,
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
