from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def select_base_model_users(
    order_products: DataFrame,
) -> DataFrame:
    return order_products.filter(
        (
            (F.col("user_cohort") == "established")
            & (F.col("development_split") == "base_train")
        )
        | (F.col("user_cohort") == "new_user")
    )


def select_stacking_model_users(order_products: DataFrame) -> DataFrame:
    order_products = order_products.filter(
        (F.col("user_cohort") == "established")
        & (F.col("development_split") == "stacking_train")
    )

    return order_products.withColumn(
        "order_role",
        F.when(
            F.col("order_number") < F.col("order_history") - 1,
            F.lit("history"),
        )
        .when(
            F.col("order_number") == F.col("order_history") - 1,
            F.lit("train_label"),
        )
        .when(
            F.col("order_number") == F.col("order_history"),
            F.lit("validation_label"),
        ),
    )


def split_order_products_by_role(
    order_products: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame]:

    history = order_products.filter(F.col("order_role") == "history")

    train_label = order_products.filter(F.col("order_role") == "train_label")

    validation_label = order_products.filter(F.col("order_role") == "validation_label")

    return history, train_label, validation_label
