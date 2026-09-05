from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_user_product_count(
    order_products: DataFrame,
) -> DataFrame:
    """
    Build leakage-safe historical user-product order counts.

    Assumes the input is an already-filtered temporal snapshot where each
    user's latest available order is the prediction target. Only orders
    strictly before that target contribute to the count.

    Args:
        order_products: Product-level snapshot containing user_id, order_id,
            order_number, and product_id.

    Returns:
        One row per historical user-product pair with the number of orders
        containing that product.
    """
    target_orders = order_products.groupBy("user_id").agg(
        F.max("order_number").cast("int").alias("_target_order_number")
    )

    history = order_products.join(
        target_orders,
        on="user_id",
        how="inner",
    ).filter(F.col("order_number") < F.col("_target_order_number"))

    return history.groupBy(
        "user_id",
        "product_id",
    ).agg(F.countDistinct("order_id").cast("int").alias("count"))
