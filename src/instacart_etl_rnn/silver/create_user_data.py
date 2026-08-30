from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_order_level_data(df: DataFrame) -> DataFrame:
    """
    Aggregate item-level records into order-level sequences.

    Groups items by user and order, preserves the add-to-cart order of
    products, aggregates order metadata, and concatenates product,
    reorder, department, and aisle values with '_' as
    sequence features.

    Args:
        df: Item-level DataFrame containing one row per product in an order.
    Returns:
        A DataFrame containing one row per (user_id, order_id) with
        order metadata and ordered sequence features.
    """

    items_struct = F.struct(
        F.col("add_to_cart_order"),
        F.col("product_id").cast("string").alias("product_id"),
        F.col("reordered").cast("string").alias("reordered"),
        F.col("aisle_id").cast("string").alias("aisle_id"),
        F.col("department_id").cast("string").alias("department_id"),
    )

    df = (
        df.withColumn("item", items_struct)
        .groupby("user_id", "order_id")
        .agg(
            F.sort_array(F.collect_list("item")).alias("items"),
            F.first("order_number").alias("order_number"),
            F.first("order_dow").alias("order_dow"),
            F.first("order_hour_of_day").alias("order_hour"),
            F.first("days_since_prior_order").alias("days_since_prior_order"),
            F.first("eval_set").alias("eval_set"),
        )
        .withColumn(
            "products", F.concat_ws("_", F.expr("transform(items, x -> x.product_id)"))
        )
        .withColumn(
            "reorders", F.concat_ws("_", F.expr("transform(items, x -> x.reordered)"))
        )
        .withColumn(
            "aisles", F.concat_ws("_", F.expr("transform(items, x -> x.aisle_id)"))
        )
        .withColumn(
            "departments",
            F.concat_ws("_", F.expr("transform(items, x -> x.department_id)")),
        )
        .drop("items")
    )

    return df


def build_user_level_data(df: DataFrame) -> DataFrame:
    """
    Aggregate order-level features into user-level purchase histories.

    Groups orders by user, perserved by order number sequence, and concatenates
    order metadata and product-related sequences into user-level history
    features.

    Args:
        df: Order-level DataFrame containing one row per (user_id, order_id).
    Returns:
        A DataFrame containing one row per user with chronological order history
    """

    orders = F.struct(
        F.col("order_number").cast("int").alias("order_number_sort"),
        F.col("order_id").cast("string").alias("order_id"),
        F.col("order_number").cast("string").alias("order_number"),
        F.col("order_dow").cast("string").alias("order_dow"),
        F.col("order_hour").cast("string").alias("order_hour"),
        (
            F.col("days_since_prior_order")
            .cast("string")
            .alias("days_since_prior_order")
        ),
        F.col("products"),
        F.col("reorders"),
        F.col("aisles"),
        F.col("departments"),
        F.col("eval_set"),
    )

    df = (
        df.withColumn("orders", orders)
        .groupby("user_id")
        .agg(F.sort_array(F.collect_list("orders")).alias("orders"))
        .withColumn(
            "order_ids", F.concat_ws(" ", F.expr("transform(orders, x -> x.order_id)"))
        )
        .withColumn(
            "order_numbers",
            F.concat_ws(" ", F.expr("transform(orders, x -> x.order_number)")),
        )
        .withColumn(
            "order_dows",
            F.concat_ws(" ", F.expr("transform(orders, x -> x.order_dow)")),
        )
        .withColumn(
            "order_hours",
            F.concat_ws(" ", F.expr("transform(orders, x -> x.order_hour)")),
        )
        .withColumn(
            "days_since_prior_orders",
            F.concat_ws(
                " ", F.expr("transform(orders, x -> x.days_since_prior_order)")
            ),
        )
        .withColumn(
            "product_ids",
            F.concat_ws(" ", F.expr("transform(orders, x -> x.products)")),
        )
        .withColumn(
            "reorders", F.concat_ws(" ", F.expr("transform(orders, x -> x.reorders)"))
        )
        .withColumn(
            "department_ids",
            F.concat_ws(" ", F.expr("transform(orders, x -> x.departments)")),
        )
        .withColumn(
            "aisle_ids", F.concat_ws(" ", F.expr("transform(orders, x -> x.aisles)"))
        )
        .withColumn(
            "eval_set", F.expr("element_at(transform(orders, x -> x.eval_set), -1)")
        )
        .drop("orders")
    )

    return df
