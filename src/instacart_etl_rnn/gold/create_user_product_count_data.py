from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_user_product_count(df: DataFrame) -> DataFrame:
    return (
        df.filter(F.col("eval_set") == "prior")
        .groupBy("user_id", "product_id")
        .agg(F.count("*").cast("int").alias("count"))
    )
