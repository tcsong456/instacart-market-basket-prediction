from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def build_hpt_user_sample(df: DataFrame) -> DataFrame:
    """Build a deterministic 30% user sample for HPT."""
    return (
        df.select("user_id")
        .distinct()
        .withColumn(
            "_hpt_bucket",
            F.pmod(
                F.xxhash64("user_id", F.lit("hpt_sample")),
                F.lit(10),
            ),
        )
        .filter(F.col("_hpt_bucket") < 3)
        .select("user_id")
    )


if __name__ == "__main__":
    from pyspark.sql import SparkSession

    from instacart_etl_rnn.common.io import read_parquet, write_parquet

    spark = (
        SparkSession.builder.appName("build_hpt_sample")
        .config("spark.driver.memory", "12g")
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )

    train = read_parquet("data/hpt/product_training_data_train", spark)

    validation = read_parquet("data/hpt/product_training_data_validation", spark)

    hpt_users = build_hpt_user_sample(train)

    hpt_train = train.join(
        hpt_users,
        on="user_id",
        how="left_semi",
    )

    hpt_validation = validation.join(
        hpt_users,
        on="user_id",
        how="left_semi",
    )

    write_parquet("data/hpt/sample_users_train", hpt_train)

    write_parquet("data/hpt/sample_users_validation", hpt_validation)
