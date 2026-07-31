from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType


def read_csv(
    path: str | Path, spark: SparkSession, schema: StructType | None = None
) -> DataFrame:
    reader = spark.read.option("header", True)
    if schema is not None:
        reader = reader.schema(schema)
    else:
        reader = reader.option("inferSchema", True)
    return reader.csv(str(path))


def read_parquet(path: str | Path, spark: SparkSession) -> DataFrame:
    return spark.read.parquet(str(path))


def write_parquet(path: str | Path, df: DataFrame) -> None:
    (df.write.mode("overwrite").parquet(str(path)))
