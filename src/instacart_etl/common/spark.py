from pyspark.sql import SparkSession

def create_spark_session(name: str) -> SparkSession:
    return SparkSession.builder.appName(name).getOrCreate()