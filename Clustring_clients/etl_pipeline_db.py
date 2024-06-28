import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, DoubleType, TimestampType
from pyspark.sql.functions import col, max, datediff, count, sum
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def create_spark_session():
    spark = SparkSession.builder \
        .appName("clustering") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "5g") \
        .config("spark.executor.cores", "4") \
        .config("spark.driver.maxResultSize", "2g") \
        .config("spark.jars", "/home/ghani/spark/jars/postgresql-42.2.23.jar") \
        .getOrCreate()
    return spark

def get_schema():
    schema = StructType([
        StructField("event_time", TimestampType(), True),
        StructField("event_type", StringType(), True),
        StructField("product_id", IntegerType(), True),
        StructField("category_id", LongType(), True),
        StructField("category_code", StringType(), True),
        StructField("brand", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("user_id", IntegerType(), True),
        StructField("user_session", StringType(), True)
    ])
    return schema

def read_data(spark, jdbc_url, jdbc_properties, tables):
    dfs = [spark.read.jdbc(url=jdbc_url, table=table, properties=jdbc_properties) for table in tables]
    combined_df = dfs[0]
    for df in dfs[1:]:
        combined_df = combined_df.union(df)
    return combined_df

def clean_and_transform_data(df):
    df = df.select("event_time", "event_type", "price", "user_id")
    df = df.filter(col("event_type") == "purchase")
    df = df.dropna(subset=["event_time", "event_type", "price", "user_id"])
    df = df.dropDuplicates()
    df = df.filter(df['price'] != 0)
    df = df.withColumn("event_time", F.to_date(col("event_time")))
    return df

def calculate_rfm_scores(df):
    reference_date = df.agg(max("event_time")).collect()[0][0]
    rfm_table = df.groupBy("user_id").agg(
        datediff(F.lit(reference_date), max("event_time")).alias("recency"),
        count("event_time").alias("frequency"),
        sum("price").alias("monetary")
    )

    recency_window = Window.orderBy("recency")
    frequency_window = Window.orderBy(col("frequency").desc())
    monetary_window = Window.orderBy(col("monetary").desc())

    rfm_table = rfm_table.withColumn("recency_rank", F.row_number().over(recency_window))
    rfm_table = rfm_table.withColumn("frequency_rank", F.row_number().over(frequency_window))
    rfm_table = rfm_table.withColumn("monetary_rank", F.row_number().over(monetary_window))

    total_users = rfm_table.count()
    rfm_table = rfm_table.withColumn("recency_score", F.ceil((rfm_table.recency_rank / total_users) * 5))
    rfm_table = rfm_table.withColumn("frequency_score", F.ceil((rfm_table.frequency_rank / total_users) * 5))
    rfm_table = rfm_table.withColumn("monetary_score", F.ceil((rfm_table.monetary_rank / total_users) * 5))

    return rfm_table.select("user_id", "recency_score", "frequency_score", "monetary_score")

def write_results(df, jdbc_url, jdbc_properties, output_table):
    df.write.jdbc(url=jdbc_url, table=output_table, mode='overwrite', properties=jdbc_properties)

def main(jdbc_url, jdbc_properties, input_tables, output_table):
    spark = create_spark_session()
    schema = get_schema()
    combined_df = read_data(spark, jdbc_url, jdbc_properties, input_tables)
    clean_df = clean_and_transform_data(combined_df)
    rfm_scores_df = calculate_rfm_scores(clean_df)
    write_results(rfm_scores_df, jdbc_url, jdbc_properties, output_table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline for RFM scoring")
    parser.add_argument("--jdbc-url", required=True, help="JDBC URL for the database")
    parser.add_argument("--jdbc-user", required=True, help="Database user")
    parser.add_argument("--jdbc-password", required=True, help="Database password")
    parser.add_argument("--input-tables", nargs='+', required=True, help="List of input table names")
    parser.add_argument("--output-table", required=True, help="Output table name")

    args = parser.parse_args()

    jdbc_properties = {
        "user": args.jdbc_user,
        "password": args.jdbc_password,
        "driver": "org.postgresql.Driver"
    }

    main(args.jdbc_url, jdbc_properties, args.input_tables, args.output_table)
