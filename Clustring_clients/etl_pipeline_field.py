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

def read_data(spark, schema, input_paths):
    dfs = [spark.read.csv(file, header=True, schema=schema) for file in input_paths]
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

def write_results(df, output_path):
    df.write.parquet(output_path)

def main(input_paths, output_path):
    spark = create_spark_session()
    schema = get_schema()
    combined_df = read_data(spark, schema, input_paths)
    clean_df = clean_and_transform_data(combined_df)
    rfm_scores_df = calculate_rfm_scores(clean_df)
    write_results(rfm_scores_df, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETL pipeline for RFM scoring")
    parser.add_argument("--input", nargs='+', required=True, help="List of input file paths")
    parser.add_argument("--output", required=True, help="Output file path")

    args = parser.parse_args()
    main(args.input, args.output)
