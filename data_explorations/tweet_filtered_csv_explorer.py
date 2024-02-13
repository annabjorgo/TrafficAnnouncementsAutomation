import csv
from datetime import date, datetime

import tweepy
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_utc_timestamp, from_unixtime, unix_timestamp, year, month
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format, hour, dayofmonth
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType, TimestampType, \
    LongType

path = [
    "data/NRK/filtered_json_data.csv"
  ]

spark = SparkSession.builder.appName("TAA_json").getOrCreate()
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
spark.conf.set("spark.sql.session.timeZone", "CET")
#%%
df_tweet = spark.read.csv(path, header=True, multiLine=True)