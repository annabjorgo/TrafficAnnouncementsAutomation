# %%
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format, hour, year, dayofmonth, month
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType, TimestampType, \
    LongType

path = "/Users/madslun/Documents/Fag-Programmering/TrafficAnnouncementsAutomation/data/NRK/output-3-10-.csv"

# %%
data_schema = StructType(fields=[
    StructField("Author_id", IntegerType()),
    StructField("Tweet_id", LongType()),
    StructField("Text", StringType()),
    StructField("Referenced_tweet", LongType()),
    StructField("Referenced_type", StringType()),
    StructField("Referenced_user", StringType()),
    StructField("Created_at", StringType())
])

spark = SparkSession.builder.appName("TAA").getOrCreate()

df_tweet = spark.read.schema(data_schema).option("header", "true").option("multiline", "true").csv(path=path)

# %%

df_tweet.limit(5).show()

# %%

# Checking for null values
df_tweet.where(col("Tweet_id").isNotNull()).count()

# %%
# Convert to CET/local timezone
df_tweet = df_tweet.withColumn("Created_at_local", date_format(col(("Created_at")), "yyyy-MM-dd HH:mm:ss.SSSX"))

# %%
# Get the hour distribution
hours = df_tweet.groupby(hour("Created_at_local")).count().collect()

# %%
# Visualize tweet hourly
import matplotlib.pyplot as plt

tmp = dict(hours)
for i in range(1, 25):
    if i not in tmp:
        tmp[i] = 0

tmp = dict(sorted(tmp.items()))

fig, ax = plt.subplots(figsize=(15, 10))
ax.margins(x=0)
ax.plot(tmp.keys(), tmp.values())
ax.set_title("Hourly tweets")
fig.show()

# %%
replied_count = df_tweet.where(col("Referenced_type") == "replied_to").count()
print(f"Replied to count {replied_count}")

# %%
replied_to_self = df_tweet.where(col("Referenced_type") == "replied_to").where(col("Referenced_user") == 20629858).count()
print(f"Replied to own id {replied_to_self}")

# %%
# NB elendig kode
tweet_ids = df_tweet.rdd.map(lambda x: x['Tweet_id']).collect()
replied_to_id_in_df = df_tweet.where(col("Referenced_type") == "replied_to").where(
    col("Referenced_tweet").isin(tweet_ids)).count()
print(f"Replied id exists in df {replied_to_id_in_df}")
# %%
quoted_count = df_tweet.where(col("Referenced_type") == "quoted").count()
print(f"Quoted count {quoted_count}")

# %%
# NB elendig kode
tweet_ids = df_tweet.rdd.map(lambda x: x['Tweet_id']).collect()
quoted_to_id_in_df = df_tweet.where(col("Referenced_type") == "quoted").where(
    col("Referenced_tweet").isin(tweet_ids)).count()
print(f"Quoted id exists in df {quoted_to_id_in_df}")

#%%
#Slow code
not_in_df = df_tweet.where(col("Referenced_tweet").isin(tweet_ids) == False).count()
print(f"Not in df {not_in_df}")

#%%
df_tweet.where((year('Created_at_local') == 2022) & (month('Created_at_local') == 10) & (dayofmonth('Created_at_local') == 10)).show(truncate=False)