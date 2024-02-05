from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_utc_timestamp, from_unixtime, unix_timestamp, year, month
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format, hour, dayofmonth
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType, TimestampType, \
    LongType

path = [
        "data/NRK/tweets.js",
        "data/NRK/tweets-part1.js",
        "data/NRK/tweets-part2.js"
        ]

spark = SparkSession.builder.appName("TAA_json").getOrCreate()
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
spark.conf.set("spark.sql.session.timeZone", "CET")
#%%
data_schema = StructType(fields=[
    # StructField("entities", ),
    StructField("id", LongType()),
    StructField("full_text", StringType()),
    StructField("in_reply_to_status_id_str", LongType()),
    StructField("created_at", StringType())
])

df_tweet = spark.read.json(path=path, multiLine=True, )


fields = ['id', 'created_at', 'full_text']

#%%
df_tweet = df_tweet.select('*', 'tweet.id', "tweet.full_text", "tweet.in_reply_to_status_id_str", "tweet.created_at")\
    # .drop("tweet")

#%%
# Checking for null values
df_tweet.where(col("id").isNull()).count()
df_tweet.count()

#%%
df_tweet = df_tweet.withColumn("created_at",
                               from_unixtime(unix_timestamp(col("created_at"), "EEE MMM dd HH:mm:ss ZZZZ yyyy"),
              "yyyy-MM-dd HH:mm:ss.SSSSSS"))

 # %%
# Get the hour distribution
hours = df_tweet.groupby(hour("created_at")).count().collect()
days = df_tweet.groupby(dayofmonth("created_at")).count().collect()

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
tmp = dict(days)

for i in range(1, 32):
    if i not in tmp:
        tmp[i] = 0

tmp = dict(sorted(tmp.items()))

fig, ax = plt.subplots(figsize=(15, 10))
ax.set_ylim([1000, 10_000])
ax.margins(x=0)
ax.plot(tmp.keys(), tmp.values())
ax.set_title("Day of months")
fig.show()

#%%

first = df_tweet.orderBy(col("created_at")).first()
last = df_tweet.orderBy(col("created_at").desc()).first()
between = 5349
#%%

summed = 0
for day in days:
    summed += day['count']
print(f"Average tweet {df_tweet.count() / between}")
# %%
replied_count = df_tweet.where(col("in_reply_to_status_id_str").isNotNull()).count()
print(f"Replied to count {replied_count}")
print(f"Percentage {(replied_count / df_tweet.count()) * 100}%")

# %%
# NB elendig kode
tweet_ids = df_tweet.rdd.map(lambda x: x['id']).collect()
replied_to_id_in_df = df_tweet.where(
    col("in_reply_to_status_id_str").isin(tweet_ids)).count()
print(f"Replied id exists in df {replied_to_id_in_df}")


#%%
#Slow code
not_in_df = df_tweet.where(col("in_reply_to_status_id_str").isin(tweet_ids) == False)
print(f"Not in df {not_in_df}")

#%%


df_tweet.where((year("Created_at") == 2022) & (month("Created_at") == 10) & (dayofmonth("Created_at") == 15) & (hour("Created_at") > 10)& (hour("Created_at") < 12) ).drop("tweet").show(truncate=False)