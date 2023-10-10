from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_utc_timestamp, from_unixtime, unix_timestamp
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format, hour, dayofmonth
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType, TimestampType, \
    LongType

path = [
        "/Users/madslun/Documents/Fag-Programmering/TrafficAnnouncementsAutomation/tweets.js",
        "/Users/madslun/Documents/Fag-Programmering/TrafficAnnouncementsAutomation/tweets-part1.js",
        "/Users/madslun/Documents/Fag-Programmering/TrafficAnnouncementsAutomation/tweets-part2.js"
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

df = spark.read.json(path=path, multiLine=True,)


fields = ['id', 'created_at', 'full_text']

#%%
df = df.select('*', 'tweet.id', "tweet.full_text", "tweet.in_reply_to_status_id_str", "tweet.created_at")\
    # .drop("tweet")

#%%
# Checking for null values
df.where(col("id").isNull()).count()
df.count()

#%%
df = df.withColumn("created_at",
              from_unixtime(unix_timestamp(col("created_at"), "EEE MMM dd HH:mm:ss ZZZZ yyyy"),
              "yyyy-MM-dd HH:mm:ss.SSSSSS"))

 # %%
# Get the hour distribution
hours = df.groupby(hour("created_at")).count().collect()
days = df.groupby(dayofmonth("created_at")).count().collect()

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

first = df.orderBy(col("created_at")).first()
last = df.orderBy(col("created_at").desc()).first()
between = 5349
#%%

summed = 0
for day in days:
    summed += day['count']
print(f"Average tweet {df.count()/between}")
# %%
replied_count = df.where(col("in_reply_to_status_id_str").isNotNull()).count()
print(f"Replied to count {replied_count}")
print(f"Percentage {(replied_count/df.count()) * 100}%")

# %%
# NB elendig kode
tweet_ids = df.rdd.map(lambda x: x['id']).collect()
replied_to_id_in_df = df.where(
    col("in_reply_to_status_id_str").isin(tweet_ids)).count()
print(f"Replied id exists in df {replied_to_id_in_df}")


#%%
#Slow code
not_in_df = df.where(col("in_reply_to_status_id_str").isin(tweet_ids) == False)
print(f"Not in df {not_in_df}")


