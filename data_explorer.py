#%%
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format, hour
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType, TimestampType, LongType
path = "/Users/madslun/Documents/Fag:Programmering/TrafficAnnouncementsAutomation/output-3-10-.csv"

#%%
data_schema = StructType(fields=[
    StructField("Author_id", IntegerType()),
    StructField("Tweet_id", LongType()),
    StructField("Text", StringType()),
    StructField("Refrenced_tweet", LongType()),
    StructField("Referenced_type", StringType()),
    StructField("Referenced_user", StringType()),
    StructField("Created_at", StringType())
])

spark = SparkSession.builder.appName("TAA").getOrCreate()


df = spark.read.schema(data_schema).option("header", "true").option("multiline", "true").csv(path=path)

#%%

df.limit(5).show()

#%%

# Checking for null values
df.where(col("Tweet_id").isNotNull()).count()

#%%
# Convert to CET/local timezone
df = df.withColumn("Created_at_local", date_format(col(("Created_at")), "yyyy-MM-dd HH:mm:ss.SSSX"))

#%%
# Get the hour distribution
hours = df.groupby(hour("Created_at_local")).count().collect()

#%%
#Visualize tweet hourly
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