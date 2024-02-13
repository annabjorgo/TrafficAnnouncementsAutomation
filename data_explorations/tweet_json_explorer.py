import csv
from datetime import date, datetime

#import tweepy
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, from_utc_timestamp, from_unixtime, unix_timestamp, year, month
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, date_format, hour, dayofmonth
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, DateType, TimestampType, \
    LongType

tweet_path = [
    "data/NRK/raw/tweets.js",
    "data/NRK/raw/tweets-part1.js",
    "data/NRK/raw/tweets-part2.js"
]

spark = SparkSession.builder.appName("TAA_json").getOrCreate()
spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")
spark.conf.set("spark.sql.session.timeZone", "CET")
# %%
data_schema = StructType(fields=[
    # StructField("entities", ),
    StructField("id", LongType()),
    StructField("full_text", StringType()),
    StructField("in_reply_to_status_id_str", LongType()),
    StructField("created_at", StringType())
])

df_tweet = spark.read.json(path=tweet_path, multiLine=True, )
df_tweet = df_tweet.select('*', 'tweet.id', "tweet.full_text", "tweet.in_reply_to_status_id_str", "tweet.created_at",
                           "tweet.in_reply_to_user_id")
df_tweet = df_tweet.withColumn("created_at",
                               from_unixtime(unix_timestamp(col("created_at"), "EEE MMM dd HH:mm:ss ZZZZ yyyy"),
                                             "yyyy-MM-dd HH:mm:ss.SSSSSS"))

fields = ['id', 'created_at', 'full_text']

# %%
# .drop("tweet")

# %%
# Checking for null values
df_tweet.where(col("id").isNull()).count()
df_tweet.count()

# %%

# %%
# Get the hour distribution
hours = df_tweet.groupby(hour("created_at")).count().collect()
days = df_tweet.groupby(dayofmonth("created_at")).count().collect()

# %%
# Visualize tweet hourly
import matplotlib.pyplot as plt
import tikzplotlib

tmp = dict(hours)
for i in range(1, 25):
    if i not in tmp:
        tmp[i] = 0

tmp = dict(sorted(tmp.items()))

fig, ax = plt.subplots(figsize=(15, 10))
ax.margins(x=0)
ax.plot(tmp.keys(), tmp.values())
ax.set_title("Hourly tweets")
tikzplotlib.save(f"NRKtweets-total-hour.tex")
# fig.show()

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

# %%

first = df_tweet.orderBy(col("created_at")).first()
last = df_tweet.orderBy(col("created_at").desc()).first()
between = 5349
# %%

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
    ~col("in_reply_to_status_id_str").isin(tweet_ids)).count()
print(f"Replied id exists in df {replied_to_id_in_df}")

# %%
# Slow code
not_in_df = df_tweet.where(col("in_reply_to_status_id_str").isin(tweet_ids) == False)
print(f"Not in df {not_in_df}")

# %%


df_tweet.where((year("Created_at") == 2022)
               & (month("Created_at") == 10)
               & (dayofmonth("Created_at") == 15)
               & (hour("Created_at") > 10)
               & (hour("Created_at") < 12)) \
    .drop("tweet").show(truncate=False)
# %%
referenced_tweets = df_tweet.select("in_reply_to_status_id_str").distinct().na.drop()
tweet_ids_list = df_tweet.select("id")
unmatched_referenced_tweets_df = referenced_tweets.subtract(tweet_ids_list)

unmatched_referenced_tweets_df.limit(20).show()

# %%
csv_name = f"output-{date.today().day}-{date.today().month}-{datetime.today().hour}-{datetime.today().minute}.csv"
final_date = datetime(2023, 9, 27)
nrk_user_name = "NrkTrafikk"
nrk_user_id = "20629858"
fields = ['author_id', "tweet_id", "text", "referenced_tweets", "referenced_type", "referenced_user", "created_at"]

end_point = f"https://api.twitter.com/2/users/{nrk_user_id}/tweets"

client = tweepy.Client(consumer_key="FOXeW9BEaY5iFhtJ8aQd6eBiD",
                       consumer_secret="1GCitmiJeM8QoSFtvlrJ4iSz7MGLIbTReI6byVpYO4JvniljLo",
                       bearer_token="AAAAAAAAAAAAAAAAAAAAAK63qAEAAAAAwUKPI3dfF86J1hYU4xXqdlNB5ps%3D6CSrgKmFPQPdLtfnydCTFn0Ws0cLzoKGhxZjApYg8bhsepH5dF",
                       access_token="1704477155596390400-KG4af3anIWCpp4RqOLch1ceKolRKuf",
                       access_token_secret="Ilyg2C6hhmaQCphc8EibJ1ZG8LjM9MmEc9flicPHGSDko")


# Kode for å spørre om disse tweetsene:

def fetch_tweets(tweet_ids, batch_size=100):
    data_list = []
    for i in range(0, len(tweet_ids), batch_size):
        batch = tweet_ids[i:i + batch_size]
        try:
            res = client.get_tweets(ids=batch, tweet_fields=tweepy.PUBLIC_TWEET_FIELDS)
            extract_data(data_list, res)
        except Exception as e:
            save_csv(data_list)
            print(e)
    save_csv(data_list)
    return data_list


def save_csv(list_dict, csv_name=csv_name):
    with open(csv_name, 'w') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(list_dict)


def extract_data(data_list, res):
    for tweet in res.data:
        data_list.append(
            {fields[0]: tweet.author_id,
             fields[1]: tweet.id,
             fields[2]: tweet.text,
             fields[3]: "" if tweet.referenced_tweets is None else tweet.referenced_tweets[0].id,
             fields[4]: "" if tweet.referenced_tweets is None else tweet.referenced_tweets[0].type,
             fields[5]: "" if tweet.in_reply_to_user_id is None else tweet.in_reply_to_user_id,
             fields[6]: tweet.created_at
             })


# %%
# Retrieve ids from text string in tweets
import pyspark.sql.functions as sf

def filter_and_save_dataset(df_tweet=df_tweet):
    r = df_tweet.select("tweet", 'id', 'tweet.entities.urls.expanded_url', "tweet.full_text", 'created_at',
                    'in_reply_to_status_id_str', 'in_reply_to_user_id').where(
    (sf.col('expanded_url')[0].isNotNull()) & (~(sf.col('expanded_url')[0].contains("NRK"))
                                               & (sf.col("expanded_url")[0].contains("twitter"))
                                               )) \
    .select(
    sf.substring_index(sf.col('expanded_url')[0], 'status/', -1).alias('first'), 'id', 'tweet', 'full_text',
    'created_at', 'in_reply_to_status_id_str', 'in_reply_to_user_id').select(
    sf.substring_index(sf.col('first'), '?', 1).alias('id_in_text'), 'id',
    # 'tweet',
    'full_text', 'created_at',
    'in_reply_to_status_id_str', 'in_reply_to_user_id')

# Join the id from text to df
    df_tweet = df_tweet.join(r, df_tweet.id == r.id, "full").select(df_tweet['*'], r['id_in_text'])
    # Save df with new data
    df_tweet.where(sf.year('created_at') > 2015).toPandas().to_csv("data/NRK/filtered_json_data.csv")

    def query_data():
        to_query = r.select(sf.col("id_in_text"), sf.col("id_in_text").cast("long").isNotNull().alias("val"), 'id',
                    'full_text').where(
        sf.col("val") == True).orderBy(sf.col('created_at').desc()).rdd.map(lambda x: x['id_in_text']).collect()
# %%
filter_and_save_dataset()
# %%
filtered_df = df_tweet.where(sf.year('created_at') > 2015)
# %%
r =filtered_df.select("tweet", 'id', 'tweet.entities.urls.expanded_url', "tweet.full_text", 'created_at',
                    'in_reply_to_status_id_str', 'in_reply_to_user_id').where(
    (sf.col('expanded_url')[0].isNotNull()) & (~(sf.col('expanded_url')[0].contains("NRK"))
                                               & (sf.col("expanded_url")[0].contains("twitter"))
                                               )) \
    .select(
    sf.substring_index(sf.col('expanded_url')[0], 'status/', -1).alias('first'), 'id', 'tweet', 'full_text',
    'created_at', 'in_reply_to_status_id_str', 'in_reply_to_user_id').select(
    sf.substring_index(sf.col('first'), '?', 1).alias('id_in_text'), 'id',
    # 'tweet',
    'full_text', 'created_at',
    'in_reply_to_status_id_str', 'in_reply_to_user_id')