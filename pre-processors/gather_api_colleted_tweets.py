from pyspark.sql import SparkSession

files = [
    'output-24-11-10.csv',
    'output-26-11-14.csv',
    'output-26-11-15.csv',
    'output-26-11-15-20.csv',
    'output-26-11-15-51.csv',
    'output-26-11-16-39.csv',
    'output-26-11-16-49.csv',
    'output-26-11-17-2.csv',
    'output-26-11-17-4.csv',
    'output-26-11-17-46.csv',
    'output-26-11-17-50.csv',
    'output-26-11-17-59.csv',
    'output-27-11-11-43.csv',
    'output-27-11-11-44.csv']

spark = SparkSession.builder.appName("API_collected_NRK").getOrCreate()
df = spark.read.option("header", "true").csv(files, multiLine=True)

# %%
df = df.dropDuplicates()

# %%
ids = df.rdd.map(lambda x: x['tweet_id']).collect()

# %%
df.toPandas().to_csv("data/NRK/api_data.csv")

