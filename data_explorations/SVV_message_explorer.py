import pyspark.pandas.plot.matplotlib
from pyspark.sql import SparkSession
import pyspark.sql.functions as sf
import matplotlib.pyplot as plt
import time

# %%

# path = "data/SVV/ML-Situations/unzipped/ML Situations000000000000.json"
path = "data/SVV/ML-Situations/unzipped"
spark = SparkSession.builder.appName("TAA_SVV_json").getOrCreate()
# %%
df = spark.read.json(path=path)
#Remove duplicates
df = df.select(["*", "locations.locationDescriptor"]).dropDuplicates(["creationTime", "locationDescriptor", "dataProcessingNote"])

# %%
df.show()
# %%
df.count()

# %%
# Verify that a row either has description or dataProcessingNote
print(f"Amount of rows where description is null: {df.where(sf.col('description').isNull()).count()}")
print(f"Amount of rows where dataProcessingNote is null: {df.where(sf.col('dataProcessingNote').isNull()).count()}")
print(
    f"Amount of rows where description is not null and dataProcessingNote is not null: {df.where(sf.col('description').isNotNull() & sf.col('dataProcessingNote').isNotNull()).count()}")
print(
    f"Amount of rows where description is null and dataProcessingNote is not null: {df.where(sf.col('description').isNull() & sf.col('dataProcessingNote').isNotNull()).count()}")
print(
    f"Amount of rows where description is not null and dataProcessingNote is null: {df.where(sf.col('description').isNotNull() & sf.col('dataProcessingNote').isNull()).count()}")
print(
    f"Amount of rows where both description and dataProcessingNote is null: {df.where(sf.col('description').isNull() & sf.col('dataProcessingNote').isNull()).count()}")

# %%
# Show all record types
df.select(sf.col("recordType")).distinct().show(30)

# %%
# Not understanding what this field actually is
df.where(sf.size("forVehiclesWithCharacteristicsOf") > 0).select("forVehiclesWithCharacteristicsOf").show(
    truncate=False)

# %%
#
df.where(sf.col('recordId').isNotNull()) \
    .groupby(sf.col("recordId")) \
    .count() \
    .where(sf.col("count") > 1) \
    .select(sf.col('count')) \
    .sort(sf.col('count').desc()) \
    .show()
# %%
df.where(sf.col('recordId').isNotNull()) \
    .groupby(sf.col("recordId")) \
    .count() \
    .groupby(sf.col('count')) \
    .count().show()


# %%
def explore_timestamp(col_name: str, df=df):
    start = time.time()
    num_null = df.where(sf.col(col_name).isNull()).count()
    print(f"Number of null rows: {num_null}")
    df_tmp = df.filter(
        (sf.year(col_name) < 2040) & (sf.year(col_name) > 2005) & (
            sf.col(col_name).isNotNull()) & (sf.col("description").isNotNull())).withColumn('ts',
                                                                                            sf.to_timestamp(col_name))
    years = df_tmp.groupby(sf.year('ts')).count().collect()
    months = df_tmp.groupby(sf.month('ts')).count().collect()
    days = df_tmp.groupby(sf.dayofmonth('ts')).count().collect()
    weekday = df_tmp.groupby(sf.dayofweek('ts')).count().collect()
    hours = df_tmp.groupby(sf.hour('ts')).count().collect()

    def visualize(input, name):
        tmp = dict(sorted(dict(input).items()))
        fig, ax = plt.subplots()
        ax.plot(tmp.keys(), tmp.values())
        fig.suptitle(f"Traffic messages grouped by {name}", fontsize=14, fontweight='bold')
        ax.set_title(col_name)
        fig.show()

    end = time.time()
    print(f"execution {end - start}")

    visualize(hours, "hours")
    visualize(days, "days")
    visualize(weekday, "weekdays")
    visualize(months, "months")
    visualize(years, "years")


# %%
explore_timestamp('creationTime')


# explore_timestamp('lastPublicationTime')
# explore_timestamp('overallStartTime')
# explore_timestamp('overallEndTime')


# %%

# Average length of text
def note_insight():
    print(
        f'Average length of dataProcessingNote: {df.where(sf.col("dataProcessingNote").isNotNull()).withColumn("length_of_note", sf.length("dataProcessingNote")).selectExpr("avg(length_of_note)").first()[0]}')

    print(f"Rows where dataProcessingNote contains '|'")
    df.where((sf.col('dataProcessingNote').isNotNull()) & (sf.col('dataProcessingNote').contains('|'))).show(
        truncate=False)

    print("Rows split on '|', grouped and counted")
    df.where(sf.col('dataProcessingNote').isNotNull()) \
        .select(sf.substring_index(sf.col('dataProcessingNote'), '|', 1).alias("split")) \
        .groupby('split') \
        .count() \
        .orderBy(sf.col("count").desc())\
        .show()

    print(
        f"Rows containing '|': {df.where((sf.col('dataProcessingNote').isNotNull()) & (sf.col('dataProcessingNote').contains('|'))).count()}")
    print(
        f"Rows split by '|', grouped, counted and counted amount : {df.where((sf.col('dataProcessingNote').isNotNull()) & (sf.col('dataProcessingNote').contains('|')).isNotNull()).select(sf.substring_index(sf.col('dataProcessingNote'), '|', 1).alias('split')).groupby('split').count().count()}")
    print(
        f"Percentage of total rows containing '|': {df.where((sf.col('dataProcessingNote').isNotNull()) & (sf.col('dataProcessingNote').contains('|'))).count() / df.count()}")
    print(
        f"Percentage of rows where dataProcessingNote is not null containing '|': {df.where((sf.col('dataProcessingNote').isNotNull()) & (sf.col('dataProcessingNote').contains('|'))).count() / df.where(sf.col('dataProcessingNote').isNotNull()).count()}")


note_insight()
# %%

df.where(sf.length('description') > 170).show(truncate=False)
# %%
df.where((sf.year('overallStartTime') == 2022) & (sf.month('overallStartTime') == 10) & (
        sf.dayofmonth('overallStartTime') == 15) & (10 < sf.hour('overallStartTime')) & (
                 sf.hour('overallStartTime') < 12)
         # & (sf.col('description').isNotNull())
         ).select("recordId",
                  "dataProcessingNote",
                  "description",
                  "locations.locationDescriptor",
                  "locations.roadNumber",
                  "overallStartTime",
                  "creationTime"
                  ).show(
    n=3000, truncate=False)
# %%
df.select(sf.col('locations')).show(truncate=False)
# %%
print(df.count())
print(df.distinct().count())
#%%
df.groupby("version").count().orderBy(sf.col("count").desc()).show()

#%%
