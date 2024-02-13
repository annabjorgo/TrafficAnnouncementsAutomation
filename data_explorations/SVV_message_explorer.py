import pyspark.pandas.plot.matplotlib
from pyspark.sql import SparkSession
import pyspark.sql.functions as sf
import matplotlib.pyplot as plt
import time

# %%

# path = "data/SVV/ML-Situations/unzipped/ML Situations000000000000.json"
path = "data/SVV/ML-Situations/unzipped"
spark = SparkSession.builder.config("spark.driver.memory", "15g").appName("TAA_SVV_json").getOrCreate()

# %%
df = spark.read.json(path=path)
df.count()
# Remove duplicates
df = (df.select(["*", "locations.locationDescriptor"])
    .dropDuplicates(
    ["overallStartTime", "description", "recordId"]))\
    #TODO: NY METODE .dropDuplicates(['overallStartTime', 'description', 'recordId'])

#%%
df = df.groupby("situationId").agg(sf.collect_list("description").alias("description")
                                        , sf.collect_list("recordId").alias("recordId")
                                        , sf.collect_list("creationTime").alias("creationTime")
                                        , sf.collect_list("dataProcessingNote").alias("dataProcessingNote")
                                        , sf.collect_list("datexVersion").alias("datexVersion")
                                        , sf.collect_list("importTime").alias("importTime")
                                        , sf.collect_list("lastPublicationTime").alias("lastPublicationTime")
                                        , sf.collect_list("locations").alias("locations")
                                        , sf.collect_list("overallEndTime").alias("overallEndTime")
                                        , sf.collect_list("overallStartTime").alias("overallStartTime")
                                        , sf.collect_list("recordSubtypes").alias("recordSubtypes")
                                        , sf.collect_list("recordType").alias("recordType")
                                        , sf.collect_list("recurringTimePeriods").alias("recurringTimePeriods")
                                        , sf.collect_list("version").alias("version")
                                        , sf.collect_list("versionTime").alias("versionTime")
                                        )

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
            sf.col(col_name).isNotNull())).withColumn('ts', sf.to_timestamp(col_name))

    years = df_tmp.groupby(sf.year(col_name)).count().collect()
    months = df_tmp.groupby(sf.month(col_name)).count().collect()
    days = df_tmp.groupby(sf.dayofmonth(col_name)).count().collect()
    weekday = df_tmp.groupby(sf.dayofweek(col_name)).count().collect()
    hours = df_tmp.groupby(sf.hour(col_name)).count().collect()
    print(hours)

    def visualize(input, name):
        import tikzplotlib

        tmp = dict(sorted(dict(input).items()))
        print(tmp)
        fig, ax = plt.subplots()
        ax.plot(tmp.keys(), tmp.values())
        fig.suptitle(f"Traffic messages grouped by {name}", fontsize=14, fontweight='bold')
        ax.set_title(col_name)
        # tikzplotlib.save(f"{col_name}-{name}")
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
explore_timestamp('lastPublicationTime')
explore_timestamp('overallStartTime')
explore_timestamp('overallEndTime')

# %%
tmp = df.where(sf.col('recordId').contains('NPRA')).withColumn('ts', sf.to_timestamp('creationTime')).groupby(
    sf.year('ts')).count().collect()
tmp = dict(sorted(dict(tmp).items()))
fig, ax = plt.subplots()
ax.plot(tmp.keys(), tmp.values())
fig.suptitle(f"aaa", fontsize=14, fontweight='bold')
fig.show()


# %%

# Average length of text
def note_insight():
    print(
        f'Average length of dataProcessingNote: {df.where(sf.col("dataProcessingNote").isNotNull()).withColumn("length_of_note", sf.length("dataProcessingNote")).selectExpr("avg(length_of_note)").first()[0]}')

    print(f"Rows where dataProcessingNote contains '|'")
    df.where((sf.col('dataProcessingNote').isNotNull()) & (sf.col('dataProcessingNote').contains('|'))).show(
        truncate=False)

    print("Rows split on '|', index 1, grouped and counted")
    df.where(sf.col('dataProcessingNote').isNotNull()) \
        .select(sf.substring_index(sf.col('dataProcessingNote'), '|', 1).alias("split")) \
        .groupby('split') \
        .count() \
        .orderBy(sf.col("count").desc()) \
        .show(truncate=False)

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

# %%
print(
    f'Average length of description: {df.where(sf.col("description").isNotNull()).withColumn("length_of_description", sf.length("description")).selectExpr("avg(length_of_description)").first()[0]}')

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
                  "creationTime",
                  "locations"
                  ).show(
    n=3000, truncate=False)
# %%
df.select(sf.col('locations')).show(truncate=False)

# %%
# Stavanger hendelsen. Får opp to stk
df.where(sf.col("recordId") == "eeda5d6e-a0c6-4edb-a1cc-125e8cf2ce4e_1").show()
# %%
df.groupby("version").count().orderBy(sf.col("count").desc()).show()


# %%
def double_df_search():
    days = [  10]
    year = 2022
    search_month = 3
    start_hour = 8
    end_hour = 14
    for d in days:
        a = df_tweet.where((sf.year("Created_at") == year)
                           & (sf.month("Created_at") == search_month)
                           & (sf.dayofmonth("Created_at") == d)
                           & (sf.hour("Created_at") > start_hour)
                           & (sf.hour("Created_at") < end_hour)) \
            .drop("tweet").toPandas()
        b = df.where((sf.year('overallStartTime') == year)
                     & (sf.month('overallStartTime') == search_month)
                     & (sf.dayofmonth('overallStartTime') == d)
                     & (start_hour < sf.hour('overallStartTime'))
                     & (sf.hour('overallStartTime') < end_hour)
                     # & (sf.col('description').isNotNull())
                     ).select(
            "dataProcessingNote",
            "description",
            "locations.locationDescriptor",
            "locations.roadNumber",
            "locations",
            "overallStartTime",
            "creationTime",
            "recordId",
        ).toPandas()
        return a, b


searched_tweets, searched_svv = double_df_search()

# %%
night_df = df.where((sf.hour("overallStartTime") > 22) & (sf.hour("overallStartTime") < 7))

# %%
most_ocr_note = (
    df
    .withColumn('dataProcessingNote', sf.explode(sf.split("dataProcessingNote", " ")))
    .groupBy('dataProcessingNote').count()
    .orderBy(sf.desc('count'))
)
top_note_dict = {r['dataProcessingNote']: r['count'] for r in most_ocr_note.head(100)}
print(top_note_dict)
# %%
most_ocr_desc = (
    df
    .withColumn('description', sf.explode(sf.split("description", " ")))
    .groupBy('description').count()
    .orderBy(sf.desc('count'))
)
top_descp_dict = {r['description']: r['count'] for r in most_ocr_desc.head(100)}
print(top_note_dict)


# %%
def gen_tables_splitted():
    df.where(sf.col('dataProcessingNote').isNotNull()) \
        .select(sf.substring_index(sf.col('dataProcessingNote'), '|', 1).alias("split")) \
        .groupby('split') \
        .count() \
        .orderBy(sf.col("count").desc()) \
        .show(truncate=False)

    df.where(sf.col('description').isNotNull()) \
        .select(sf.substring_index(sf.col('description'), '|', 1).alias("split")) \
        .groupby('split') \
        .count() \
        .orderBy(sf.col("count").desc()) \
        .show(truncate=False)
    night_df.where(sf.col('dataProcessingNote').isNotNull()) \
        .select(sf.substring_index(sf.col('dataProcessingNote'), '|', 1).alias("split")) \
        .groupby('split') \
        .count() \
        .orderBy(sf.col("count").desc()) \
        .show(truncate=False)

    night_df.where(sf.col('description').isNotNull()) \
        .select(sf.substring_index(sf.col('description'), '|', 1).alias("split")) \
        .groupby('split') \
        .count() \
        .orderBy(sf.col("count").desc()) \
        .show(truncate=False)


gen_tables_splitted()

# %%
nrk_ids = ['1579429693274853377',
           '1579390029402902529',
           '1579379449925373954',
           '1579803250299392001',
           '1579799343644958721',
           '1579795847239593986',
           '1579794832322867201',
           '1579787644778995712',
           '1579785075826503681',
           '1579776407529721856',
           '1579771779417243655',
           '1580151163361849344',
           '1580114486308241408']

svv_ids = ['f859e2e0-b192-4a29-896a-6790c161a816_2',
          '1b3ebabc-d52a-45ff-9851-f20879d0cb60_2',
          '47dd55b2-be1a-45b5-9f32-c59dee38c3d3_1',
          '050e7481-d649-445e-89e6-c4b77ae3816d_1',
          '75e7d320-2cd0-48d2-8dd0-7f2fd8b8c716_1',
          'f9632a1b-9a37-44ab-a64e-4d39577f6971_1',
          'c9e62f41-e7b3-47d6-8985-8357eb4200dd_1',
          '8028009e-4c79-43a5-89a0-3fe9074c1396_2',
          'b807f3f9-5964-4cf6-8788-5a24f117facd_2',
          '1be42be1-08f7-4d7e-a9fa-8f194af70461_1',
          '90c1e706-daa1-4327-82ff-583241c0eb77_1',
          'b7ee977c-0b94-418e-bcc7-e7c45ec55a32_1',
          'bdf91359-cd32-44dd-895a-52e37458d600_1', ]

for nrk,svv in zip(nrk_ids, svv_ids):
    print(df.where(sf.col("recordId") == svv).first())
    print(df_tweet.where(sf.col("id") == nrk).first())

# %%
