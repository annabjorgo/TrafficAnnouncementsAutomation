from datetime import timezone
import datetime

import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, unix_timestamp

from data_explorations.link_dict import link_dict

from sentence_transformers import SentenceTransformer, util


def pipeline_starter():
    spark = SparkSession.builder.config("spark.executor.memory", "15g").config("spark.driver.memory", "15g").appName(
        "TAA_SVV_json").getOrCreate()
    spark.sql("set spark.sql.legacy.timeParserPolicy=LEGACY")

    svv_path = "data/SVV/ML-Situations/unzipped"
    tweet_path = [
        "data/NRK/raw/tweets.js",
        "data/NRK/raw/tweets-part1.js",
        "data/NRK/raw/tweets-part2.js"
    ]

    df = spark.read.json(path=svv_path)
    df_tweet = spark.read.json(path=tweet_path, multiLine=True, )
    df_tweet = df_tweet.select('*', 'tweet.id', "tweet.full_text", "tweet.in_reply_to_status_id_str",
                               "tweet.created_at",
                               "tweet.in_reply_to_user_id")
    df_tweet = df_tweet.withColumn("created_at",
                                   from_unixtime(unix_timestamp(sf.col("created_at"), "EEE MMM dd HH:mm:ss ZZZZ yyyy"),
                                                 "yyyy-MM-dd HH:mm:ss.SSSSSS"))

    return df, df_tweet, link_dict, spark


def filter_pyspark_df(df, df_tweet, link_dict, spark):
    link_df = spark.createDataFrame(link_dict.items(), schema=["nrk_id", "svv_id"])
    df = df.withColumn("dataProcessingNote",
                       sf.regexp_replace("dataProcessingNote", "Merk at sluttiden er usikker og kan endres.", "")) \
        .withColumn("dataProcessingNote", sf.regexp_replace("dataProcessingNote", "[|]", " "))
    df = df.withColumn("concat_text", sf.concat_ws("", df.dataProcessingNote, df.description))
    df = df.where(sf.year("overallStartTime") > 2010)
    # Note: only using first location descriptor
    df = df.withColumn("concat_text", sf.concat_ws(", ", df.locations.locationDescriptor[0], df.concat_text))
    link_df = link_df.join(df_tweet, link_df.nrk_id == df_tweet.id).select("nrk_id", "svv_id",
                                                                           "full_text", "created_at").withColumnRenamed(
        "full_text", "nrk_text").withColumnRenamed("created_at", "nrk_created_at") \
        # .join(df, link_df.svv_id == df.recordId).select("nrk_id", "svv_id", "nrk_text", "concat_text").withColumnRenamed("concat_text", "svv_text")
    # todo: This might be incorrect since I am dropping duplicates
    link_df = link_df.join(df.dropDuplicates(["recordId"]), link_df.svv_id == df.recordId).select("nrk_id", "svv_id",
                                                                                                  "nrk_text",
                                                                                                  "nrk_created_at",
                                                                                                  "situationId")

    return df, link_df


def transfer_to_pandas(df, link_df, model):
    pd_link = link_df.toPandas()
    pd_link['nrk_embed'] = model.encode(pd_link['nrk_text'], show_progress_bar=True).tolist()
    pd_df = df.select("recordId", "concat_text", "overallStartTime", "situationId").toPandas()
    # pd_df['svv_embed'] = model.encode(pd_df['concat_text'], show_progress_bar=True).tolist()
    pd_df['overallStartTime'] = pd.to_datetime(pd_df['overallStartTime'])
    pd_link['nrk_created_at'] = pd.to_datetime(pd_link['nrk_created_at'])
    pd_link['nrk_created_at'] = pd_link['nrk_created_at'].dt.tz_localize(timezone.utc)
    return pd_df, pd_link


def align_data_sentence_transformer(q_df, svv_df, timedelta, model):
    alignment = []
    time_window = pd.Timedelta(hours=timedelta)

    for nrk_it in q_df.itertuples():
        max_sim = 0
        svv_id = 0
        svv_situation = 0
        search_df = svv_df[abs(svv_df['overallStartTime'] - nrk_it.nrk_created_at) <= time_window].copy()
        search_df['svv_embed'] = model.encode(search_df['concat_text'].tolist()).tolist()
        for svv_it in search_df.itertuples():
            sim = util.pytorch_cos_sim(nrk_it.nrk_embed, svv_it.svv_embed).item()
            if sim > max_sim:
                max_sim = sim
                svv_id = svv_it.recordId
                svv_situation = svv_it.situationId
        alignment.append(
            {"nrk_id": nrk_it.nrk_id, "recordId": svv_id, "situationId": svv_situation, "similarity": max_sim})

    tmp_df = pd.DataFrame(alignment, columns=(['nrk_id', 'prediction_svv_id', 'situationId', 'cos_sim']))
    file_name = f'data/pipeline_runs/alignment d:{datetime.datetime.now().day} m:{datetime.datetime.now().month} h:{datetime.datetime.now().hour}.csv'
    tmp_df.to_csv(
        file_name,
        index=False)
    return alignment

def align_data_bm25(q_df, svv_df, timedelta, model):
    alignment = []
    time_window = pd.Timedelta(hours=timedelta)

    for nrk_it in q_df.itertuples():
        max_sim = 0
        svv_id =

def measure_accuracy(aligned_list):
    def correct_by_recordId(input_list, pd_link):
        correct_recordId_list = []
        for prediction in input_list:
            nrk_id = prediction['nrk_id']
            corr_recordId = str(pd_link[pd_link['nrk_id'] == nrk_id].iloc[0]['svv_id']).strip()
            pred_recordId = str(prediction['recordId']).strip()

            if corr_recordId == pred_recordId:
                correct_recordId_list.append(prediction)
        return correct_recordId_list

    def correct_by_situationId(input_list, pd_link):
        correct_situationId_list = []
        for prediction in input_list:
            nrk_id = prediction['nrk_id']
            corr_situationId = str(pd_link[pd_link['nrk_id'] == nrk_id].iloc[0]['situationId']).strip()
            pred_situationId = str(prediction['situationId']).strip()

            if corr_situationId == pred_situationId:
                correct_situationId_list.append(prediction)
        return correct_situationId_list

    def correct_by_location(input_list, df, pd_link):
        group_df = df.groupby(sf.year("overallStartTime"), sf.month("overallStartTime"), sf.day("overallStartTime"),
                              sf.col("locations.coordinatesForDisplay")).agg(
            sf.collect_set(sf.col("recordId")).alias("ids"))
        group_df.persist()

        correct_location_list = []
        for prediction in input_list:
            nrk_id = prediction['nrk_id']
            corr_recordId = str(pd_link[pd_link['nrk_id'] == nrk_id].iloc[0]['svv_id']).strip()
            pred_recordId = str(prediction['recordId']).strip()

            if not group_df.where(
                    sf.array_contains("ids", corr_recordId) & sf.array_contains("ids", pred_recordId)).isEmpty():
                correct_location_list.append(prediction)
        group_df.unpersist()
        return correct_location_list

    situation_id_list = correct_by_situationId(aligned_list)
    record_id_list = correct_by_recordId(aligned_list)
    location_list = correct_by_location(aligned_list)

    correct_list = list({v['nrk_id']: v for v in (location_list + record_id_list + situation_id_list)}.values())
    incorrect_list = [item for item in aligned_list if item not in correct_list]
    avg_correct_similarity = sum(it['similarity'] for it in correct_list) / len(correct_list)

    print(f"Accuracy for situationId: {len(situation_id_list) / len(aligned_list)}")
    print(f"Accuracy for recordId: {len(record_id_list) / len(aligned_list)}")
    print(f"Accuracy for location: {len(location_list) / len(aligned_list)}")
    print(f"Avg similarity: {avg_correct_similarity}")

    return correct_list, incorrect_list


def check_incorrect(incorrect_list, pd_link, pd_df):
    analysed_list = []
    for i, it in enumerate(incorrect_list):
        try:
            corr_nrk_id = it['nrk_id']
            corr_svv_id = pd_link[pd_link['nrk_id'] == corr_nrk_id].iloc[0]['svv_id']
            pred_svv_id = it['recordId']

            nrk_corr_text = pd_link[pd_link['nrk_id'] == corr_nrk_id].iloc[0]['nrk_text']
            # fixme: this only takes the first record with the svv_id

            svv_corr_text = pd_df[pd_df['recordId'] == corr_svv_id].iloc[0]['concat_text']
            svv_pred_text = pd_df[pd_df['recordId'] == pred_svv_id].iloc[0]['concat_text']

            analysed_list.append(
                {"correct_nrk_id_text": (corr_nrk_id, nrk_corr_text),
                 "correct_svv_id_text": (corr_svv_id, svv_corr_text),
                 "predicted_svv_id_text": (pred_svv_id, svv_pred_text), "similarity": it['similarity']})
        except:
            pass
    return analysed_list


def print_incorrect(incorrect):
    for it in incorrect:
        print(
            f'Correct nrk id and text: {it["correct_nrk_id_text"][0].strip()}, {it["correct_nrk_id_text"][1].strip()}')
        print(f'Correct svv id and text: {it["correct_svv_id_text"][0]}, {it["correct_svv_id_text"][1]}')
        print(f'Predicted svv id and text: {it["predicted_svv_id_text"][0]}, {it["predicted_svv_id_text"][1]}')
        print(f'Similarity: {it["similarity"]}')
        print("\n")

