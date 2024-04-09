import datetime
from datetime import timezone

import nltk
import numpy as np
import pandas as pd
import pyspark.sql.functions as sf
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime, unix_timestamp
from sentence_transformers import util
from tqdm import tqdm
from multiprocessing import Process

from data_explorations.link_dict import link_dict

nltk.download("stopwords")
from nltk.corpus import stopwords
from string import punctuation


def pipeline_starter():
    spark = SparkSession.builder.config("spark.executor.memory", "25g").config("spark.driver.memory", "25g").appName(
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
    df_tweet = df_tweet.where(sf.year('created_at') > 2015)
    df_tweet = df_tweet.filter(~sf.col("full_text").rlike(
        "^https:\/\/t\.co\/[a-zA-Z0-9]+$"))  # To remove the tweets where only a url is contained

    link_df = spark.createDataFrame(link_dict.items(), schema=["nrk_id", "svv_id"])

    df = df.withColumn("dataProcessingNote",
                       sf.regexp_replace("dataProcessingNote", "Merk at sluttiden er usikker og kan endres.", "")) \
        .withColumn("dataProcessingNote", sf.regexp_replace("dataProcessingNote", "[|]", " "))
    df = df.withColumn("concat_text", sf.concat_ws("", df.dataProcessingNote, df.description))
    df = df.where(sf.year("overallStartTime") > 2010)
    # todo: note; only using first location descriptor
    df = df.withColumn("concat_text", sf.concat_ws(", ", df.locations.locationDescriptor[0], df.concat_text))

    link_df = link_df.join(df_tweet, link_df.nrk_id == df_tweet.id).select("nrk_id", "svv_id",
                                                                           "full_text", "created_at").withColumnRenamed(
        "full_text", "nrk_text").withColumnRenamed("created_at", "nrk_created_at")
    # todo: This might be incorrect since I am dropping duplicates
    link_df = link_df.join(df.dropDuplicates(["recordId"]), link_df.svv_id == df.recordId).select("nrk_id", "svv_id",
                                                                                                  "nrk_text",
                                                                                                  "nrk_created_at",
                                                                                                  "situationId")

    return df, link_df


def split_sentences(text_list):
    return [x.split(" ") for x in text_list]


def join_sentences(text_list):
    return [' '.join(x) for x in text_list]


def remove_stopwords_punctuation(text_list):
    text = [x.split(" ") for x in text_list]
    stp = stopwords.words("norwegian")
    punc = list(punctuation)
    for i, arr in enumerate(text):
        text[i] = [word for word in arr if word not in stp and word not in punc]
    return text


def transfer_to_pandas_test(df, link_df):
    pd_link = link_df.toPandas()
    pd_df = df.select("recordId", "concat_text", "overallStartTime", "situationId").toPandas()
    pd_df['overallStartTime'] = pd.to_datetime(pd_df['overallStartTime'])
    pd_link['nrk_created_at'] = pd.to_datetime(pd_link['nrk_created_at'])
    pd_link['nrk_created_at'] = pd_link['nrk_created_at'].dt.tz_localize(timezone.utc)
    return pd_df, pd_link


def embed_pandas(pd_link, model):
    pd_link['nrk_embed'] = model.encode(pd_link['nrk_text'], show_progress_bar=True).tolist()
    return pd_link


def embed_nrk(df_tweet, model):
    df_tweet['nrk_embed'] = model.encode(df_tweet['full_text'], show_progress_bar=True).tolist()
    return df_tweet


def transform_for_alignment_e5(df, df_tweet):
    df_tweet = df_tweet.toPandas()
    df_tweet['nrk_id'] = df_tweet['id']
    pd_df = df.select("recordId", "concat_text", "overallStartTime", "situationId").toPandas()
    pd_df['overallStartTime'] = pd.to_datetime(pd_df['overallStartTime'])
    df_tweet['nrk_created_at'] = pd.to_datetime(df_tweet['created_at'])
    df_tweet['nrk_created_at'] = df_tweet['nrk_created_at'].dt.tz_localize(timezone.utc)
    return pd_df, df_tweet


def align_multi(q_df, svv_df, timedelta):
    alignment = []
    time_window = pd.Timedelta(hours=timedelta)
    processes = [Process(target=task, args=(nrk_it, svv_df, time_window, alignment)) for nrk_it in q_df.itertuples()]
    for process in tqdm(processes):
        process.start()

    for process in tqdm(processes):
        process.join()

    return alignment


def task(nrk_it, svv_df, time_window, alignment):
    search_df = svv_df[abs(svv_df['overallStartTime'] - nrk_it.nrk_created_at) <= time_window].copy()
    sim = util.pytorch_cos_sim(nrk_it.nrk_embed, search_df['svv_embed'].tolist())
    pos = np.argmax(sim).item()
    alignment.append((sim[0][pos], search_df.iloc[pos].recordId, search_df.iloc[pos].situationId))


def align_data(q_df, svv_df, timedelta, model, sim_func):
    alignment = []
    no_alignment = []
    time_window = pd.Timedelta(hours=timedelta)

    for nrk_it in tqdm(q_df.itertuples(), total=q_df.shape[0]):
        search_df = svv_df[abs(svv_df['overallStartTime'] - nrk_it.nrk_created_at) <= time_window].copy()
        if len(search_df) == 0:
            print(f"Found no matching id for {nrk_it.nrk_id}")
            no_alignment.append(
                {"nrk_id": nrk_it.nrk_id, "recordId": svv_id, "situationId": svv_situation, "svv_text": svv_text,
                 "nrk_text": nrk_it.full_text})
            continue

        max_sim, svv_id, svv_situation, svv_text, svv_start_time = sim_func(model, nrk_it, search_df)
        alignment.append(
            {"nrk_id": nrk_it.nrk_id, "recordId": svv_id, "situationId": svv_situation, "similarity": max_sim,
             "svv_text": svv_text, "nrk_text": nrk_it.full_text, "svv_ts": svv_start_time,
             "nrk_ts": nrk_it.nrk_created_atm})

    tmp_df = pd.DataFrame(alignment)
    file_name = f'data/pipeline_runs/alignment d:{datetime.datetime.now().day} m:{datetime.datetime.now().month} h:{datetime.datetime.now().hour}.csv'
    tmp_df.to_csv(
        file_name,
        index=False)
    return alignment


def max_sim_sentence_transformer(model, nrk_it, search_df):
    search_df['svv_embed'] = model.encode(search_df['concat_text'].tolist()).tolist()
    sim = util.pytorch_cos_sim(nrk_it.nrk_embed, search_df['svv_embed'].tolist())
    pos = np.argmax(sim).item()
    return sim[0][pos], search_df.iloc[pos].recordId, search_df.iloc[pos].situationId


def max_sim_sentence_transformer_precomputed(model, nrk_it, search_df):
    sim = util.pytorch_cos_sim(nrk_it.nrk_embed, search_df['svv_embed'].tolist())
    pos = np.argmax(sim).item()
    return sim[0][pos].item(), search_df.iloc[pos].recordId, search_df.iloc[pos].situationId, search_df.iloc[
        pos].concat_text, search_df.iloc[pos].overallStartTime


def max_sim_bm25(model, nrk_it, search_df):
    out = model(search_df['concat_text'].tolist())
    q = str(nrk_it.nrk_text).split(" ")
    scores = out.get_scores(q)
    pos = np.argmax(scores)
    return scores[pos], search_df.iloc[pos].recordId, search_df.iloc[pos].situationId


def max_sim_jaccard(model, nrk_it, search_df):
    sim_arr = []
    a = set(nrk_it.nrk_text.lower().split(" "))
    for it in search_df['concat_text'].tolist():
        it = [x.lower() for x in it]
        b = set(it)
        inter = a.intersection(b)
        uni = a.union(b)
        sim_arr.append(float(len(inter) / len(uni)))
    pos = np.argmax(sim_arr)
    return sim_arr[pos], search_df.iloc[pos].recordId, search_df.iloc[pos].situationId
