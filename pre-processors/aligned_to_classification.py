# %%
import datetime
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pipelines.alignment_pipeline_utils import pipeline_starter, filter_pyspark_df, transform_for_alignment_e5


def load_and_preproc_data():
    aligned_df = pd.read_csv("data/pipeline_runs/alignment/alignment d:23 m:4 h:16.csv")
    annotated_df = pd.read_csv("data/pipeline_runs/alignment/annotated_alignment d:10 m:4 h:17.csv")
    no_df = pd.read_csv("data/pipeline_runs/alignment/no_alignment d:23 m:4 h:16.csv")

    rows_to_publish = aligned_df[aligned_df['similarity'] > threshold][
        ['recordId', 'situationId', 'svv_text', 'svv_ts']].copy()
    rows_to_publish = rows_to_publish.rename(columns={"svv_text": 'concat_text', "svv_ts": 'overallStartTime'})
    rows_to_publish['post'] = 1
    rows_to_publish['overallStartTime'] = pd.to_datetime(rows_to_publish['overallStartTime'])

    published_recordIds = list(set(list(aligned_df['recordId'])))
    published_situationIds = list(set(list(aligned_df['situationId'])))

    rows_not_publish = pd_df[
        ~pd_df['recordId'].isin(published_recordIds) & ~pd_df['situationId'].isin(published_situationIds)]
    rows_not_publish['post'] = 0

    return rows_to_publish, rows_not_publish


def check_overlap(rows_to_publish, rows_not_publish):
    publish_intersection_recordId = set(rows_to_publish['recordId']).intersection(set(rows_not_publish['recordId']))
    publish_intersection_situationId = set(rows_to_publish['situationId']).intersection(
        set(rows_not_publish['situationId']))

    assert len(publish_intersection_recordId) == 0
    assert len(publish_intersection_situationId) == 0


def no_night_100k_as_negative():
    print(category)

    test = pd.read_csv(f"data/pipeline_runs/classification/{category}/static_test.csv")
    test_night = pd.read_csv("data/pipeline_runs/classification/static_night_test.csv")

    rows_to_publish, rows_not_publish = load_and_preproc_data()
    if category == "without_night":
        print("without_night")
        print(f"before {len(rows_not_publish)}")
        rows_not_publish = records_withing_opening_hours(rows_not_publish)
        print(f"after {len(rows_not_publish)}")

    rows_not_publish = rows_not_publish.sample(100_000, random_state=seed_value)

    combined = concat_publish_not_publish(rows_not_publish, rows_to_publish)

    rows_to_keep = remove_train_from_test_data(combined, test)
    assert 0 == len(set(list(rows_to_keep['recordId'])).intersection(set(list(test['recordId']))))

    rows_to_keep = remove_train_from_test_data(rows_to_keep, test_night)

    print(
        f"Saving no_night_100k_as_negative, with size {len(rows_to_keep)}, where post is {(len(rows_to_keep[rows_to_keep['post'] == 1]) / len(rows_to_keep))} percent")
    rows_to_keep.to_csv(f"data/pipeline_runs/classification/{category}/100k_as_negative.csv", index=False)


def no_night_all_except_test():
    print(category)

    test = pd.read_csv(f"data/pipeline_runs/classification/{category}/static_test.csv")
    test_night = pd.read_csv("data/pipeline_runs/classification/static_night_test.csv")

    rows_to_publish, rows_not_publish = load_and_preproc_data()
    if category == "without_night":
        print("without_night")

        rows_not_publish = records_withing_opening_hours(rows_not_publish)

    combined = concat_publish_not_publish(rows_not_publish, rows_to_publish)

    rows_to_keep = remove_train_from_test_data(combined, test)
    assert 0 == len(set(list(rows_to_keep['recordId'])).intersection(set(list(test['recordId']))))

    rows_to_keep = remove_train_from_test_data(rows_to_keep, test_night)

    print(
        f"Saving no_night_all_except_test, with size {len(rows_to_keep)}, where post is {(len(rows_to_keep[rows_to_keep['post'] == 1]) / len(rows_to_keep))} percent")
    rows_to_keep.to_csv(f"data/pipeline_runs/classification/{category}/all_except_test.csv", index=False)


def no_night_after_2020_rest_90_percent():
    print(category)
    test = pd.read_csv(f"data/pipeline_runs/classification/{category}/static_test.csv")
    test_night = pd.read_csv("data/pipeline_runs/classification/static_night_test.csv")

    rows_to_publish, rows_not_publish = load_and_preproc_data()
    if category == "without_night":
        print("without_night")
        rows_not_publish = records_withing_opening_hours(rows_not_publish)

    combined = concat_publish_not_publish(rows_not_publish, rows_to_publish)
    combined = combined[combined['overallStartTime'].dt.year > 2020]

    rows_to_keep = remove_train_from_test_data(combined, test)
    assert 0 == len(set(list(rows_to_keep['recordId'])).intersection(set(list(test['recordId']))))

    rows_to_keep = remove_train_from_test_data(rows_to_keep, test_night)

    print(
        f"Saving no_night_after_2020_rest_90_percent, with size {len(rows_to_keep)}, where post is {(len(rows_to_keep[rows_to_keep['post'] == 1]) / len(rows_to_keep))} percent")
    rows_to_keep.to_csv(f"data/pipeline_runs/classification/{category}/after_2020_rest_90_percent.csv", index=False)


def remove_train_from_test_data(combined, test):
    print(f"Len before removing from combined: {len(combined)}")
    test_recordIds = list(test['recordId'])
    test_situationIds = list(test['situationId'])
    rows_to_keep = combined[~combined['recordId'].isin(test_recordIds)]
    assert (len(combined) - len(rows_to_keep)) == len(combined[combined['recordId'].isin(test_recordIds)])
    assert len(combined) != len(rows_to_keep)

    rows_to_keep = rows_to_keep[~rows_to_keep['situationId'].isin(test_situationIds)]

    assert 0 == len(rows_to_keep[rows_to_keep['situationId'].isin(test_situationIds)])
    assert len(combined) != len(rows_to_keep)

    print(f"Len after removing from combined: {len(rows_to_keep)}")

    return rows_to_keep


def concat_publish_not_publish(rows_not_publish, rows_to_publish):
    return pd.concat([rows_to_publish, rows_not_publish])


def create_static_test():
    # Only take incidents occuring during NRK's opening hours
    # Take the most recent data, incidents occuring after 2020
    # Sample 10% with a natural distribution
    rows_to_publish, rows_not_publish = load_and_preproc_data()

    if category == "without_night":
        rows_not_publish = records_withing_opening_hours(rows_not_publish)

    combined = concat_publish_not_publish(rows_not_publish, rows_to_publish)
    combined = combined[combined['overallStartTime'].dt.year > 2020]

    test = combined.sample(frac=0.1, random_state=seed_value)
    print(f"Percent to post {len(test[test['post'] == 1]) / len(test)} for static test")
    print(f"Size of test {len(test)}")
    test.to_csv(f"data/pipeline_runs/classification/{category}/static_test.csv", index=False)


def records_withing_opening_hours(rows_not_publish):
    rows_not_publish = rows_not_publish[
        ((rows_not_publish['overallStartTime'].dt.hour > 6) & (rows_not_publish['overallStartTime'].dt.hour < 22))]
    return rows_not_publish


if __name__ == '__main__':
    global df
    threshold = 0.9
    seed_value = 42

    if 'df' not in globals():
        df, df_tweet, link_dict, spark = pipeline_starter()
        df, link_df, df_tweet = filter_pyspark_df(df, df_tweet, link_dict, spark)
        pd_df, pd_df_tweet = transform_for_alignment_e5(df, df_tweet)

    category = "without_night"
    create_static_test()
    no_night_after_2020_rest_90_percent()
    no_night_all_except_test()
    no_night_100k_as_negative()

    category = "with_night"
    create_static_test()
    no_night_after_2020_rest_90_percent()
    no_night_all_except_test()
    no_night_100k_as_negative()
