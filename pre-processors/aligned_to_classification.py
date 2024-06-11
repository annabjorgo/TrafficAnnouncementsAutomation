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


def create_split(rows_to_publish, rows_not_publish):
    combined_df = pd.concat([rows_not_publish, rows_to_publish], axis=0)

    train, validate, test = np.split(combined_df.sample(frac=1, random_state=seed_value),
                                     [int(train_size * len(combined_df)), int(test_and_val_size * len(combined_df))])
    return train, test, validate, combined_df


def print_stat(train, test, validate, combined_df):
    print(f"Negative size is {negative_size}")
    print(f"Percent to post {len(train[train['post'] == 1]) / len(train)} for train",
          f"Percent of combined {len(train) / len(combined_df)} for train")
    print(f"Percent to post {len(validate[validate['post'] == 1]) / len(validate)} for validate",
          f"Percent of combined {len(validate) / len(combined_df)} for validate")
    print(f"Percent to post {len(test[test['post'] == 1]) / len(test)} for test",
          f"Percent of combined {len(test) / len(combined_df)} for test")


def save_to_csv(train, test, validate, comment):
    if comment is not None:
        folder_name = f'data/pipeline_runs/classification/{comment}, threshold: {threshold}, negative_size:{negative_size} - d:{datetime.datetime.now().day} m:{datetime.datetime.now().month} h:{datetime.datetime.now().hour}/'

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    train.to_csv(f"{folder_name}train.csv", index=False)
    test.to_csv(f"{folder_name}test.csv", index=False)
    validate.to_csv(f"{folder_name}validate.csv", index=False)
    print(f"Saved the filed to {folder_name}")
    return folder_name


def check_overlap(rows_to_publish, rows_not_publish):
    publish_intersection_recordId = set(rows_to_publish['recordId']).intersection(set(rows_not_publish['recordId']))
    publish_intersection_situationId = set(rows_to_publish['situationId']).intersection(
        set(rows_not_publish['situationId']))

    assert len(publish_intersection_recordId) == 0
    assert len(publish_intersection_situationId) == 0


def create_small_file():
    test_file = "data/pipeline_runs/classification/threshold: 0.9, negative_size:794873 - d:24 m:4 h:8/test.csv"
    small_file = "data/pipeline_runs/classification/threshold: 0.9, negative_size:794873 - d:24 m:4 h:8/small.csv"
    pd.read_csv(test_file).sample(100).to_csv(small_file)


def oversample_all():
    rows_to_publish, rows_not_publish = load_and_preproc_data()
    rows_not_publish = rows_to_publish.sample(n=negative_size, random_state=seed_value, replace=True).copy()
    train, test, validate, combined_df = create_split(rows_to_publish=rows_to_publish,
                                                      rows_not_publish=rows_not_publish)
    print_stat(train, test, validate, combined_df)
    save_to_csv(train, test, validate, "oversampled_all")


def oversample_train(train):
    rows_to_publish, rows_not_publish = load_and_preproc_data()
    train, test, validate, combined_df = create_split(rows_to_publish=rows_to_publish,
                                                      rows_not_publish=rows_not_publish)
    train = train.append(
        train[train['post'] == 1].sample(len(train[train['post'] == 0]) - len(train[train['post'] == 1]),
                                         random_state=seed_value, replace=True), ignore_index=True)
    print_stat(train, test, validate, combined_df)
    save_to_csv(train, test, validate, "oversampled_train")


def remove_night_and_save_as_extra_validate():
    rows_to_publish, rows_not_publish = load_and_preproc_data()
    to_validate = rows_not_publish[
        ((rows_not_publish['overallStartTime'].dt.hour < 8) | (rows_not_publish['overallStartTime'].dt.hour > 21))]
    rows_not_publish = rows_not_publish[
        ~((rows_not_publish['overallStartTime'].dt.hour < 8) | (rows_not_publish['overallStartTime'].dt.hour > 21))]
    train, test, validate, combined_df = create_split(rows_to_publish=rows_to_publish,
                                                      rows_not_publish=rows_not_publish)

    print(len(to_validate))
    print_stat(train, test, validate, combined_df)
    folder_name = save_to_csv(train, test, validate, "remove_night_and_save_as_extra_validate")
    to_validate.to_csv(f"{folder_name}night_validate.csv", index=False)


def night_and_x_negative_as_train():
    rows_to_publish, rows_not_publish = load_and_preproc_data()

    night = rows_not_publish[
        ((rows_not_publish['overallStartTime'].dt.hour < 8) | (rows_not_publish['overallStartTime'].dt.hour > 21))]
    negative = rows_not_publish.sample(n=negative_size, random_state=seed_value, replace=True).copy()
    print(len(night))
    rows_not_publish = pd.concat([night, negative], axis=0)

    print(rows_not_publish)

    train, test, validate, combined_df = create_split(rows_to_publish=rows_to_publish,
                                                      rows_not_publish=rows_not_publish)

    to_validate = rows_not_publish[
        ((rows_not_publish['overallStartTime'].dt.hour < 8) | (rows_not_publish['overallStartTime'].dt.hour > 21))]

    print_stat(train, test, validate, combined_df)
    folder_name = save_to_csv(train, test, validate, f"night and {negative_size} as train")


def no_night_100k_as_negative():
    test = pd.read_csv("data/pipeline_runs/classification/static_test.csv")
    rows_to_publish, rows_not_publish = load_and_preproc_data()
    rows_not_publish = records_withing_opening_hours(rows_not_publish)
    rows_not_publish = rows_not_publish.sample(100_000, random_state=seed_value)

    combined = concat_publish_not_publish(rows_not_publish, rows_to_publish)
    rows_to_keep = ~combined.isin(test.to_dict(orient='list')).all(axis=1)
    rest = combined[rows_to_keep]
    print(f"Saving no_night_100k_as_negative, with size {len(rest)}, where post is {(len(rest[rest['post'] == 1]) / len(rest))} percent")
    rest.to_csv("data/pipeline_runs/classification/no_night_100k_as_negative.csv", index=False)


def no_night_all_except_test():
    test = pd.read_csv("data/pipeline_runs/classification/static_test.csv")
    rows_to_publish, rows_not_publish = load_and_preproc_data()
    rows_not_publish = records_withing_opening_hours(rows_not_publish)

    combined = concat_publish_not_publish(rows_not_publish, rows_to_publish)
    rows_to_keep = ~combined.isin(test.to_dict(orient='list')).all(axis=1)
    rest = combined[rows_to_keep]
    print(f"Saving no_night_all_except_test, with size {len(rest)}, where post is {(len(rest[rest['post'] == 1]) / len(rest))} percent")
    rest.to_csv("data/pipeline_runs/classification/no_night_all_except_test.csv", index=False)


def no_night_after_2020_rest_90_percent():
    test = pd.read_csv("data/pipeline_runs/classification/static_test.csv")
    rows_to_publish, rows_not_publish = load_and_preproc_data()
    rows_not_publish = records_withing_opening_hours(rows_not_publish)

    combined = concat_publish_not_publish(rows_not_publish, rows_to_publish)
    combined = combined[combined['overallStartTime'].dt.year > 2020]

    rows_to_keep = ~combined.isin(test.to_dict(orient='list')).all(axis=1)
    rest = combined[rows_to_keep]
    print(f"Saving no_night_after_2020_rest_90_percent, with size {len(rest)}, where post is {(len(rest[rest['post'] == 1]) / len(rest))} percent")
    rest.to_csv("data/pipeline_runs/classification/no_night_after_2020_rest_90_percent.csv", index=False)


def concat_publish_not_publish(rows_not_publish, rows_to_publish):
    return pd.concat([rows_to_publish, rows_not_publish])


def create_static_test():
    # Only take incidents occuring during NRK's opening hours
    # Take the most recent data, incidents occuring after 2020
    # Sample 10% with a natural distribution
    rows_to_publish, rows_not_publish = load_and_preproc_data()

    rows_not_publish = records_withing_opening_hours(rows_not_publish)
    combined = concat_publish_not_publish(rows_not_publish, rows_to_publish)
    combined = combined[combined['overallStartTime'].dt.year > 2020]

    test = combined.sample(frac=0.1, random_state=seed_value)
    print(f"Percent to post {len(test[test['post'] == 1]) / len(test)} for static test")
    print(f"Size of test {len(test)}")
    test.to_csv("data/pipeline_runs/classification/static_test.csv", index=False)


def records_withing_opening_hours(rows_not_publish):
    rows_not_publish = rows_not_publish[
        ((rows_not_publish['overallStartTime'].dt.hour > 6) | (rows_not_publish['overallStartTime'].dt.hour < 22))]
    return rows_not_publish


if __name__ == '__main__':
    global df
    threshold = 0.9
    seed_value = 42
    negative_size = 740_000
    train_size = 0.6
    test_and_val_size = 0.8  # 20% for each
    folder_name = f'data/pipeline_runs/classification/threshold: {threshold}, negative_size:{negative_size} - d:{datetime.datetime.now().day} m:{datetime.datetime.now().month} h:{datetime.datetime.now().hour}/'

    if 'df' not in globals():
        df, df_tweet, link_dict, spark = pipeline_starter()
        df, link_df, df_tweet = filter_pyspark_df(df, df_tweet, link_dict, spark)
        pd_df, pd_df_tweet = transform_for_alignment_e5(df, df_tweet)

    # oversample_train()
    # oversample_all()
    # remove_night_and_save_as_extra_validate()
    # night_and_x_negative_as_train()

    # create_static_test()
    # no_night_after_2020_rest_90_percent()
    # no_night_all_except_test()
    # no_night_100k_as_negative()