#TODO: How to classify:
# - Set threshold on aligned data.
# - Extract the SVV incident from those
# - Find SVV data that has not been aligned to anything
# - Sample a number of those
# - Combine aligned data with SVV incidents that has not been aligned

# %%
import datetime
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# %%
from pipelines.alignment_pipeline_utils import pipeline_starter, filter_pyspark_df, transform_for_alignment_e5

df, df_tweet, link_dict, spark = pipeline_starter()
df, link_df, df_tweet = filter_pyspark_df(df, df_tweet, link_dict, spark)
pd_df, pd_df_tweet = transform_for_alignment_e5(df, df_tweet)
#%%
aligned_df = pd.read_csv("data/pipeline_runs/alignment/alignment d:23 m:4 h:16.csv")
annotated_df = pd.read_csv("data/pipeline_runs/alignment/annotated_alignment d:10 m:4 h:17.csv")
no_df = pd.read_csv("data/pipeline_runs/alignment/no_alignment d:23 m:4 h:16.csv")
#%%
threshold = 0.9
rows_to_publish = aligned_df[aligned_df['similarity'] > threshold][['recordId', 'situationId','svv_text', 'svv_ts']].copy()
rows_to_publish = rows_to_publish.rename(columns={"svv_text": 'concat_text', "svv_ts": 'overallStartTime'})
rows_to_publish['post'] = 1
#%%
published_recordIds = list(set(list(aligned_df['recordId'])))
published_situationIds = list(set(list(aligned_df['situationId'])))
seed_value = 42
negative_size = 300_00

rows_not_publish = pd_df[~pd_df['recordId'].isin(published_recordIds) & ~pd_df['situationId'].isin(published_situationIds)].sample(n=negative_size, random_state=seed_value).copy()
rows_not_publish['post'] = 0
#%%
train_size = 0.6
test_and_val_size = 0.2 # 20% for each
combined_df = pd.concat([rows_not_publish, rows_to_publish], axis=0)

train, validate, test = np.split(combined_df.sample(frac=1, random_state=seed_value), [int(train_size*len(combined_df)), int(test_and_val_size * len(combined_df))])

#%%
folder_name = f'data/pipeline_runs/classification/threshold: {threshold}, negative_size:{negative_size} - d:{datetime.datetime.now().day} m:{datetime.datetime.now().month} h:{datetime.datetime.now().hour}/'
if not os.path.exists(folder_name):
    os.mkdir(folder_name)
train.to_csv(f"{folder_name}train.csv", index=False)
test.to_csv(f"{folder_name}test.csv", index=False)
validate.to_csv(f"{folder_name}validate.csv", index=False)

#%%
publish_intersection_recordId = set(rows_to_publish['recordId']).intersection(set(rows_not_publish['recordId']))
publish_intersection_situationId = set(rows_to_publish['situationId']).intersection(set(rows_not_publish['situationId']))

assert len(publish_intersection_recordId) == 0

assert len(publish_intersection_situationId) == 0
#%%
test_file = "data/pipeline_runs/classification/threshold: 0.9, negative_size:300000 - d:18 m:4 h:12/test.csv"

small_file = "data/pipeline_runs/classification/threshold: 0.9, negative_size:300000 - d:18 m:4 h:12/small.csv"
pd.read_csv(test_file).sample(100).to_csv(small_file)