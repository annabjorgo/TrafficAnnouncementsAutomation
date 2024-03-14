import datetime

import pandas as pd
import pyspark.sql.functions as sf
import torch
from sentence_transformers import SentenceTransformer, util

from pipelines.pipeline_utils import pipeline_starter, filter_pyspark_df, transfer_to_pandas, align_data_sentence_transformer, \
    measure_accuracy, check_incorrect, print_incorrect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df, df_tweet, link_dict, spark = pipeline_starter()
models_to_test = [
    'BAAI/bge-large-en-v1.5',
    # 'Cohere/Cohere-embed-multilingual-v3.0', doesn't exist on sentene transformers
    #     "Salesforce/SFR-Embedding-Mistral", too big

]

model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
df.persist()

# %%

df, link_df = filter_pyspark_df(df, df_tweet, link_dict, spark)
pd_df, pd_link = transfer_to_pandas(df, link_df, model)

aligned = align_data_sentence_transformer(pd_link, pd_df, timedelta=6, model=model)
correct, incorrect = measure_accuracy(aligned)
incorrect_ = check_incorrect(incorrect, pd_link, pd_df)
print_incorrect(incorrect_)
