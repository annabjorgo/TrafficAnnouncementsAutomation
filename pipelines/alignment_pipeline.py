import datetime

import pandas as pd
import pyspark.sql.functions as sf
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

from pipelines.alignment_pipeline_utils import pipeline_starter, filter_pyspark_df, transfer_to_pandas, \
    align_data, \
    max_sim_sentence_transformer, max_sim_bm25, remove_stopwords_punctuation, split_sentences
from pipelines.alignment_result_utils import measure_accuracy, check_incorrect, print_incorrect
#%%
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
# %%
aligned = align_data(pd_link, pd_df, timedelta=6, model=model,
                     sim_func=max_sim_sentence_transformer)
correct, incorrect = measure_accuracy(aligned, pd_link, df)
joined_incorrect = check_incorrect(incorrect, pd_link, pd_df)
print_incorrect(joined_incorrect)

# %%
model = BM25Okapi
pre_pro_pd_df = pd_df.copy()
pre_pro_pd_df['concat_text'] = split_sentences(pre_pro_pd_df['concat_text'])
aligned = align_data(pd_link, pre_pro_pd_df, timedelta=6, model=model, sim_func=max_sim_bm25)
correct, incorrect = measure_accuracy(aligned, pd_link, df)
joined_incorrect = check_incorrect(incorrect, pd_link, pd_df)
# print_incorrect(joined_incorrect)

#%%
model = BM25Okapi
pre_pro_pd_df = pd_df.copy()
pre_pro_pd_df['concat_text'] = remove_stopwords_punctuation(pre_pro_pd_df['concat_text'])
aligned = align_data(pd_link, pre_pro_pd_df, timedelta=6, model=model, sim_func=max_sim_bm25)
correct, incorrect = measure_accuracy(aligned, pd_link, df)
joined_incorrect = check_incorrect(incorrect, pd_link, pd_df)
# print_incorrect(joined_incorrect)
#%%
