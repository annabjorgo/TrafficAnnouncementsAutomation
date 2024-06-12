import datetime

import pandas as pd
import pyspark.sql.functions as sf
import torch
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi

from pipelines.alignment_pipeline_utils import pipeline_starter, filter_pyspark_df, transfer_to_pandas_test, \
    align_data, \
    max_sim_sentence_transformer, max_sim_bm25, remove_stopwords_punctuation, split_sentences, embed_pandas, \
    max_sim_jaccard, join_sentences, max_sim_sentence_transformer_precomputed
from pipelines.alignment_result_utils import measure_accuracy, check_incorrect, print_incorrect
#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df, df_tweet, link_dict, spark = pipeline_starter()
df.persist()

# %%

df, link_df, df_tweet = filter_pyspark_df(df, df_tweet, link_dict, spark)
pd_df, pd_link = transfer_to_pandas_test(df, link_df)
# %%
model_name = 'intfloat/multilingual-e5-large'
model = SentenceTransformer(model_name, device=device)
pre_pro_pd_lin = pd_link.copy()
pre_pro_pd_lin = embed_pandas(pre_pro_pd_lin, model)
aligned = align_data(pre_pro_pd_lin, pd_df, timedelta=6, model=model,
                     sim_func=max_sim_sentence_transformer,)
print(f"\nAccuracy for {model_name}")
# correct, incorrect = measure_accuracy(aligned, pre_pro_pd_lin, df)
# joined_incorrect = check_incorrect(incorrect, pre_pro_pd_lin, pd_df)

record, situation, location = measure_accuracy(aligned, pre_pro_pd_lin, df)
# print_incorrect(joined_incorrect)

#%%
model_name = "KennethEnevoldsen/dfm-sentence-encoder-large-exp1"
model = SentenceTransformer(model_name, device=device)
pre_pro_pd_lin = pd_link.copy()
pre_pro_pd_lin = embed_pandas(pre_pro_pd_lin, model)
aligned = align_data(pre_pro_pd_lin, pd_df, timedelta=6, model=model,
                     sim_func=max_sim_sentence_transformer)
print(f"\nAccuracy for {model_name}")
correct, incorrect = measure_accuracy(aligned, pre_pro_pd_lin, df)
joined_incorrect = check_incorrect(incorrect, pre_pro_pd_lin, pd_df)
#%%
model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
model = SentenceTransformer(model_name, device=device)
pre_pro_pd_lin = pd_link.copy()
pre_pro_pd_lin = embed_pandas(pre_pro_pd_lin, model)
aligned = align_data(pre_pro_pd_lin, pd_df, timedelta=6, model=model,
                     sim_func=max_sim_sentence_transformer)
print(f"\nAccuracy for {model_name}")
correct, incorrect = measure_accuracy(aligned, pre_pro_pd_lin, df)
joined_incorrect = check_incorrect(incorrect, pre_pro_pd_lin, pd_df)
# print_incorrect(joined_incorrect)

#%%
model_name = 'sentence-transformers/distiluse-base-multilingual-cased-v2'
model = SentenceTransformer(model_name, device=device)
pre_pro_pd_lin = pd_link.copy()
pre_pro_pd_lin = embed_pandas(pre_pro_pd_lin, model)
aligned = align_data(pre_pro_pd_lin, pd_df, timedelta=6, model=model,
                     sim_func=max_sim_sentence_transformer)
print(f"\nAccuracy for {model_name}")
correct, incorrect = measure_accuracy(aligned, pre_pro_pd_lin, df)
joined_incorrect = check_incorrect(incorrect, pre_pro_pd_lin, pd_df)
# print_incorrect(joined_incorrect)

#%%
model_name = 'sentence-transformers/LaBSE'
model = SentenceTransformer(model_name, device=device)
pre_pro_pd_lin = pd_link.copy()
pre_pro_pd_lin = embed_pandas(pre_pro_pd_lin, model)
aligned = align_data(pre_pro_pd_lin, pd_df, timedelta=6, model=model,
                     sim_func=max_sim_sentence_transformer)
print(f"\nAccuracy for {model_name}")
correct, incorrect = measure_accuracy(aligned, pre_pro_pd_lin, df)
joined_incorrect = check_incorrect(incorrect, pre_pro_pd_lin, pd_df)
# print_incorrect(joined_incorrect)

# %%
model_name = 'BM25Okapi unprocessed'
model = BM25Okapi
pre_pro_pd_df = pd_df.copy()
pre_pro_pd_df['concat_text'] = split_sentences(pre_pro_pd_df['concat_text'])
aligned = align_data(pd_link, pre_pro_pd_df, timedelta=6, model=model, sim_func=max_sim_bm25)
pre_pro_pd_df['concat_text'] = join_sentences(pre_pro_pd_df['concat_text'])
print(f"\nAccuracy for {model_name}")
correct, incorrect = measure_accuracy(aligned, pd_link, df)
joined_incorrect = check_incorrect(incorrect, pd_link, pre_pro_pd_df)
# print_incorrect(joined_incorrect)

#%%
model_name = 'BM25Okapi w/o stopwords + punctuation'
model = BM25Okapi
pre_pro_pd_df = pd_df.copy()
pre_pro_pd_df['concat_text'] = remove_stopwords_punctuation(pre_pro_pd_df['concat_text'])
aligned = align_data(pd_link, pre_pro_pd_df, timedelta=6, model=model, sim_func=max_sim_bm25)
pre_pro_pd_df['concat_text'] = join_sentences(pre_pro_pd_df['concat_text'])
print(f"\nAccuracy for {model_name}")
correct, incorrect = measure_accuracy(aligned, pd_link, df)
joined_incorrect = check_incorrect(incorrect, pd_link, pre_pro_pd_df)
# print_incorrect(joined_incorrect)
#%%
model_name = "Jaccard Distance"
model = None #We are only using the similarity function on this
pre_pro_pd_df = pd_df.copy()
pre_pro_pd_df['concat_text'] = split_sentences(pre_pro_pd_df['concat_text'])
aligned = align_data(pd_link, pre_pro_pd_df, timedelta=6, model=model, sim_func=max_sim_jaccard)
pre_pro_pd_df['concat_text'] = join_sentences(pre_pro_pd_df['concat_text'])
print(f"\nAccuracy for {model_name}")
correct, incorrect = measure_accuracy(aligned, pd_link, df)
joined_incorrect = check_incorrect(incorrect, pd_link, pre_pro_pd_df)
print_incorrect(joined_incorrect)
#%%
