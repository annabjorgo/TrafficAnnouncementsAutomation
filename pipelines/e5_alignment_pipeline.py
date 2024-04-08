import sys

import torch
from sentence_transformers import SentenceTransformer
import logging

from pipelines.alignment_pipeline_utils import pipeline_starter, filter_pyspark_df, align_data, \
    max_sim_sentence_transformer, transform_for_alignment_e5, embed_nrk, max_sim_sentence_transformer_precomputed, \
    align_multi
logging.basicConfig(filename="logfile.txt")
stderrLogger=logging.StreamHandler()
stderrLogger.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
logging.getLogger().addHandler(stderrLogger)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df, df_tweet, link_dict, spark = pipeline_starter()
# df.persist()
#%%

df, link_df = filter_pyspark_df(df, df_tweet, link_dict, spark)
pd_df, df_tweet = transform_for_alignment_e5(df, df_tweet)
 #%%
model_name = 'intfloat/multilingual-e5-large'
model = SentenceTransformer(model_name, device=device)
pd_df['svv_embed'] = model.encode(pd_df['concat_text'].tolist(), show_progress_bar=True).tolist()
df_tweet = embed_nrk(df_tweet, model)
del model
aligned = align_data(df_tweet, pd_df, timedelta=6, model=None, sim_func=max_sim_sentence_transformer_precomputed)
 # %%
#%%
# aligned = align_multi(df_tweet, pd_df, timedelta=6)
