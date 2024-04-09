import pandas as pd
import pyspark.sql.functions as sf

from pipelines.alignment_pipeline_utils import pipeline_starter, filter_pyspark_df, transform_for_alignment_e5

df, df_tweet, link_dict, spark = pipeline_starter()

aligned_df = pd.read_csv("data/pipeline_runs/annotated_alignment d_21 m_3 h_5.csv", sep=";")
#%%
df, link_df = filter_pyspark_df(df, df_tweet, link_dict, spark)
pd_df, df_tweet = transform_for_alignment_e5(df, df_tweet)
#%%

aligned_df[aligned_df['correct'] == "1"]

for it in list(aligned_df.itertuples())[:10]:
    # df.where((sf.col("recordId") == it.recordId) & (sf.col("situationId") == it.situationId)).show()
    print(pd_df[pd_df['concat_text'] == it.svv_text])

    # df.where((sf.col("concat_text") == it.svv_text)).show()
