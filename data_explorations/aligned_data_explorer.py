import pandas as pd
import pyspark.sql.functions as sf
import matplotlib.pyplot as plt
import tikzplotlib
from pipelines.alignment_pipeline_utils import pipeline_starter, filter_pyspark_df, transform_for_alignment_e5

#%%
df, df_tweet, link_dict, spark = pipeline_starter()
df, link_df, df_tweet = filter_pyspark_df(df, df_tweet, link_dict, spark)
pd_df, pd_df_tweet = transform_for_alignment_e5(df, df_tweet)
#%%
aligned_df = pd.read_csv("data/pipeline_runs/alignment/alignment d:10 m:4 h:17.csv")
annotated_df = pd.read_csv("data/pipeline_runs/alignment/annotated_alignment d:10 m:4 h:17.csv")
no_df = pd.read_csv("data/pipeline_runs/alignment/no_alignment d:10 m:4 h:17.csv")

#%%
def merge_annotated():
    r = annotated_df[(annotated_df['correct'] == "1") | (annotated_df['correct'] == "0")]
    t = pd.merge(aligned_df, r[['nrk_id', 'correct']], on="nrk_id", how="inner")
    t.to_csv("data/pipeline_runs/alignment/annotated_alignment d:10 m:4 h:17.csv", index=False)
#%%

def process_no_alignment():
    no_df = pd.read_csv("data/pipeline_runs/no_alignment d:10 m:4 h:17.csv")
    no_df = no_df[['nrk_id', 'nrk_text']]
    no_df.to_csv("data/pipeline_runs/alignment/no_alignment d:10 m:4 h:17.csv", index=False)
process_no_alignment()

#%%
def check_time_delta(no_df):
    join_df = pd_df_tweet[['nrk_id', 'nrk_created_at']].astype("string")
    no_df['nrk_id'] = no_df['nrk_id'].astype("string")
    no_df = pd.merge(no_df, join_df, on="nrk_id", how="inner")
    no_df['nrk_ts'] = pd.to_datetime(no_df['nrk_created_at'])
    it = no_df.iloc[0]
    time_window = pd.Timedelta(hours=24)
    search_df = pd_df[abs(pd_df['overallStartTime'] - it.nrk_ts) <= time_window].copy()
    return it, search_df
q, r = check_time_delta(no_df)

#%%
def save_200_aligned_above_90(aligned_df):
    (aligned_df[aligned_df['similarity'] > 0.9].sample(n=200)).to_csv("data/pipeline_runs/alignment/above_.9_alignment d:10 m:4 h:17.csv", index=False)
save_200_aligned_above_90(aligned_df)
#%%
annotated_df['svv_ts'] = pd.to_datetime(annotated_df['svv_ts'])
annotated_df['nrk_ts'] = pd.to_datetime(annotated_df['nrk_ts'])

time_sum = 0
for it in annotated_df:
    print(it['svv_ts'], it['nrk_ts'])
    print(it['svv_ts'] - it['nrk_ts'])

print(time_sum)

#%%
annotated_df['Average_time'] = (annotated_df[['svv_ts', 'nrk_ts']]
                                .aannstype(str)
                                .agg(pd.to_timedelta)
                                .eval("Time_1+Time_2").div(2))

#%%
#TODO:
# -Acc på annotert data
# -Gjennomsnitt similarity av annoterte rette
# -Gjennomsnitt similarity av annoterte feil
# -Gjennomsnitt av similarity på alle
# -Distribusjon av similarity annotert rett
# -Distribusjon av similarity annotert feil
# -Distribusjon av similarity på alle
# -Lage filter på de som ikke kommer fra en svv hendelse men kanskje kommer fra en twitter konto og se hvor mye det reduserer no alignments

#%%
len(annotated_df[annotated_df['correct'] == 1]) / len(annotated_df)
#%%
annotated_df[annotated_df['correct'] == 1]['similarity'].mean()
#%%
annotated_df[annotated_df['correct'] == 0]['similarity'].mean()
#%%
aligned_df['similarity'].mean()
#%%
annotated_df[annotated_df['correct'] == 1]['similarity'].hist()
# plt.title('Distribution of correctly aligned incidents')
tikzplotlib.save(f"correctly aligned incidents histogram.tex")
plt.clf()
plt.cla()
plt.close()
# plt.show()

#%%
annotated_df[annotated_df['correct'] == 0]['similarity'].hist()
# plt.title('Distribution of incorrectly aligned incidents')
tikzplotlib.save(f"incorrectly aligned incidents histogram.tex")
plt.clf()
plt.cla()
plt.close()
# plt.show()
#%%
aligned_df['similarity'].hist()
# plt.title('Distribution of all aligned incidents')
tikzplotlib.save(f"aligned incidents histogram.tex")
plt.clf()
plt.cla()
plt.close()
# plt.show()