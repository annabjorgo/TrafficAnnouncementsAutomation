import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, TrainingArguments, \
    DataCollatorWithPadding
from transformers.integrations import WandbCallback
import tikzplotlib

from pipelines.classification_pipeline import pre_process_logits

# %%
id = '8s6rb80d'
path = f"model_run/{id}"
category = "with_night/"
test_file = f"data/pipeline_runs/classification/{category}static_test.csv"
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")


def preprocess_text(df):
    # TODO: should i use [CLS] and such here
    # tokenized_text = tokenizer(list(map(lambda x: "[CLS]" + x + "[SEP]", df['concat_text'])))
    tokenized_text = tokenizer(df['concat_text'])
    tokenized_text["labels"] = df['post']
    return tokenized_text


label2id = {'post': 1, 'discard': 0}
id2label = {1: 'post', 0: 'discard'}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(path)
test_data = load_dataset("csv", data_files=test_file) \
    .map(preprocess_text, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    path, num_labels=2, id2label=id2label, label2id=label2id, trust_remote_code=True,
).to(device)

from torch.utils.data import DataLoader

data_collator = DataCollatorWithPadding(tokenizer)
test_dataloader = DataLoader(test_data['train'], batch_size=8, collate_fn=data_collator)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print("\n" + classification_report(y_pred=predictions, y_true=labels, labels=[0, 1]))
    pr = precision.compute(predictions=predictions, references=labels)
    a = accuracy.compute(predictions=predictions, references=labels)
    r = recall.compute(predictions=predictions, references=labels)
    f = f1.compute(predictions=predictions, references=labels)
    return {**a, **pr, **r, **f}
def get_training_args():
    return TrainingArguments(
        output_dir="model_run",
        logging_dir="model_run_logs",
        per_device_train_batch_size=32,
        dataloader_num_workers=4,
        do_train=False,
        num_train_epochs=10,
        weight_decay=0.01,
        save_steps=0.15,
        eval_steps=0.15,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=4,

        report_to="wandb"
    )


trainer = Trainer(
    model=model,
    # model_init=model_init,
    args=get_training_args(),
    eval_dataset=test_data["train"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=pre_process_logits,
    callbacks=[WandbCallback],
)
#%%
print(f"Predicting on {test_file} with run {id}")

out = trainer.predict(test_data['train'])
test_df = pd.DataFrame(test_data['train'])
test_df['prediction'] = out.predictions
print(out.metrics)
sampled_df = test_df.sample(n=200, random_state=42)
# sampled_df.to_csv(f"data/pipeline_runs/classification/samples/{id}_sample.csv")

#%%
test_df['hour'] = pd.to_datetime(test_df['overallStartTime']).dt.hour
test_df['hour'].hist(bins=24, figsize=(10, 6), color='skyblue', edgecolor='black')
plt.xlabel('Hour of the Day')
plt.ylabel('Count')
plt.title('Histogram of test data by Hour')
plt.xticks(range(24))
plt.grid(False)
# plt.show()
tikzplotlib.save(f"test_data_hour_histogram.tex")
plt.clf()
plt.cla()
plt.close()

test_df[test_df['post'] == 1]['hour'].hist(bins=24, figsize=(10, 6), color='skyblue', edgecolor='black')
plt.xlabel('Hour of the Day')
plt.ylabel('Count')
plt.title('Histogram of test data to post by Hour')
plt.xticks(range(24))
plt.grid(False)
# plt.show()
tikzplotlib.save(f"test_data_post_hour_histogram.tex")
plt.clf()
plt.cla()
plt.close()


test_df[test_df['prediction'] == 1]['hour'].hist(bins=24, figsize=(10, 6), color='skyblue', edgecolor='black')
plt.xlabel('Hour of the Day')
plt.ylabel('Count')
plt.title('Histogram of test data prediction by Hour')
plt.xticks(range(24))
plt.grid(False)
# plt.show()
tikzplotlib.save(f"test_data_prediction_hour_histogram.tex")
plt.clf()
plt.cla()
plt.close()

#%%
test_df['w_day'] = pd.to_datetime(test_df['overallStartTime']).dt.dayofweek
test_df['w_day'].hist(bins=7, figsize=(10, 6), color='skyblue', edgecolor='black')
plt.xlabel('Day of week')
plt.ylabel('Count')
plt.title('Histogram of test data by day of week')
plt.xticks(range(0,7))
plt.grid(False)
# plt.show()
tikzplotlib.save(f"test_data_w_day_histogram.tex")

plt.clf()
plt.cla()
plt.close()


ax =test_df[test_df['post'] == 1]['w_day'].plot.hist(bins=7, figsize=(10, 6), color='skyblue', edgecolor='black')
plt.xlabel('Day of week')
plt.ylabel('Count')
plt.title('Histogram of label test data by day of week')
plt.xticks(range(0,7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
plt.grid(False)
# plt.show()
tikzplotlib.save(f"test_data_label_w_day_histogram.tex")

plt.clf()
plt.cla()
plt.close()

ax =test_df[test_df['prediction'] == 1]['w_day'].plot.hist(bins=7, figsize=(10, 6), color='skyblue', edgecolor='black')
plt.xlabel('Day of week')
plt.ylabel('Count')
plt.title('Histogram of prediction data by day of week')
plt.xticks(range(0,7))
plt.grid(False)
# plt.show()
tikzplotlib.save(f"test_data_prediction_w_day_histogram.tex")

plt.clf()
plt.cla()
plt.close()
#%%
aa = test_df.sort_values(by=['post', 'concat_text'],ascending=[False, False] )