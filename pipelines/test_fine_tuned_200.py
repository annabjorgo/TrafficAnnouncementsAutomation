import os
from typing import Any
from typing import Dict, Union
import pandas as pd
import evaluate
import numpy as np
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

import torch
import wandb
from datasets import load_dataset
from optuna import Trial
from setfit import SetFitModel
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, TrainingArguments, \
    DataCollatorWithPadding, IntervalStrategy, ProgressCallback, DefaultDataCollator
from transformers.integrations import WandbCallback
from transformers.trainer_utils import has_length
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
from pipelines.classification_pipeline import compute_metrics, pre_process_logits

# %%
test_metrics = []
ids = [
    "k2ymrsj0", "kp3m12k1", "klkdgorn", "8gyoguvd", "0nqtlq0t",
    "6tlbtu7s", "ij2u8q5w", "8s6rb80d", "bul9bc47", "0hq4oytu",
    "xqxizl13", "x1qbssf4", "x2veyc5y", "bat40xiz", "yxmt4a7l",
    "3p2vh2dx", "5q0x2dah", "c617wpft", "ky0ikzwo", "4b2assty",
    "1bgxsaku", "4630nz1x", "xfz3t4gy", "r327do7k", "544p19ud",
    "sysughko", "mkbde6yx", "tj32loca", "yncso1s4", "bd5vbtho",
    "c0097eem", "e3ujqx59", "2ieobhwu", "ccwm0ggq", "ynqw9g92",
    "nwtstmn5", "omf8temh", "fyo12e28", "alykryfa", "0d5neo8n",
    "3mbbvt3w", "iwj2tztj", "0e3o9lao", "z04qyses", "b84misum",
    "4up66tym", "gujwea0l", "s9ovqaxy", "9iczx938", "xp43tgy2",
    "055gv9co", "3nsgake1"
]
# id = 'kp3m12k1'

for id in ids:
    try:
        path = f"model_run/{id}"
        category = "with_night/"
        test_file = "data/pipeline_runs/classification/annotated_200_only_night.csv"
        # test_file = f"data/pipeline_runs/classification/{category}static_test.csv"
        accuracy = evaluate.load("accuracy")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")



        def preprocess_text(df):
            # TODO: should i use [CLS] and such here
            # tokenized_text = tokenizer(list(map(lambda x: "[CLS]" + x + "[SEP]", df['concat_text'])))
            tokenized_text = tokenizer(df['concat_text'])
            tokenized_text["labels"] = df['Dag']

            # tokenized_text["labels"] = df['post']
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
        # %%
        from torch.utils.data import DataLoader

        data_collator = DataCollatorWithPadding(tokenizer)
        test_dataloader = DataLoader(test_data['train'], batch_size=8, collate_fn=data_collator)


        # %%
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            print(f"Current model: {id}")
            print("\n" + classification_report(y_pred=predictions, y_true=labels, labels=[0, 1]))
            pr = precision.compute(predictions=predictions, references=labels)
            a = accuracy.compute(predictions=predictions, references=labels)
            r = recall.compute(predictions=predictions, references=labels)
            f = f1.compute(predictions=predictions, references=labels)
            test_metrics.append({"id":id,**a, **pr, **r, **f})
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
        # %%
        print(f"Predicting on {test_file} with run {id}")

        out = trainer.predict(test_data['train'])
        test_df = pd.DataFrame(test_data['train'])
        test_df['prediction'] = out.predictions
        print(out.metrics)
        sampled_df = test_df.sample(n=200, random_state=42)
        sampled_df.to_csv(f"data/pipeline_runs/classification/samples/{id}_sample.csv")
    except Exception as ex:
        print(ex)
        print("Did not work for id", id)
        test_metrics.append({"id":id})

# %%
day_df = pd.DataFrame(test_metrics)
day_df = day_df.rename(columns={"accuracy": 'day_accuracy', "precision": 'day_precision', "recall": 'day_recall', "f1": 'day_f1'})

# %%


for id in ids:
    try:
        path = f"model_run/{id}"
        category = "with_night/"
        test_file = "data/pipeline_runs/classification/annotated_200_only_night.csv"
        # test_file = f"data/pipeline_runs/classification/{category}static_test.csv"
        accuracy = evaluate.load("accuracy")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")



        def preprocess_text(df):
            # TODO: should i use [CLS] and such here
            # tokenized_text = tokenizer(list(map(lambda x: "[CLS]" + x + "[SEP]", df['concat_text'])))
            tokenized_text = tokenizer(df['concat_text'])
            tokenized_text["labels"] = df['Natt']

            # tokenized_text["labels"] = df['post']
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
        # %%
        from torch.utils.data import DataLoader

        data_collator = DataCollatorWithPadding(tokenizer)
        test_dataloader = DataLoader(test_data['train'], batch_size=8, collate_fn=data_collator)


        # %%
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            print(f"Current model: {id}")
            print("\n" + classification_report(y_pred=predictions, y_true=labels, labels=[0, 1]))
            pr = precision.compute(predictions=predictions, references=labels)
            a = accuracy.compute(predictions=predictions, references=labels)
            r = recall.compute(predictions=predictions, references=labels)
            f = f1.compute(predictions=predictions, references=labels)
            test_metrics.append({"id":id,**a, **pr, **r, **f})
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
        # %%
        print(f"Predicting on {test_file} with run {id}")

        out = trainer.predict(test_data['train'])
        test_df = pd.DataFrame(test_data['train'])
        test_df['prediction'] = out.predictions
        print(out.metrics)
        sampled_df = test_df.sample(n=200, random_state=42)
        sampled_df.to_csv(f"data/pipeline_runs/classification/samples/{id}_sample.csv")
    except Exception as ex:
        print(ex)
        print("Did not work for id", id)
        test_metrics.append({"id":id})

night_df = pd.DataFrame(test_metrics)
night_df = night_df.rename(columns={"accuracy": 'night_accuracy', "precision": 'night_precision', "recall": 'night_recall', "f1": 'night_f1'})

merged_df = pd.merge(day_df, night_df, on="id")
all_df  = pd.read_csv("data/pipeline_runs/classification/all_runs.csv")
all_df['id'] = all_df['ID']
combined_df = pd.merge(merged_df, all_df, on="id")
combined_df.to_csv("data/pipeline_runs/classification/combined_runs.csv")