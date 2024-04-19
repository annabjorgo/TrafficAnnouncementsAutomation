import os
from typing import Any
from typing import Dict, Union

import evaluate
import numpy as np
import torch
import wandb
from datasets import load_dataset
from optuna import Trial
from setfit import SetFitModel
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, TrainingArguments, \
    DataCollatorWithPadding

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
training_file = "data/pipeline_runs/classification/threshold: 0.9, negative_size:300000 - d:18 m:4 h:12/train.csv"
test_file = "data/pipeline_runs/classification/threshold: 0.9, negative_size:300000 - d:18 m:4 h:12/test.csv"
small_file = "data/pipeline_runs/classification/threshold: 0.9, negative_size:300000 - d:18 m:4 h:12/small.csv"
os.environ['WANDB_PROJECT'] = "master-classification"
# %%
label2id = {'post': 1, 'discard': 0}
id2label = {1: 'post', 0: 'discard'}


def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 9e-8, 1e-3, log=True),
    }


def preprocess_text(df):
    # TODO: should i use [CLS] and such here
    tokenized_text = tokenizer(df['concat_text'])
    labels = df['post']
    labels_matrix = np.zeros((len(tokenized_text.encodings), len(id2label)))
    for pos, obj in enumerate(labels):
        labels_matrix[pos, obj] = 1

    tokenized_text["label"] = labels_matrix.tolist()
    return tokenized_text


def prepare_dataset():
    train_data = load_dataset("csv", data_files=training_file) \
        .map(preprocess_text, batched=True)
    test_data = load_dataset("csv", data_files=test_file) \
        .map(preprocess_text, batched=True)

    return train_data, test_data


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    labels = np.argmax(labels, axis=1)
    pr = precision.compute(predictions=predictions, references=labels)
    a = accuracy.compute(predictions=predictions, references=labels)
    return {"accuracy": a, "precision": pr}


def get_training_args(pre_model):
    return TrainingArguments(
        output_dir="model_run",
        logging_dir="model_run_logs",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        dataloader_num_workers=8,
        do_train=True,
        num_train_epochs=20,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=4,
        run_name=pre_model,
    )


def run_training(pre_model):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pre_model)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_train, tokenized_test = prepare_dataset()

    model = AutoModelForSequenceClassification.from_pretrained(
        pre_model, num_labels=2, id2label=id2label, label2id=label2id, trust_remote_code=True,
    ).to(device)

    trainer = Trainer(
        model=model,
        args=get_training_args(pre_model),
        train_dataset=tokenized_train["train"],
        eval_dataset=tokenized_test["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    best_trials = trainer.hyperparameter_search(
        direction=["minimize", "maximize"],
        backend="optuna",
        hp_space=optuna_hp_space,
        n_trials=40,
    )
    trainer.train()
    trainer.evaluate()


# %%
curr_model = "bert-base-multilingual-cased"
run_training(curr_model)
# %%
curr_model = "bert-base-cased"
run_training(curr_model)
# %%
curr_model = "ltg/norbert3-base"
run_training(curr_model)
# %%
wandb
