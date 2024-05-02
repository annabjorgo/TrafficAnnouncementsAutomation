import os
from typing import Any
from typing import Dict, Union

import evaluate
import numpy as np
from tqdm.auto import tqdm

import torch
import wandb
from datasets import load_dataset
from optuna import Trial
from setfit import SetFitModel
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, TrainingArguments, \
    DataCollatorWithPadding, IntervalStrategy, ProgressCallback, DefaultDataCollator
from transformers.trainer_utils import has_length
from sklearn.metrics import classification_report

# %%

# %%
label2id = {'post': 1, 'discard': 0}
id2label = {1: 'post', 0: 'discard'}

class ProgressOverider(ProgressCallback):
    def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
        if state.is_world_process_zero and has_length(eval_dataloader):
            if self.prediction_bar is None:
                self.prediction_bar = tqdm(
                    total=len(eval_dataloader), dynamic_ncols=True
                )
            self.prediction_bar.update(1)

#%%

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 9e-8, 1e-3, log=True),
    }

def wandb_hp_space(trial):
    return {
        "method": "random",
        "metric": {"name": "precision", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-8, "max": 1e-5},
        },
    }


def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
        pre_model, num_labels=2, id2label=id2label, label2id=label2id, trust_remote_code=True,
    ).to(device)

def preprocess_text(df):
    # TODO: should i use [CLS] and such here
    # tokenized_text = tokenizer(list(map(lambda x: "[CLS]" + x + "[SEP]", df['concat_text'])))
    tokenized_text = tokenizer(df['concat_text'])
    tokenized_text["label"] = df['post']
    return tokenized_text


def prepare_dataset():
    train_data = load_dataset("csv", data_files=training_file) \
        .map(preprocess_text, batched=True)
    test_data = load_dataset("csv", data_files=test_file) \
        .map(preprocess_text, batched=True)
    return train_data, test_data

def pre_process_logits(pred, label):
    return torch.argmax(pred, axis=1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    print(classification_report(y_pred=predictions, y_true=labels, labels=[0,1]))
    pr = precision.compute(predictions=predictions, references=labels)
    a = accuracy.compute(predictions=predictions, references=labels)
    r = recall.compute(predictions=predictions, references=labels)
    f = f1.compute(predictions=predictions, references=labels)
    return {**a, **pr, **r, **f}



def get_training_args(pre_model):
    return TrainingArguments(
        output_dir="model_run",
        logging_dir="model_run_logs",
        per_device_train_batch_size=8,
        learning_rate=9e-6,
        dataloader_num_workers=4,
        do_train=True,
        num_train_epochs=1,
        weight_decay=0.01,
        save_steps=0.15,
        eval_steps=0.15,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=4,
        run_name=f"{pre_model}",
        report_to="wandb"
    )


def run_training(pre_model):
    print(f"started fine-tuning {pre_model}")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pre_model)
    # data_collator = DefaultDataCollator()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"Current training file {training_file}")

    tokenized_train, tokenized_test = prepare_dataset()
    print(f"Size of train dataset {len(tokenized_train['train'])}")
    print(f"Size of test dataset {len(tokenized_test['train'])}")

    model = AutoModelForSequenceClassification.from_pretrained(
        pre_model, num_labels=2, id2label=id2label, label2id=label2id, trust_remote_code=True,
        problem_type="single_label_classification"
    ).to(device)

    wandb.init(reinit=True, project=project_name, notes=sampling_type, name=pre_model)
    trainer = Trainer(
        model=model,
        # model_init=model_init,
        args=get_training_args(pre_model),
        train_dataset=tokenized_train["train"],
        eval_dataset=tokenized_test["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=pre_process_logits,
    )

    # progress_callback = next(filter(lambda x: isinstance(x, ProgressCallback), trainer.callback_handler.callbacks),
    #                              None)
    # trainer.remove_callback(progress_callback)
    # trainer.add_callback(ProgressOverider)
    # trainer.hyperparameter_search(n_trials=5)

    trainer.train()
    # trainer.evaluate(eval_dataset=tokenized_validate['train'])
    wandb.finish()


# %%
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    project_name = "m_classification_2nd"
    path_to_data = "data/pipeline_runs/classification/"
    sampling_type = "no_night_100k_as_negative.csv"
    training_file = f"{path_to_data}{sampling_type}"
    test_file = f"{path_to_data}static_test.csv"

    curr_model = "bert-base-multilingual-cased"
    pre_model = curr_model
    run_training(curr_model)

    curr_model = "ltg/norbert3-large"
    pre_model = curr_model
    run_training(curr_model)

    sampling_type = "no_night_after_2020_rest_90_percent.csv"
    training_file = f"{path_to_data}{sampling_type}"

    curr_model = "bert-base-multilingual-cased"
    pre_model = curr_model
    run_training(curr_model)

    curr_model = "ltg/norbert3-large"
    pre_model = curr_model
    run_training(curr_model)

    sampling_type ="no_night_all_except_test.csv"
    training_file = f"{path_to_data}{sampling_type}"

    curr_model = "bert-base-multilingual-cased"
    pre_model = curr_model
    run_training(curr_model)

    curr_model = "ltg/norbert3-large"
    pre_model = curr_model
    run_training(curr_model)
