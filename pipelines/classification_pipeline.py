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
from transformers.integrations import WandbCallback
from transformers.trainer_utils import has_length
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

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


class TrainerOverrider(Trainer):

    def __init__(self, class_weights, *args, **kwargs):
        self.class_weights = class_weights
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        labels = torch.eye(2, device="cuda:0")[labels]
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss = torch.nn.BCEWithLogitsLoss(weight=torch.tensor(self.class_weights)).cuda()
        loss = loss(logits.squeeze(), labels.squeeze())
        return (loss, outputs) if return_outputs else loss


# %%

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 9e-8, 1e-3, log=True),
    }


def wandb_hp_space(trial):
    return {
        "project": project_name,
        "description": pre_model,
        "method": "bayes",
        "metric": {"name": "precision", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 1e-8, "max": 1e-5},
            "batch_size": {
                "distribution": "q_log_uniform_values",
                "max": 64,
                "min": 8,
                "q": 8,
            },
            "dropout": {"values": [0.3, 0.4, 0.5]},
            "optimizer": {"values": ["adamw"]},
        },
    }


def model_init(trial):
    return AutoModelForSequenceClassification.from_pretrained(
        pre_model, num_labels=2, id2label=id2label, label2id=label2id, trust_remote_code=True,
        problem_type="single_label_classification"
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
    print(classification_report(y_pred=predictions, y_true=labels, labels=[0, 1]))
    pr = precision.compute(predictions=predictions, references=labels)
    a = accuracy.compute(predictions=predictions, references=labels)
    r = recall.compute(predictions=predictions, references=labels)
    f = f1.compute(predictions=predictions, references=labels)
    return {**a, **pr, **r, **f}


def get_training_args(pre_model):
    return TrainingArguments(
        output_dir="model_run",
        logging_dir="model_run_logs",
        per_device_train_batch_size=16,
        learning_rate=lr if lr is not None else 5e-5,
        dataloader_num_workers=4,
        do_train=True,
        num_train_epochs=epoch if epoch is not None else 1,
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


def run_training(pre_model, class_weight=None):
    print(f"started fine-tuning {pre_model}")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pre_model)
    # data_collator = DefaultDataCollator()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    print(f"Current training file {training_file}")

    tokenized_train, tokenized_test = prepare_dataset()
    if class_weight is None:
        class_weights = compute_class_weight(class_weight="balanced", y=tokenized_train['train']['label'],
                                             classes=np.unique(tokenized_train['train']['label']))

    model = AutoModelForSequenceClassification.from_pretrained(
        pre_model, num_labels=2, id2label=id2label, label2id=label2id, trust_remote_code=True,
    ).to(device)

    wandb.init(reinit=True, project=project_name, notes=f"{class_weights}_{sampling_type}_{category}", name=pre_model)
    model_output_folder = f"model_run/{wandb.run.id}"

    if not os.path.exists(model_output_folder):
        os.mkdir(model_output_folder)

    trainer = TrainerOverrider(
        model=model,
        # model_init=model_init,
        args=get_training_args(pre_model),
        train_dataset=tokenized_train["train"],
        eval_dataset=tokenized_test["train"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=pre_process_logits,
        callbacks=[WandbCallback],
        class_weights=class_weights
    )

    # trainer.hyperparameter_search(n_trials=5, backend="wandb", hp_space=wandb_hp_space)
    print(f"Size of train dataset {len(tokenized_train['train'])}")
    print(f"Size of test dataset {len(tokenized_test['train'])}")
    print(f"Class weights: {class_weights}")

    trainer.train()
    trainer.save_model(model_output_folder)

    wandb.finish()


# %%
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    accuracy = evaluate.load("accuracy")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")
    f1 = evaluate.load("f1")

    project_name = "m_classification_4th"

    lr = 2e-5
    epoch = 5

    category = "without_night"
    path_to_data = f"data/pipeline_runs/classification/{category}/"
    test_file = f"{path_to_data}static_test.csv"
    print(f"Currently running on {category}, with path {path_to_data}")

    sampling_type = "100k_as_negative.csv"
    training_file = f"{path_to_data}{sampling_type}"

    curr_model = "ltg/norbert3-large"
    pre_model = curr_model
    run_training(curr_model)

    curr_model = "bert-base-multilingual-cased"
    pre_model = curr_model
    run_training(curr_model)


    sampling_type = "after_2020_rest_90_percent.csv"
    training_file = f"{path_to_data}{sampling_type}"

    curr_model = "ltg/norbert3-large"
    pre_model = curr_model
    run_training(curr_model)

    curr_model = "bert-base-multilingual-cased"
    pre_model = curr_model
    run_training(curr_model)



    sampling_type = "all_except_test.csv"
    training_file = f"{path_to_data}{sampling_type}"

    curr_model = "ltg/norbert3-large"
    pre_model = curr_model
    run_training(curr_model)

    curr_model = "bert-base-multilingual-cased"
    pre_model = curr_model
    run_training(curr_model)
#%%

    category = "with_night"
    path_to_data = f"data/pipeline_runs/classification/{category}/"
    test_file = f"{path_to_data}static_test.csv"

    print(f"Currently running on {category}, with path {path_to_data}")

    sampling_type = "100k_as_negative.csv"
    training_file = f"{path_to_data}{sampling_type}"

    curr_model = "ltg/norbert3-large"
    pre_model = curr_model
    run_training(curr_model)

    curr_model = "bert-base-multilingual-cased"
    pre_model = curr_model
    run_training(curr_model)


    sampling_type = "after_2020_rest_90_percent.csv"
    training_file = f"{path_to_data}{sampling_type}"

    curr_model = "ltg/norbert3-large"
    pre_model = curr_model
    run_training(curr_model)

    curr_model = "bert-base-multilingual-cased"
    pre_model = curr_model
    run_training(curr_model)



    sampling_type = "all_except_test.csv"
    training_file = f"{path_to_data}{sampling_type}"

    curr_model = "ltg/norbert3-large"
    pre_model = curr_model
    run_training(curr_model)

    curr_model = "bert-base-multilingual-cased"
    pre_model = curr_model
    run_training(curr_model)