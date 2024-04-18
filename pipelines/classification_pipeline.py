import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, TrainingArguments, \
    DataCollatorWithPadding, EvalPrediction

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
accuracy = evaluate.load("accuracy")
training_file = "data/pipeline_runs/classification/threshold: 0.9, negative_size:300000 - d:18 m:4 h:12/train.csv"
test_file = "data/pipeline_runs/classification/threshold: 0.9, negative_size:300000 - d:18 m:4 h:12/test.csv"
# test_file = "data/pipeline_runs/classification/threshold: 0.9, negative_size:300000 - d:18 m:4 h:12/small.csv"

# %%
label2id = {'post': 1, 'discard': 0}
id2label = {1: 'post', 0: 'discard'}

def preprocess_text(df):
    #TODO: should i use [CLS] and such here
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
    return accuracy.compute(predictions=predictions, references=labels)
def get_training_args(pre_model):
    return TrainingArguments(
        output_dir="model_run",
        learning_rate=9e-6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        do_train=True,
        num_train_epochs=1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=4,
        run_name=pre_model
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
    trainer.train()
    trainer.evaluate()

 #%%
curr_model = "bert-base-multilingual-cased"
run_training(curr_model)