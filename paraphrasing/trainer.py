import os
from pathlib import Path
import wandb
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import torch
from tqdm import tqdm
import evaluate as hf_evaluate
import pandas as pd
from pyspark.sql import SparkSession
import pyspark.sql.functions as sf
import re
from tweet_json_explorer import df_tweet
from dataclasses import dataclass, field
import argparse
import json
from typing import List



# %%
@dataclass
class ScriptArguments:
    model_name: str = "normistral-ui0"
    model_path: str = "norallm/normistral-7b-warm"
    torch_dtype: str = "float16"
    batch_size: int = 8
    epochs: int = 5
    learning_rate: int = 0.000009
    target_modules: List[str] 
    date: str = "1505"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for running model inference")
    parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config_data = json.load(f)
    
    script_args = ScriptArguments(**config_data)
    return script_args


# %%
spark = SparkSession.builder.config("spark.driver.memory", "10g").appName("TwitterThreadsAnalysis").getOrCreate()


# group the tweets in df_tweet into threads
def find_thread():
    replies_df = df_tweet.select(sf.col("id").alias("child_id"), sf.col("in_reply_to_status_id_str").alias("parent_id"
                                                                                                           ),
                                 sf.col("created_at").alias("child_created_at")).where(
        sf.col("in_reply_to_status_id_str").isNotNull())

    # uses for loop for memory issues
    # assumes no thread is larger than 10
    for _ in range(10):
        replies_df = (replies_df.alias("r1")
        .join(replies_df.alias("r2"), sf.col("r1.parent_id") == sf.col("r2.child_id"), "left_outer")
        .select(
            sf.col("r1.child_id"),
            sf.coalesce(sf.col("r2.parent_id"), sf.col("r1.parent_id")).alias("parent_id"),

        ))

    thread = replies_df.groupBy("parent_id").agg(
        sf.array_union(
            sf.array(sf.col("parent_id")),
            sf.collect_list("child_id"),

        ).alias("thread_members")
    )

    # parent_id, thread_members
    return thread


# %%
thread_df = find_thread()


# %%

# remove the tweets that are updates to other tweets ie. remove those that are NOT parent tweet
def remove_updates_data(data, thread_df):
    thread_df_pandas = thread_df.toPandas()
    parent_ids = thread_df_pandas["parent_id"].tolist()
    parent_ids = [int(id) for id in parent_ids if id.isdigit()]
    filtered_data = data[data["nrk_id"].isin(parent_ids)]
    return filtered_data


# %%
def load_data():
    data_path = 'data/aligning_data.csv' # data is saved locally, not in git
    data = pd.read_csv(data_path)

    data = remove_updates_data(data, thread_df=thread_df)
    dataset = Dataset.from_pandas(data)

    dataset = dataset.rename_column("svv_text", "input_text")
    dataset = dataset.rename_column("nrk_text", "target_text")
    dataset = dataset.remove_columns(
        ['nrk_id', 'recordId', 'situationId', 'similarity', 'svv_ts', 'nrk_ts', '__index_level_0__'])

    try:
        # first split data into train and test
        split_data = dataset.train_test_split(test_size=0.3, seed=42)  # keep 30% for combining validation and test
        train_data = split_data['train']
        test_val_data = split_data['test']

        # split then test_val_data into test and validation 
        final_splits = test_val_data.train_test_split(test_size=0.5, seed=42) 
        test_data = final_splits['test']
        validation_data = final_splits['train']

        return DatasetDict({
            'train': train_data,
            'validation': validation_data,
            'test': test_data
        })
    except AttributeError as e:
        print(f"AttributeError: {e}")

# %%
def create_instruction(example):
    example['prediction'] = "Omformuler følgende setning: " + example["input_text"] + "\nOmskrevet setning: " + example[
        "target_text"] 
    return example


def set_model(model, torch_type):
    model_checkpoint = model
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint, device_map='auto', load_in_8bit=True, torch_dtype=torch_type)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    return model, tokenizer


def set_lora_config(target_modules):
    lora_config = LoraConfig(
        r=16, 
        target_modules=target_modules,
        lora_alpha=8,
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    return lora_config

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")


# %%

def main():
    args = parse_arguments()

    model_name = args.model_name
    model_path = args.model_path
    date = args.date
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    target_modules_lora = args.target_modules
    torch_type =  getattr(torch, args.torch_dtype)

    wandb.init(project="", entity="")
   

    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    }

    data_dict = load_data()
    train_dataset = data_dict['train']
    val_dataset = data_dict['validation']
    test_dataset = data_dict['test']
    test_dataset.set_format(type='pandas') 
    df_test = test_dataset.to_pandas()
    df_test.to_csv(f'Test_dataset/{model_name}_test_dataset.csv', index=False)


    train_dataset = train_dataset.map(create_instruction)
    val_dataset = val_dataset.map(create_instruction)

    model, tokenizer = set_model(model_path, torch_type=torch_type)
    tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #model.to(device)
    print(f"Model is using {device}")

    model_modules = str(model.modules)
    pattern = r'\((\w+)\): Linear'
    linear_layer_names = re.findall(pattern, model_modules)


    names = []
    for name in linear_layer_names:
        names.append(name)

    target_modules = list(set(names))
    print(target_modules)

    lora_config = set_lora_config(target_modules_lora)
    model = get_peft_model(model, lora_config)

    print_model_parameters(model)



    train_dataset = train_dataset.map(
        lambda samples: tokenizer(samples['prediction'], padding=True, truncation=True, max_length=300),
        batched=True)

    val_dataset = val_dataset.map(
        lambda samples: tokenizer(samples['prediction'], padding=True, truncation=True, max_length=300),
        batched=True)

    training_args = TrainingArguments(
        evaluation_strategy='epoch',
        eval_steps=500,  
        logging_strategy='steps',
        logging_steps=100,  
        per_device_train_batch_size=8,
        per_device_eval_batch_size=64,
        num_train_epochs=3,
        warmup_steps=500,
        learning_rate=5e-5,
        save_strategy='epoch',
        load_best_model_at_end=True,  
        metric_for_best_model='eval_loss',  
        output_dir=f'./{model_name}_output_{date}/',
        report_to="wandb",  

    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt"),
        eval_dataset=val_dataset,
        args=training_args

    )
    trainer.train()

    checkpoint_dir = os.path.join(f"./{model_name}_save_{date}/", "checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer.model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)


if __name__ == "__main__":
    main()
