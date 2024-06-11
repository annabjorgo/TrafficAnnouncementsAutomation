import pandas as pd
import numpy as np
import torch
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import evaluate
from peft import PeftModel, PeftConfig
from dataclasses import dataclass, field
import argparse
import json

hg_token = ""


@dataclass
class ScriptArguments:
    model_path: str = "norallm/normistral-7b-warm"
    torch_dtype: str = "float16"
    model_name: str = "normistral-ui0"
    date: str = "1505"
    peft_model_id: str = "Models/mistral_save_0805/checkpoint"
    max_length: int = 200


def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for running model inference")
    parser.add_argument("--config", type=str, required=True, help="Path to the config JSON file")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config_data = json.load(f)
    
    script_args = ScriptArguments(**config_data)
    return script_args


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

def prepare_data(df):
    df['instruction'] = "Omformuler følgende setning: " + df["input_text"] + "\nOmskrevet setning: " 
    return df


def load_model(model_path, peft_model_id, torch_dtype, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,
        #attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
        #token
        #device_map="auto" 
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    tokenizer = AutoTokenizer.from_pretrained(peft_model_id)
    # tokenizer = AutoTokenizer.from_pretrained(model_path)

    model.to(device)
    return model, tokenizer


def remove_instruction(example):
    return re.sub(r'Omformuler følgende setning: ', '', example)


def remove_prefix(text, prefix="Omskrevet setning:"):
    parts = text.split(prefix)
    if len(parts) > 1:
        return parts[1].strip()
    else:
        return text


def remove_urls(example):
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    text_without_urls = re.sub(url_pattern, '', example)
    text_cleaned = re.sub(r'\s+', ' ', text_without_urls).strip()
    return text_cleaned


def generate_rephrases(df, model, device, tokenizer, max_length):
    model.eval()
    for idx, example in tqdm(df.iterrows(), total=len(df)):
        input_text = example['instruction']

        inputs_ids = tokenizer(input_text, 
                                return_tensors="pt", 
                                max_length=max_length, 
                                truncation=True,
                                return_token_type_ids=False)
        inputs_ids = {k: v.to(device) for k, v in inputs_ids.items()}
        

        output_ids = model.generate(**inputs_ids, max_new_tokens=max_length, num_return_sequences=1, do_sample=False, num_beams=3, no_repeat_ngram_size=2 )
        generated_output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


        try:   
            output_processed = remove_prefix(generated_output)
            df.at[idx, "rephrase"] = output_processed
            
        except:
            print("ERROR: Something wrong happened during rephrasing")

     
 
def calculate_corpus_metrics(df):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    meteor = evaluate.load("meteor")

    predictions = df['rephrase'].tolist()
    references = [[ref] for ref in df['target_text'].tolist()]


    
    bleu_result = bleu.compute(predictions=predictions, references=references)
    meteor_result = meteor.compute(predictions=predictions, references=references)
    rouge_result = rouge.compute(predictions=predictions, references=references)

    return {
            'BLEU': bleu_result['bleu'],
            'METEOR': meteor_result['meteor'],
            'ROUGE-1': rouge_result['rouge1'],
            'ROUGE-2': rouge_result['rouge2'],
            'ROUGE-L': rouge_result['rougeL']
        }


def count_sentences(text):
    sentences = re.split(r'[.!?]', text)
    sentences = [s for s in sentences if s.strip()]
    return len(sentences)


def clean_text(text, max_sentences, rephrase):
    # urls
    text = re.sub(r'https?://[^\s]+|www\.[^\s]+', '', text)

    # dates
    text = re.sub(r'\b(?:\d{1,2}[/-])?(?:\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b)?(?:\d{1,2}[/-])?\d{4}\b', '', text)

    # hashtags
    text = re.sub(r'#\w+', '', text)

    # Twitter handles
    text = re.sub(r'@\w+', '', text)
    
    # question marks and exclamation points
    # text = re.sub(r'[?!]', '', text)
    
    # lowercase
    # text = text.lower()
    
    # punctuation
    # text = text.translate(str.maketrans('', '', string.punctuation))
    
    # special characters
    # text = re.sub(r'[^a-zA-ZæøåÆØÅ\s]', '', text)
        
    # redundant whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # remove extra sentences to not measure hallucination
    if rephrase:
        sentences = re.split(r'(?<=[.!?]) +', text)
        if len(sentences) > max_sentences + 1:
            sentences = sentences[:max_sentences + 1]
            text = ' '.join(sentences)
    return text
    

def main():
    wandb.init(project="", entity="")
    args = parse_arguments()
    torch_dtype = getattr(torch, args.torch_dtype)



    with torch.no_grad():
        torch.cuda.empty_cache()

    data_path = "data/test_dataset.csv"
    df = load_data(data_path)

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(model_path=args.model_path, peft_model_id=args.peft_model_id, torch_dtype=torch_dtype, device=device)
    generate_rephrases(df, model, device, tokenizer, max_length=args.max_length)

    df.to_csv(f'{args.model_name}_{args.date}_evaluation_rephrases.csv', index=False)
   

    
    df = df.dropna()
    df['target_text'] = df.apply(lambda row: clean_text(row['target_text'], count_sentences(row['target_text']), False), axis=1)
    df['rephrase'] = df.apply(lambda row: clean_text(row['rephrase'], count_sentences(row['input_text']), True), axis=1)

    print("after cleaning")
    df_str = df[["input_text", "target_text", "rephrase"]].head(50).to_string()
    print(df_str)

    metrics = calculate_corpus_metrics(df)
    print(metrics)
    wandb.log(metrics)

    with torch.no_grad():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()