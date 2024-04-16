import evaluate
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, TrainingArguments, \
    DataCollatorWithPadding, EvalPrediction

#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
accuracy = evaluate.load("accuracy")
#%%
