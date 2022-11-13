#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os
import pandas as pd
#import pyarrow.parquet as pq
import sys
import time
import torch

from torch import nn

try:
  from tqdm import tqdm
except ImportError:
  tqdm = lambda x: x

from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from scipy.special import softmax

from tweet_utils import *

import transformers
from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers.optimization import Adafactor, get_cosine_with_hard_restarts_schedule_with_warmup #AdafactorSchedule
#transformers.logging.set_verbosity_info()

class LinearHead(nn.Module):
   def __init__(self, hidden_size, num_labels):
     super(LinearHead, self).__init__()
     self.out_proj = nn.Linear(hidden_size, num_labels)

   def forward(self, features, **kwargs):
     x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
     x = self.out_proj(x)
     return x

def main():
    parser = argparse.ArgumentParser(description='Finetune multilingual transformer model')
    parser.add_argument('-o','--output_file', dest='output_file', type=str, default=None, help='Output file for predictions.')
    parser.add_argument('-n','--nepochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('-m','--modelname', type=str, default='models/model.bin', help='Output filename for model (leave empty for not saving)')
    parser.add_argument('-t','--tokenizername', type=str, default='models/tokenizer.bin', help='Output filename for tokenizer (leave empty for not saving)')
    parser.add_argument('-T','--nthreads', type=int, default=None, help='Number of threads for tokenizing (default: OMP_NUM_THREADS)')
    parser.add_argument('-s','--seed', type=int, default=0, help='Random seed (default: 0)')
    parser.add_argument('-b','--batchsize', type=int, default=24, help='Per-device batchsize (default: 24)')
    parser.add_argument('-l','--linear', action="store_true", help="Train only linear model")
    parser.add_argument('--fp16', action="store_true", help="Use fp16 mixed precision")
    parser.add_argument('--fp32', dest='fp16', action="store_false", help="Use fp32 precision")
    parser.add_argument('-L','--lrscheduler', type=str, default="cosine", choices=['linear', 'cosine', 'cosine_with_restarts','polynomial','constant','constant_with_warmup'], help="Learning Rate Scheduler")
    parser.add_argument('files', nargs="+", type=str, metavar='INPUTFILES', help="Input files (in parquet format)")
    args = parser.parse_args()

    if not args.files:
      print("Usage: {} INPUTFILES".format(sys.argv[0]))
      raise SystemExit

    print(f"Loading {len(args.files)} data files.")
    df = load_data(args.files)

    tokenizer = XLMRobertaTokenizer.from_pretrained('sentence-transformers/stsb-xlm-r-multilingual')
    if args.tokenizername:
      print(f"Writing tokenizer {args.tokenizername}")
      tokenizer.save_pretrained(args.tokenizername)

    num_workers = get_num_workers(args.nthreads)
    print(f"Using pool with {num_workers} workers for tokenization.")
    tokenized_results = tokenize_dataset(df, tokenizer=tokenizer, num_workers=num_workers)

    # insert results 
    df['input_ids'] = sum([part['input_ids'] for part in tokenized_results],[])
    df['attention_mask'] = sum([part['attention_mask'] for part in tokenized_results],[])

    val_size = min(len(df)//10, 10000)
    df_train, df_val = train_test_split(df, test_size=val_size, random_state=args.seed)
    del df
    data_train = TwitterDataset(df_train)
    del df_train
    data_val = TwitterDataset(df_val)
    del df_val

    if args.linear:
      # create model with frozen sentence embedding part
      with torch.no_grad():
        model = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/stsb-xlm-r-multilingual')
        for param in model.parameters():
          param.requires_grad = False
      model.classifier = LinearHead(hidden_size=768, num_labels=2)
    else:
      model = AutoModelForSequenceClassification.from_pretrained('sentence-transformers/stsb-xlm-r-multilingual')
 
    num_gpus = max(1, torch.cuda.device_count())
    print(f"Using {num_gpus} devices to train.")
    training_args = TrainingArguments(
      output_dir='./results/',
      report_to = "all",
      adafactor=True,                  # not using AdamW, let's see
      learning_rate=5e-5,               # default is 5e-5
      num_train_epochs=args.nepochs,
      per_device_train_batch_size=args.batchsize,
      per_device_eval_batch_size=args.batchsize,
      #warmup_steps=500,                # number of warmup steps for learning rate scheduler
      warmup_steps=0.1,                # number of warmup steps for learning rate scheduler
      #lr_scheduler_type=args.lrscheduler,
      #weight_decay=0.0,               # strength of weight decay
      logging_dir='./logs/',            # directory for storing logs
      logging_steps=100,               # log often
      evaluation_strategy="steps",
      eval_steps=500,                  # evaluate often
      save_strategy="epoch",            # save rarely
      fp16 = args.fp16,
      #fp16_full_eval=True
    )
    num_steps_per_epoch = len(data_train)//(args.batchsize*num_gpus)
    num_updates = num_steps_per_epoch * args.nepochs
    if num_updates > 5000:
      training_args.eval_steps = 1000 # evaluate less often for big datasets
    if num_updates < 100:
      training_args.logging_steps = 1
      training_args.eval_steps = num_updates//5
    elif num_updates < 1000:
      training_args.logging_steps = 10
      training_args.eval_steps = num_updates//10

    optimizer = Adafactor(model.parameters(), scale_parameter=False, relative_step=False, warmup_init=False, lr=5e-5)
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer, 
                                                                      num_warmup_steps=0,
                                                                      num_training_steps=num_updates, 
                                                                      num_cycles=args.nepochs)
    
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=data_train,
      eval_dataset=data_val,
      compute_metrics=compute_metrics,
      optimizers=(optimizer, lr_scheduler)
    )
    trainer.train()
    trainer.evaluate()

    del data_train, data_val

    if args.modelname:
      print(f"Writing final model {args.modelname}")
      trainer.save_model(args.modelname)

if __name__ == "__main__":
    main()
