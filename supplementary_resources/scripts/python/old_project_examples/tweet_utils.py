from functools import partial
import multiprocessing
import numpy as np
import os
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, roc_auc_score
import torch

def load_parquet(filename):
  df = pq.read_table(filename).to_pandas()
  df.set_index('id')
  return df

def load_data(filenames):
  df = pd.concat(load_parquet(f) for f in filenames)
  return df

class TwitterDataset(torch.utils.data.Dataset):
  def __init__(self, df):
    self.input_ids = df.input_ids.to_numpy()
    self.attention_mask = df.attention_mask.to_numpy()
    self.labels = df.label.to_numpy()

  def __getitem__(self, idx):
    item = {}
    item['input_ids'] = torch.tensor(self.input_ids[idx])
    item['attention_mask'] = torch.tensor(self.attention_mask[idx])
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

def compute_metrics(pred):
  labels = pred.label_ids
  scores = pred.predictions[:,1]-pred.predictions[:,0] # logits, so not normalized
  preds = pred.predictions.argmax(axis=-1)
  acc = accuracy_score(labels, preds)
  auc = roc_auc_score(labels, scores)
  return { 'accuracy': acc, 'auc': auc }

def tokenize(tokenizer, df):
  return tokenizer.batch_encode_plus(df.text, padding="max_length", max_length=160, truncation=True)

def tokenize_dataset(df, tokenizer, num_workers):
  print(f"Will use pool with {num_workers} workers for tokenization.")
  with multiprocessing.Pool(num_workers) as pool:
    chunks = np.array_split(df, num_workers)
    tokenized_results = pool.map(partial(tokenize, tokenizer), chunks)
  return tokenized_results

def get_num_workers(max_nthreads):
  num_workers = max(1, multiprocessing.cpu_count()-1)
  if max_nthreads:
    num_workers = min(num_workers, max_nthreads)
  else:
    num_workers = min(num_workers, int(os.environ.get("OMP_NUM_THREADS", 1000000)))
  return num_workers
