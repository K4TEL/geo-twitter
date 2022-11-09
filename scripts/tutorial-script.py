from datasets import load_dataset

dataset = load_dataset("yelp_review_full")

print(dataset["train"][100])
dataset_df = dataset["train"].to_pandas()
print(dataset_df.head())

features = dataset["train"].features
print(features)

print(dataset_df["label"].value_counts(normalize=True).sort_index())
dataset = dataset.rename_column("label", "labels")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)

import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import TrainingArguments, Trainer
from torch import nn
import torch

class_weights = (1 - (dataset_df["label"].value_counts().sort_index() / len(dataset_df))).values
print(class_weights)
class_weights = torch.from_numpy(class_weights).float()
print(class_weights)

class WeightedLossTrainer(Trainer):
    def compute_loss(selfself, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss_func = nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_func(logits, labels)
        return (loss, outputs) if return_outputs else loss

batch_size = 8
logging_steps = len(dataset["train"]) // batch_size
output_dir = "test_trainer"
training_args = TrainingArguments(output_dir=output_dir,
                                  num_train_epochs=3,
                                  learning_rate=2e-5,
                                  per_device_train_batch_size=batch_size,
                                  per_device_eval_batch_size=batch_size,
                                  weight_decay=0.01,
                                  logging_steps=logging_steps,
                                  evaluation_strategy="epoch")

trainer = WeightedLossTrainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

trainer.save_model('save_test/model')
# alternative saving method and folder
model.save_pretrained('saving_test')
