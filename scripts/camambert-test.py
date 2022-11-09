import pandas as pd
import numpy as np
import re

from transformers import CamembertModel, CamembertTokenizer, AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm

file = 'ua_dataset.jsonl'
model_name = "camembert-base"

output_model = 'reg_saved.pth'

epochs = 3
batch_size = 16

tokenizer = CamembertTokenizer.from_pretrained(model_name)
coord_scaler = StandardScaler()

def filter_websites(text):
    pattern = r'(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z][a-z\-]*'
    text = re.sub(pattern, '', text)
    return text

def filter_long_descriptions(tokenizer, descriptions, max_len):
    indices = []
    lengths = tokenizer(descriptions, padding=False,
                     truncation=False, return_length=True)['length']
    for i in range(len(descriptions)):
        if lengths[i] <= max_len-2:
            indices.append(i)
    return indices

def create_dataloaders(inputs, masks, labels, batch_size):
    input_tensor = torch.tensor(inputs)
    mask_tensor = torch.tensor(masks)
    labels_tensor = torch.tensor(labels)
    dataset = TensorDataset(input_tensor, mask_tensor,
                            labels_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True)
    return dataloader

class BertRegressor(nn.Module):
    def __init__(self, drop_rate=0.2, freeze_camembert=False):
        super(BertRegressor, self).__init__()
        D_in, D_out = 768, 1

        self.bert = CamembertModel.from_pretrained(model_name, return_dict=True)
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(D_in, D_out))

    def forward(self, input_ids, attention_masks):

        outputs = self.bert(input_ids, attention_masks)
        class_label_output = outputs[1]
        outputs = self.regressor(class_label_output)
        return outputs

def train(model, optimizer, scheduler, loss_function, epochs,
          train_dataloader, device, clip_value=2):
    for epoch in range(epochs):
        print("Epoch:", epoch)
        print("-----")
        best_loss = 1e10
        model.train()
        for step, batch in enumerate(train_dataloader):
            print("Step:", step)
            batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
            model.zero_grad()
            outputs = model(batch_inputs, batch_masks)
            loss = loss_function(outputs.squeeze(),
                             batch_labels.squeeze())
            loss.backward()
            clip_grad_norm(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()

    return model

def evaluate(model, loss_function, test_dataloader, device):
    model.eval()
    test_loss, test_r2 = [], []
    for batch in test_dataloader:
        batch_inputs, batch_masks, batch_labels = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            outputs = model(batch_inputs, batch_masks)
        loss = loss_function(outputs, batch_labels)
        test_loss.append(loss.item())
        r2 = r2_score(outputs, batch_labels)
        test_r2.append(r2.item())
    return test_loss, test_r2

def r2_score(outputs, labels):
    labels_mean = torch.mean(labels)
    ss_tot = torch.sum((labels - labels_mean) ** 2)
    ss_res = torch.sum((labels - outputs) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def predict(model, dataloader, device):
    model.eval()
    output = []
    for batch in dataloader:
        batch_inputs, batch_masks, _ = tuple(b.to(device) for b in batch)
        with torch.no_grad():
            res = model(batch_inputs,
                            batch_masks)
            #print(res)
            #print(res.view(1,-1).tolist()[0])
            output += res.view(1,-1).tolist()[0]
            #print(output)
    return output

def pretraining(model):
    model = train(model, optimizer, scheduler, loss_function, epochs,
                  train_dataloader, device, clip_value=2)

    def save(model, optimizer):
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, output_model)

    print(evaluate(model, loss_function, test_dataloader, device))

    save(model, optimizer)

def evalueting(model):
    checkpoint = torch.load(output_model, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    val_set = val_data

    encoded_val_corpus = tokenizer(text=val_set.clear_text.tolist(),
                              add_special_tokens=True,
                              padding='max_length',
                              truncation='longest_first',
                              max_length=300,
                              return_attention_mask=True)

    val_input_ids = np.array(encoded_val_corpus['input_ids'])
    val_attention_mask = np.array(encoded_val_corpus['attention_mask'])
    val_labels = val_set.longitude.to_numpy().astype(np.float32)
    val_labels = coord_scaler.transform(val_labels.reshape(-1, 1))
    val_dataloader = create_dataloaders(val_input_ids,
                             val_attention_mask, val_labels, batch_size)

    y_pred_scaled = predict(model, val_dataloader, device)
    print(y_pred_scaled)

    y_test = val_set.longitude.to_numpy()
    y_pred = coord_scaler.inverse_transform(np.asarray(y_pred_scaled, dtype=np.float32).reshape(-1, 1))

    print(y_pred)

    for i in range(len(y_test)):
        print(y_test[i], y_pred[i][0])

    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import median_absolute_error
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_percentage_error
    from sklearn.metrics import r2_score

    mae = mean_absolute_error(y_test, y_pred)
    mdae = median_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    #mdape = ((pd.Series(y_test) - pd.Series(y_pred)) / pd.Series(y_test)).abs().median()
    r_squared = r2_score(y_test, y_pred)

    print(mae, mdae, mse, mape, r_squared)

data = pd.read_json(path_or_buf=file, lines=True)
print(data.head())
print(data.info())

data['clear_text'] = data.texts.apply(filter_websites)

train_data = data.iloc[100:, :]
val_data = data.iloc[:100, :]

df = train_data
print(df.info())

encoded_corpus = tokenizer(text=df.clear_text.tolist(),
                            add_special_tokens=True,
                            padding='max_length',
                            truncation='longest_first',
                            max_length=300,
                            return_attention_mask=True)

input_ids = encoded_corpus['input_ids']
attention_mask = encoded_corpus['attention_mask']

short_descriptions = filter_long_descriptions(tokenizer, df.clear_text.tolist(), 300)
input_ids = np.array(input_ids)[short_descriptions]
attention_mask = np.array(attention_mask)[short_descriptions]
labels = df.longitude.to_numpy()[short_descriptions].astype(np.float32)

test_size = 0.1
seed = 42

train_inputs, test_inputs, train_labels, test_labels = train_test_split(input_ids,
                                                                        labels,
                                                                        test_size=test_size,
                                                                        random_state=seed)

train_masks, test_masks, _, _ = train_test_split(attention_mask,
                                                 labels,
                                                 test_size=test_size,
                                                 random_state=seed)


coord_scaler.fit(train_labels.reshape(-1, 1))

train_labels = coord_scaler.transform(train_labels.reshape(-1, 1))
test_labels = coord_scaler.transform(test_labels.reshape(-1, 1))

train_dataloader = create_dataloaders(train_inputs, train_masks,
                                      train_labels, batch_size)
test_dataloader = create_dataloaders(test_inputs, test_masks,
                                     test_labels, batch_size)

model = BertRegressor(drop_rate=0.2)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU.")
else:
    print("No GPU available, using the CPU instead.")
    device = torch.device("cpu")

model.to(device)


optimizer = AdamW(model.parameters(),
                  lr=5e-5,
                  eps=1e-8)


total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                 num_warmup_steps=0, num_training_steps=total_steps)

loss_function = nn.MSELoss()

#pretraining(model)
evalueting(model)
