import numpy as np
import os
import string
import pandas as pd
import torch
import re
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import GPUtil
import psutil

# dataset wrapper


def load_jsonl(filename):
    filename = f"datasets/{filename}"
    print(f"DATASET\tLOAD: {psutil.virtual_memory().percent}%\tLoading dataset from {filename}")
    data = pd.read_json(path_or_buf=filename, lines=True)
    print(f"DATASET\tLOAD: {psutil.virtual_memory().percent}%\tDataset of {len(data.index)} samples is loaded")
    return data


def save_df(df, filename, prefix=None):
    if prefix is None:
        prefix = "td-"

    size = len(df.index)
    save_filename = f"datasets/{prefix}{size}-{filename}"

    with open(save_filename, "w") as f:
        df.to_json(f, orient='records', lines=True)
    print(f"DATASET\tSAVE\tTwitter Dataset of {size} samples is written to file: {filename}")


# text preprocessing
def nlp_filtering(text):
    def filter_punctuation(text):
        punctuationfree="".join([i for i in text if i not in string.punctuation])
        return punctuationfree

    def filter_websites(text):
        #pattern = r'(http\:\/\/|https\:\/\/)?([a-z0-9][a-z0-9\-]*\.)+[a-z][a-z\-]*'
        pattern = r'http\S+'
        text = re.sub(pattern, '', text)
        return text

    text = filter_websites(text)
    text = filter_punctuation(text)
    return text


def create_dataloader(inputs, masks, labels, batch_size, shuffle=False):
    dataset = TensorDataset(torch.tensor(inputs), torch.tensor(masks), torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# crop dataset to remove already used data before random sampling
def crop_dataset(data, size, seed):
    print(f"DATASET\tCropping dataset of {len(data.index)} in {size} samples with seed {seed}")
    return data.drop(data.sample(n=size, random_state=seed).index, axis=0)


def sample_users(data, users_n):
    user_list = data['user'].value_counts()[:users_n].index.tolist()
    return data[data['user'].isin(user_list)]


class TwitterDataloader():
    def __init__(self, filename, features, target, tokenizer, seed=42, scaled=False, val_feature=None):
        self.filename = filename
        self.data = load_jsonl(self.filename)
        if "texts" in self.data.columns:
            self.data.rename(columns={'longitude': 'lon',
                                      'latitude': 'lat',
                                      'created_at': 'time',
                                      'texts': 'text'}, inplace=True)
            self.data.to_json(f"datasets/{self.filename}", orient='records', lines=True)
            print(self.data.info())

        self.target = target

        self.tokenizer = tokenizer
        self.seed = seed

        self.scaled = scaled

        self.feature_columns = {
            "GEO": ["text", "place", "user"],
            "NON-GEO": ["text", "user"],
            "NON-USER": ["text", "place"],
            "META-DATA": ["place", "user"],
            "TEXT-ONLY": ["text"],
            "GEO-ONLY": ["place"],
            "USER-ONLY": ["user"],
        }

        self.features = features
        self.n_features = len(features)
        self.key_feature = features[0]
        self.minor_features = features[1:]

        self.val_feature = features[0] if val_feature is None else val_feature

        self.val_dataloader, self.train_dataloader, self.test_dataloader = None, None, None
        self.val_df = None

    # filtering dataset files by column condition
    def filter_dataset(self, column_name, filter_text=None, filter_list=None, save=True):
        if filter_text:
            filter_df = self.data[self.data[column_name] == filter_text]
            prefix = f"f-{column_name}-{filter_text}-td-"
        elif filter_list:
            filter_df = self.data[self.data[column_name].isin(filter_list)]
            prefix = f"f-{column_name}-{'+'.join(filter_list)}-td-"

        if save:
            save_df(filter_df, self.filename, prefix)

    # tokenization - text to IDs and attention masks
    def tokenize(self, column):
        encoded_corpus = self.tokenizer(text=column,
                                        add_special_tokens=True,
                                        padding='max_length',
                                        truncation='longest_first',
                                        max_length=300,
                                        return_attention_mask=True)

        input_ids = encoded_corpus['input_ids']
        attention_masks = encoded_corpus['attention_mask']
        return input_ids, attention_masks

    # forming feature columns, dropping old columns
    def feature_split_filter(self, data):
        text_features = self.features + [self.val_feature] if self.val_feature not in self.features else self.features
        for f in text_features:
            data[f] = data[self.feature_columns[f]].astype(str).agg(" ".join, axis=1) if len(self.feature_columns[f]) > 1 else data[self.feature_columns[f]]
            data[f] = data[f].astype(str).apply(nlp_filtering)

        for f in text_features:
            for column in self.feature_columns[f]:
                if column in data.columns:
                    data = data.drop(columns=[column], axis=1)
        return data

    def form_training(self, batch_size, size, test_ratio, skip_size=0, shuffle=True):
        if skip_size > 0:
            self.data = crop_dataset(self.data, skip_size, self.seed)
        print(f"DATASET\tForming training dataset of {size} samples with test size {int(test_ratio*size)} for features: {', '.join(self.features)}")
        train_df = self.data.sample(n=size, random_state=self.seed).copy()
        del self.data
        train_df = self.feature_split_filter(train_df)
        self.create_feature_dataloaders(train_df, True, batch_size, shuffle, test_ratio)

    def form_validation(self, batch_size, size, by_user=False, skip_size=0):
        if skip_size > 0:
            self.data = crop_dataset(self.data, skip_size, self.seed)

        print(f"DATASET\tForming validation dataset of {size} {'users' if by_user else 'samples'} with batch size {batch_size} for {self.val_feature} text feature")
        if by_user:
            self.val_df = sample_users(self.data, size)
            self.features += ["USER-ONLY"]
            print(f"DATASET\tSize of the validation dataset with {size} users: {len(self.val_df.index)} samples")
        else:
            self.val_df = self.data.sample(n=size, random_state=self.seed).copy()
        del self.data

        self.val_df = self.feature_split_filter(self.val_df)
        self.create_feature_dataloaders(self.val_df, False, batch_size)

    # training and evaluation dataloaders formation
    def create_feature_dataloaders(self, df, train, batch_size, shuffle=False, test_ratio=None):
        if train:
            train_index, test_index = train_test_split(df.index, test_size=test_ratio, random_state=self.seed)
            train_index, test_index = list(train_index), list(test_index)
            train_size, test_size = len(train_index), len(test_index)
            train_inputs, train_masks = np.empty((train_size, 0)), np.empty((train_size, 0))

            for feature in self.features:
                input_ids, attention_mask = self.tokenize(df.loc[train_index, feature].tolist())
                train_inputs, train_masks = np.concatenate((train_inputs, input_ids), axis=1), np.concatenate((train_masks, attention_mask), axis=1)

            test_inputs, test_masks = self.tokenize(df.loc[test_index, self.val_feature].tolist())
            if self.scaled:
                train_labels = np.reshape(np.multiply(df.loc[train_index, self.target].to_numpy(), 0.01), (train_size, 2))
                test_labels = np.reshape(np.multiply(df.loc[test_index, self.target].to_numpy(), 0.01), (test_size, 2))
            else:
                train_labels = np.reshape(df.loc[train_index, self.target].to_numpy(), (train_size, 2))
                test_labels = np.reshape(df.loc[test_index, self.target].to_numpy(), (test_size, 2))

            self.train_dataloader = create_dataloader(np.reshape(train_inputs, (train_size, self.n_features, 300)),
                                                      np.reshape(train_masks, (train_size, self.n_features, 300)), train_labels, batch_size, shuffle)
            self.test_dataloader = create_dataloader(test_inputs, test_masks, test_labels, batch_size, shuffle)
            del df
        else:
            labels = df[self.target].to_numpy()
            if self.scaled:
                labels = np.multiply(labels, 0.01)
            val_inputs, val_masks = self.tokenize(df[self.val_feature].tolist())

            self.val_dataloader = create_dataloader(val_inputs, val_masks, labels, batch_size, shuffle)
