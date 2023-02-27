import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader

import string
import re
import os
import datetime

from sklearn.model_selection import train_test_split

import GPUtil
import psutil

# dataset wrapper


def load_jsonl(filename):
    filename = f"datasets/{filename}"
    print(f"DATASET\tLOAD: {psutil.virtual_memory().percent}%\tLoading dataset from {filename}")
    data = pd.read_json(path_or_buf=filename, lines=True)
    print(f"DATASET\tLOAD: {psutil.virtual_memory().percent}%\tDataset of {len(data.index)} samples and {len(data.user.unique())} users is loaded")
    return data


def save_df(df, filename, prefix=None):
    if prefix is None:
        prefix = "td-"

    size = len(df.index)
    save_filename = f"datasets/{prefix}{size}-{filename}"

    with open(save_filename, "w") as f:
        df.to_json(f, orient='records', lines=True)
    print(f"DATASET\tSAVE\tTwitter Dataset of {size} samples is written to file: {save_filename}")


def combine_datasets(file_list, filename, prefix=None):
    if prefix is None:
        prefix = "td-"

    df  = load_jsonl(file_list[0])
    for file in file_list[1:]:
        data = load_jsonl(file)
        df = df.append(data, ignore_index=True)

    save_df(df, filename, prefix)


def filter_bots(data, min_total=10, max_day=50):
    print(f"DATASET\tFiltering dataset of {len(data['user'].unique())} users from bots posting more than {max_day} tweets per day")
    if "time" in data.columns:
        data['date'] = pd.to_datetime(data['time'], utc=False).dt.date
        user_tweets_per_day = data.groupby(['date'])['user'].value_counts()
        user_tweets = user_tweets_per_day[user_tweets_per_day < max_day].droplevel(0).groupby(["user"]).sum()
    else:
        user_tweets = data['user'].value_counts()

    # data["time"] = data['time'].apply(lambda x: datetime.datetime.combine(x, datetime.time.min).timestamp())
    data['date'] = data['time'].apply(lambda x: datetime.datetime.strptime(x, '%a %b %d %H:%M:%S %z %Y').timestamp())
    # data.drop("date", axis=1, inplace=True)
    data["date"] = data["date"].astype(float)

    user_list = user_tweets[user_tweets > min_total].index.tolist()
    data = data[data['user'].isin(user_list)]

    print(f"DATASET\tSize of the filtered dataset with {len(data['user'].unique())} users: {len(data.index)} samples")
    return data


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
def crop_dataset(data, size, seed, by_user=False, ref_file=None, save=False):
    print(f"DATASET\tCropping dataset of {len(data.index)} in {size} samples with seed {seed}{' including uniqie users' if by_user else ''}")
    if by_user:
        if ref_file:
            df = load_jsonl(ref_file)
            crop_data = df.sample(n=size, random_state=seed)
        else:
            crop_data = data.sample(n=size, random_state=seed)
            data = data.drop(crop_data.index, axis=0)

        crop_users = crop_data["user"].unique()

        print(f"DATASET\tUnique users to crop: {len(crop_users)}")
        data = data[-data["user"].isin(crop_users)]
        print(f"DATASET\tUnique users left: {len(data['user'].unique())}")
    else:
        crop_data = data.sample(n=size, random_state=seed)
        data = data.drop(crop_data.index, axis=0)

    if save and ref_file:
        save_df(crop_data, ref_file, "train-")

    print(f"DATASET\tReduced dataset lenght is {len(data.index)}")
    return data


# sample users (bots filtering) for evaluation
def sample_users(data, users_n, seed=42):
    user_tweets = data['user'].value_counts()
    user_list = user_tweets.sample(n=users_n, random_state=seed).index.tolist()
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

        #self.data = filter_bots(self.data)
        #save_df(self.data, self.filename, "filtered-")

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

    def form_validation(self, batch_size, size, by_user=False, skip_size=0, ref_train_file=None):
        if skip_size > 0:
            self.data = crop_dataset(self.data, skip_size, self.seed, by_user, ref_train_file, True)

#        self.data = filter_bots(self.data)
        save_df(self.data, self.filename, "test-")

        print(f"DATASET\tForming validation dataset of {size} {'users' if by_user else 'samples'} with batch size {batch_size} for {self.val_feature} text feature")
        if by_user:
            self.val_df = sample_users(self.data, size, seed=self.seed).copy()
            self.features += ["USER-ONLY"]
        else:
            self.val_df = self.data.sample(n=size, random_state=self.seed).copy()
        del self.data

        print(f"DATASET\tSize of the validation dataset with {len(self.val_df['user'].unique())} users: {len(self.val_df.index)} samples")

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
