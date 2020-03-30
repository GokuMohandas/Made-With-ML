import os
import collections
import json
import math
import numpy as np
import pandas as pd
import random
import re
from sklearn.model_selection import train_test_split
import ssl
import urllib

import torch
import torch.nn as nn


# Read data from URLs
ssl._create_default_https_context = ssl._create_unverified_context


def load_data(url,  data_size):
    """Load dataset from URL."""
    df = pd.read_csv(url)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle

    # Reduce dataset
    # You should always overfit your models on a small
    # dataset first so you can catch errors quickly.
    df = df[:int(len(df)*data_size)]

    X = df['title'].values
    y = df['category'].values
    return X, y


def preprocess_texts(texts, lower=True, filters=r"[!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~]"):
    preprocessed_texts = []
    for text in texts:
        if lower:
            text = ' '.join(word.lower() for word in text.split(" "))
        text = re.sub(r"([.,!?])", r" \1 ", text)
        text = re.sub(filters, r"", text)
        text = re.sub(' +', ' ', text)  # remove multiple spaces
        text = text.strip()
        preprocessed_texts.append(text)
    return preprocessed_texts


def train_val_test_split(X, y, val_size, test_size, shuffle):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, shuffle=shuffle)
    return X_train, X_val, X_test, y_train, y_val, y_test


class Tokenizer(object):
    def __init__(self, char_level, pad_token='<PAD>', oov_token='<UNK>',
                 token_to_index={'<PAD>': 0, '<UNK>': 1}):
        self.char_level = char_level
        self.separator = '' if self.char_level else ' '
        self.oov_token = oov_token
        self.token_to_index = token_to_index
        self.index_to_token = {v: k for k, v in self.token_to_index.items()}

    def __len__(self):
        return len(self.token_to_index)

    def __str__(self):
        return f"<Tokenizer(num_tokens={len(self)})>"

    def fit_on_texts(self, texts):
        for text in texts:
            for token in text.split(self.separator):
                if token not in self.token_to_index:
                    index = len(self)
                    self.token_to_index[token] = index
                    self.index_to_token[index] = token
        return self

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequence = []
            for token in text.split(self.separator):
                sequence.append(self.token_to_index.get(
                    token, self.token_to_index[self.oov_token]))
            sequences.append(sequence)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for sequence in sequences:
            text = []
            for index in sequence:
                text.append(self.index_to_token.get(index, self.oov_token))
            texts.append(self.separator.join([token for token in text]))
        return texts

    def save(self, fp):
        with open(fp, 'w') as fp:
            contents = {
                'char_level': self.char_level,
                'oov_token': self.oov_token,
                'token_to_index': self.token_to_index
            }
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, 'r') as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


class LabelEncoder(object):
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())

    def __len__(self):
        return len(self.class_to_index)

    def __str__(self):
        return f"<LabelEncoder(num_classes={len(self)})>"

    def fit(self, y_train):
        for i, class_ in enumerate(np.unique(y_train)):
            self.class_to_index[class_] = i
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.classes = list(self.class_to_index.keys())
        return self

    def transform(self, y):
        return np.array([self.class_to_index[class_] for class_ in y])

    def decode(self, index):
        return self.index_to_class.get(index, None)

    def save(self, fp):
        with open(fp, 'w') as fp:
            contents = {
                'class_to_index': self.class_to_index
            }
            json.dump(contents, fp, indent=4, sort_keys=False)

    @classmethod
    def load(cls, fp):
        with open(fp, 'r') as fp:
            kwargs = json.load(fp=fp)
        return cls(**kwargs)


def pad_sequences(sequences):
    max_seq_len = max(len(sequence) for sequence in sequences)
    padded_sequences = np.zeros((len(sequences), max_seq_len))
    for i, sequence in enumerate(sequences):
        padded_sequences[i][:len(sequence)] = sequence
    return padded_sequences


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, batch_size, max_filter_size):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.max_filter_size = max_filter_size

    def __len__(self):
        return len(self.y)

    def __str__(self):
        return f"<Dataset(N={len(self)}, " \
               f"batch_size={self.batch_size}, " \
               f"num_batches={self.get_num_batches()})>"

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]
        return X, y

    def get_num_batches(self):
        return math.ceil(len(self)/self.batch_size)

    def collate_fn(self, batch):
        """Processing on a batch."""
        # Get inputs
        X = np.array(batch)[:, 0]
        y = np.array(batch)[:, 1]

        # Pad inputs
        X = pad_sequences(sequences=X)

        return X, y

    def generate_batches(self, shuffle=False, drop_last=False):
        dataloader = torch.utils.data.DataLoader(
            dataset=self, batch_size=self.batch_size, collate_fn=self.collate_fn,
            shuffle=shuffle, drop_last=drop_last, pin_memory=True)
        for (X, y) in dataloader:
            X = torch.LongTensor(X.astype(np.int32))
            y = torch.LongTensor(y.astype(np.int32))
            yield X, y
