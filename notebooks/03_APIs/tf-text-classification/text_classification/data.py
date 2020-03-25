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

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence


# Read data from URLs
ssl._create_default_https_context = ssl._create_unverified_context


def load_data(url, overfit):
    """Load dataset from URL."""
    df = pd.read_csv(url)
    df = df.sample(frac=1).reset_index(drop=True) # shuffle

    # Reduce dataset
    # You should always overfit your models on a small
    # dataset first so you can catch errors quickly.
    if overfit:
        df = df[:int(len(df)*0.05)]

    X = df['title'].values
    y = df['category'].values
    return X, y


def train_val_test_split(X, y, val_size, test_size, shuffle):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, shuffle=shuffle)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, stratify=y_train, shuffle=shuffle)
    return X_train, X_val, X_test, y_train, y_val, y_test


class DataGenerator(Sequence):
    """Custom data loader."""

    def __init__(self, X, y, batch_size, max_filter_size, shuffle=False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.max_filter_size = max_filter_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """# of batches."""
        return math.ceil(len(self.X) / self.batch_size)

    def __str__(self):
        return (f"<DataGenerator("
                f"batch_size={self.batch_size}, "
                f"batches={len(self)}, "
                f"shuffle={self.shuffle})>")

    def __getitem__(self, index):
        """Generate a batch."""
        # Gather indices for this batch
        batch_indices = self.epoch_indices[
            index * self.batch_size:(index+1)*self.batch_size]

        # Generate batch data
        X, y = self.create_batch(batch_indices=batch_indices)

        return X, y

    def on_epoch_end(self):
        """Create indices after each epoch."""
        self.epoch_indices = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.epoch_indices)

    def create_batch(self, batch_indices):
        """Generate batch from indices."""
        # Get batch data
        X = self.X[batch_indices]
        y = self.y[batch_indices]

        # Pad batch
        max_seq_len = max(self.max_filter_size, max([len(x) for x in X]))
        X = pad_sequences(X, padding="post", maxlen=max_seq_len)

        return X, y
