import sys
sys.path.append(".")
import numpy as np
import pytest


from text_classification import data


@pytest.mark.parametrize('texts, preprocessed_texts', [
    ('Hello', 'hello'),
    ('HELLO', 'hello'),
    ('Hello, world!', 'hello world'),
    ('Hello, world!', 'hello world')
])
def test_preprocess_texts(texts, preprocessed_texts):
    assert data.preprocess_texts(texts=[texts]) == [preprocessed_texts]


@pytest.mark.parametrize('sequences, padded_sequences', [
    ([[1, 2, 3]], [[1, 2, 3]]),
    ([[1, 2], [1, 2, 3, 4]], [[1, 2, 0, 0], [1, 2, 3, 4]])
])
def test_pad_sequences(sequences, padded_sequences):
    assert data.pad_sequences(sequences=sequences).tolist() == padded_sequences