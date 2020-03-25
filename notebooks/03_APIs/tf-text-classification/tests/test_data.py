import sys
sys.path.append(".")
import numpy as np
import pytest

from tensorflow.keras.preprocessing.sequence import pad_sequences


@pytest.mark.parametrize('sequences, padded_sequences', [
    ([[1, 2, 3]], [[1, 2, 3]]),
    ([[1, 2], [1, 2, 3, 4]], [[1, 2, 0, 0], [1, 2, 3, 4]])
])
def test_pad_sequences(sequences, padded_sequences):
    max_seq_len = max([len(sequence) for sequence in sequences])
    assert pad_sequences(sequences, padding="post",
                         maxlen=max_seq_len).tolist() == padded_sequences
