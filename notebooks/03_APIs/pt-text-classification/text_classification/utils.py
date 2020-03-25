import os
import sys
sys.path.append(".")
from io import BytesIO
import numpy as np
import ssl
from urllib.request import urlopen
from zipfile import ZipFile

import config
import utilities

# Read data from URLs
ssl._create_default_https_context = ssl._create_unverified_context


def load_glove_embeddings(embeddings_file):
    """Load embeddings from a file."""
    embeddings = {}
    with open(embeddings_file, "r") as fp:
        for index, line in enumerate(fp):
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding
    return embeddings


def make_embeddings_matrix(embeddings, token_to_index, embedding_dim):
    """Create embeddings matrix to use in Embedding layer."""
    embedding_matrix = np.zeros((len(token_to_index)+1, embedding_dim))
    for word, i in token_to_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


if __name__ == '__main__':
    # Unzip and write embeddings (may take ~3-5 minutes)
    resp = urlopen('http://nlp.stanford.edu/data/glove.6B.zip')
    embeddings_dir = os.path.join(config.BASE_DIR, 'embeddings')
    utilities.create_dirs(embeddings_dir)
    with ZipFile(BytesIO(resp.read()), 'r') as zr:
        zr.extractall(embeddings_dir)