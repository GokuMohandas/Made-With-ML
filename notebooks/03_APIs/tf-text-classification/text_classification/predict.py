import os
import sys
sys.path.append(".")
from argparse import ArgumentParser
from argparse import Namespace
import collections
import json
import numpy as np

from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import tokenizer_from_json

import config
import utilities
from text_classification import data
from text_classification import models


def get_probability_distribution(y_prob, classes):
    results = {}
    for i, class_ in enumerate(classes):
        results[class_] = np.float64(y_prob[i])
    sorted_results = {k: v for k, v in sorted(
        results.items(), key=lambda item: item[1], reverse=True)}
    return sorted_results


def get_top_n_grams(tokens, conv_outputs, filter_sizes):
    # Process conv outputs for each unique filter size
    n_grams = {}
    for i, filter_size in enumerate(filter_sizes):

        # Identify most important n-gram for each filter's output
        popular_indices = collections.Counter(
            np.argmax(conv_outputs[i][0], axis=0))

        # Get corresponding text
        start = popular_indices.most_common(1)[-1][0]
        n_gram = " ".join([token for token in tokens[start:start+filter_size]])
        n_grams[filter_size] = n_gram

    return n_grams


def predict(experiment_id, text):
    """Predict the class for a text using
    a trained model from an experiment."""
    # Get experiment config
    experiment_dir = os.path.join(config.EXPERIMENTS_DIR, experiment_id)
    experiment_config = utilities.load_json(
        os.path.join(experiment_dir, 'config.json'))
    args = Namespace(**experiment_config)

    # Tokenizers
    texts = [text]
    with open(os.path.join(experiment_dir, 'X_tokenizer.json'), 'r') as fp:
        X_tokenizer = tokenizer_from_json(json.load(fp))
    y_tokenizer = LabelEncoder()
    y_tokenizer.classes_ = np.load(os.path.join(
        experiment_dir, 'y_tokenizer.npy'), allow_pickle=True)

    # Create dataset generator
    X_infer = np.array(X_tokenizer.texts_to_sequences(texts))
    preprocessed_texts = X_tokenizer.sequences_to_texts(X_infer),
    y_filler = np.array([0]*len(X_infer))
    inference_generator = data.DataGenerator(
        X=X_infer, y=y_filler, batch_size=args.batch_size,
        max_filter_size=max(args.filter_sizes))

    # Load model
    model = models.TextCNN(
        embedding_dim=args.embedding_dim,
        vocab_size=len(X_tokenizer.word_index)+1,
        num_filters=args.num_filters, filter_sizes=args.filter_sizes,
        hidden_dim=args.hidden_dim, dropout_p=args.dropout_p,
        num_classes=len(y_tokenizer.classes_))
    model.summary(input_shape=(10,))  # build it
    model_path = os.path.join(experiment_dir, 'model/cp.ckpt')
    model.load_weights(model_path)

    # Conv output model
    conv_outputs_model = models.ConvOutputsModel(
        vocab_size=len(X_tokenizer.word_index)+1,
        embedding_dim=args.embedding_dim, filter_sizes=args.filter_sizes,
        num_filters=args.num_filters)
    conv_outputs_model.summary(input_shape=(10,))  # build it

    # Set weights
    conv_outputs_model.layers[0].set_weights(model.layers[0].get_weights())
    conv_layer_start_num = 1
    for layer_num in range(conv_layer_start_num, conv_layer_start_num + len(args.filter_sizes)):
        conv_outputs_model.layers[layer_num].set_weights(
            model.layers[layer_num].get_weights())

    # Predict
    results = []
    y_prob = model.predict(x=inference_generator, verbose=1)
    conv_outputs = conv_outputs_model.predict(x=inference_generator, verbose=1)
    for index in range(len(X_infer)):
        results.append({
            'raw_input': texts[index],
            'preprocessed_input': preprocessed_texts[index][0],
            'probabilities': get_probability_distribution(y_prob[index], y_tokenizer.classes_),
            'top_n_grams': get_top_n_grams(tokens=preprocessed_texts[index][0].split(' '),
                                           conv_outputs=conv_outputs,
                                           filter_sizes=args.filter_sizes)})

    return results



if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--experiment-id', type=str,
                        default="latest", help="name of the model to load")
    parser.add_argument('--text', type=str,
                        required=True, help="text to predict")
    args = parser.parse_args()

    # Load model config
    if args.experiment_id == 'latest':
        args.experiment_id = max(os.listdir(config.EXPERIMENTS_DIR))
    experiment_dir = os.path.join(config.EXPERIMENTS_DIR, args.experiment_id)
    experiment_config = utilities.load_json(
        os.path.join(experiment_dir, 'config.json'))
    args = Namespace(**{**args.__dict__, **Namespace(**experiment_config).__dict__})
    config.logger.info(f"→ Using {args.experiment_id}")

    # Tokenizers
    texts = [args.text]
    with open(os.path.join(experiment_dir, 'X_tokenizer.json'), 'r') as fp:
        X_tokenizer = tokenizer_from_json(json.load(fp))
    y_tokenizer = LabelEncoder()
    y_tokenizer.classes_ = np.load(os.path.join(
        experiment_dir, 'y_tokenizer.npy'), allow_pickle=True)

    # Create dataset generator
    X_infer = np.array(X_tokenizer.texts_to_sequences(texts))
    preprocessed_texts = X_tokenizer.sequences_to_texts(X_infer),
    y_filler = np.array([0]*len(X_infer))
    inference_generator = data.DataGenerator(
        X=X_infer, y=y_filler, batch_size=args.batch_size,
        max_filter_size=max(args.filter_sizes))

    # Load model
    model = models.TextCNN(
        embedding_dim=args.embedding_dim,
        vocab_size=len(X_tokenizer.word_index)+1,
        num_filters=args.num_filters, filter_sizes=args.filter_sizes,
        hidden_dim=args.hidden_dim, dropout_p=args.dropout_p,
        num_classes=len(y_tokenizer.classes_))
    model.summary(input_shape=(10,))  # build it
    model_path = os.path.join(experiment_dir, 'model/cp.ckpt')
    model.load_weights(model_path)

    # Conv output model
    conv_outputs_model = models.ConvOutputsModel(
        vocab_size=len(X_tokenizer.word_index)+1,
        embedding_dim=args.embedding_dim, filter_sizes=args.filter_sizes,
        num_filters=args.num_filters)
    conv_outputs_model.summary(input_shape=(10,))  # build it

    # Set weights
    conv_outputs_model.layers[0].set_weights(model.layers[0].get_weights())
    conv_layer_start_num = 1
    for layer_num in range(conv_layer_start_num, conv_layer_start_num + len(args.filter_sizes)):
        conv_outputs_model.layers[layer_num].set_weights(
            model.layers[layer_num].get_weights())

    # Predict
    results = []
    y_prob = model.predict(x=inference_generator, verbose=1)
    conv_outputs = conv_outputs_model.predict(x=inference_generator, verbose=1)
    for index in range(len(X_infer)):
        results.append({
            'raw_input': texts[index],
            'preprocessed_input': preprocessed_texts[index][0],
            'probabilities': get_probability_distribution(y_prob[index], y_tokenizer.classes_),
            'top_n_grams': get_top_n_grams(tokens=preprocessed_texts[index][0].split(' '),
                                           conv_outputs=conv_outputs,
                                           filter_sizes=args.filter_sizes)})
    config.logger.info(json.dumps(results, indent=4, sort_keys=False))
