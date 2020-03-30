import os
import sys
sys.path.append(".")
from argparse import ArgumentParser
from datetime import datetime
import io
import itertools
import json
import matplotlib.pyplot as plt
import numpy as np
import petname
import random
import time
from tqdm import tqdm

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorboard.plugins.hparams import api as hp

from text_classification import config
from text_classification import data
from text_classification import models
from text_classification import utils


def plot_confusion_matrix(y_pred, y_target, classes, fp, cmap=plt.cm.Blues):
    """Plot a confusion matrix using ground truth and predictions."""
    # Confusion matrix
    cm = confusion_matrix(y_target, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    #  Figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    fig.colorbar(cax)

    # Axis
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    ax.set_xticklabels([''] + classes)
    ax.set_yticklabels([''] + classes)
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    # Values
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:d} ({cm_norm[i, j]*100:.1f}%)",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Save
    plt.rcParams["figure.figsize"] = (7, 7)
    plt.savefig(fp)


def get_performance(y_pred, y_target, classes):
    """Per-class performance metrics. """
    performance = {'overall': {}, 'class': {}}
    metrics = precision_recall_fscore_support(y_target, y_pred)

    # Overall performance
    performance['overall']['precision'] = np.mean(metrics[0])
    performance['overall']['recall'] = np.mean(metrics[1])
    performance['overall']['f1'] = np.mean(metrics[2])
    performance['overall']['num_samples'] = np.float64(np.sum(metrics[3]))

    # Per-class performance
    for i in range(len(classes)):
        performance['class'][classes[i]] = {
            "precision": metrics[0][i],
            "recall": metrics[1][i],
            "f1": metrics[2][i],
            "num_samples": np.float64(metrics[3][i])
        }

    return performance


if __name__ == '__main__':
    # Arguments
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true',
                        default=False, help="Use GPUs")
    parser.add_argument('--seed', type=int, default=1234,
                        help="initialization seed")
    parser.add_argument('--shuffle', action='store_true',
                        default=False, help="shuffle your data")
    parser.add_argument('--data-url', type=str,
                        required=True, help="URL of data file")
    parser.add_argument('--lower', action='store_true',
                        default=False, help="lowercase all text")
    parser.add_argument('--char-level', action='store_true',
                        default=False, help="split text to character level")
    parser.add_argument('--filters', type=str,
                        default=r"[!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~]",
                        help="text preprocessing filters")
    parser.add_argument('--data-size', type=float,
                        default=1.0, help="proportion of data to use")
    parser.add_argument('--train-size', type=float,
                        default=0.7, help="train data proportion")
    parser.add_argument('--val-size', type=float,
                        default=0.15, help="val data proportion")
    parser.add_argument('--test-size', type=float,
                        default=0.15, help="test data proportion")
    parser.add_argument('--num-epochs', type=int,
                        default=10, help="# of epochs to train")
    parser.add_argument('--batch-size', type=int, default=64,
                        help="# of samples per batch")
    parser.add_argument('--embedding-dim', type=int,
                        default=100,
                        help="dimension of embeddings (50, 100, 200, 300 if using GloVe)")
    parser.add_argument('--use-glove', action='store_true',
                        default=False, help="Use pretrained GloVe embeddings")
    parser.add_argument('--freeze-embeddings', action='store_true',
                        default=False, help="Freeze embeddings during training")
    parser.add_argument('--filter-sizes', nargs='+',
                        default=[2, 3, 4], type=int, help="cnn filter sizes")
    parser.add_argument('--num-filters', type=int, default=50,
                        help="# of filters per cnn filter size")
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help="# of hidden units in fc dense layers")
    parser.add_argument('--dropout-p', type=float, default=0.1,
                        help="dropout proportion in fc dense layers")
    parser.add_argument('--learning-rate', type=float,
                        default=1e-4, help="initial learning rate")
    parser.add_argument('--patience', type=int, default=3,
                        help="# of epochs of continued performance regression")
    args = parser.parse_args()
    config.logger.info(json.dumps(args.__dict__, indent=4))

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load data
    X, y = data.load_data(url=args.data_url, data_size=args.data_size)
    config.logger.info(
        "→ Raw data:\n"
        f"  {X[0]} → {y[0]}")

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = data.train_val_test_split(
        X=X, y=y, val_size=args.val_size, test_size=args.test_size, shuffle=args.shuffle)
    config.logger.info(
        "→ Data splits:\n"
        f"  X_train: {len(X_train)}, y_train: {len(y_train)}\n"
        f"  X_val: {len(X_val)}, y_val: {len(y_val)}\n"
        f"  X_test: {len(X_test)}, y_test: {len(y_test)}")

    # Tokenizer
    X_tokenizer = Tokenizer(
        filters=args.filters, lower=args.lower, char_level=args.char_level, oov_token='<UNK>')
    X_tokenizer.fit_on_texts(X_train)
    vocab_size = len(X_tokenizer.word_index) + 1 # +1 for padding token
    config.logger.info(f"→ vocab_size: {vocab_size}")

    # Convert texts to sequences of indices
    original_text = X_train[0]
    X_train = np.array(X_tokenizer.texts_to_sequences(X_train))
    X_val = np.array(X_tokenizer.texts_to_sequences(X_val))
    X_test = np.array(X_tokenizer.texts_to_sequences(X_test))
    preprocessed_text = X_tokenizer.sequences_to_texts([X_train[0]])[0]
    config.logger.info(
        "→ Text to indices:\n"
        f"  (raw) → {original_text}\n"
        f"  (preprocessed) → {preprocessed_text}\n"
        f"  (tokenized) → {X_train[0]}")

    # Label encoder
    y_tokenizer = LabelEncoder()
    y_tokenizer = y_tokenizer.fit(y_train)
    classes = y_tokenizer.classes_
    config.logger.info(
        "→ classes:\n"
        f"  {classes}")

    # Convert labels to tokens
    class_ = y_train[0]
    y_train = y_tokenizer.transform(y_train)
    y_val = y_tokenizer.transform(y_val)
    y_test = y_tokenizer.transform(y_test)
    config.logger.info(
        "→ Labels to indices:\n"
        f"  {class_} → {y_train[0]}")

    # Class weights
    counts = np.bincount(y_train)
    class_weights = {i: 1.0/count for i, count in enumerate(counts)}
    config.logger.info(
        "→ class counts:\n"
        f"  {counts}\n"
        "→ class weights:\n"
        f"  {class_weights}")

    # Dataset generators
    training_generator = data.DataGenerator(
        X=X_train, y=y_train, batch_size=args.batch_size,
        max_filter_size=max(args.filter_sizes))
    validation_generator = data.DataGenerator(
        X=X_val, y=y_val, batch_size=args.batch_size,
        max_filter_size=max(args.filter_sizes))
    testing_generator = data.DataGenerator(
        X=X_test, y=y_test, batch_size=args.batch_size,
        max_filter_size=max(args.filter_sizes))
    batch_X, batch_y = training_generator[0] # sample
    config.logger.info(
        "→ Dataset generators:\n"
        f"  (training_generator) → {training_generator}\n"
        f"  (validation_generator) → {validation_generator}\n"
        f"  (testing_generator) → {testing_generator}\n"
        "→ Sample batch:\n"
        f"  X: {batch_X.shape}\n"
        f"  y: {batch_y.shape}\n")

    # Load embeddings
    embedding_matrix = None
    if args.use_glove:
        if args.embedding_dim not in (50, 100, 200, 300):
            raise Exception(
                "Embedding dim must be in (50, 100, 200, 300) is using GloVe.")
        embeddings_file = os.path.join(
            config.EMBEDDINGS_DIR, f'glove.6B.{args.embedding_dim}d.txt')
        glove_embeddings = utils.load_glove_embeddings(
            embeddings_file=embeddings_file)
        embedding_matrix = utils.make_embeddings_matrix(
            embeddings=glove_embeddings, token_to_index=X_tokenizer.word_index,
            embedding_dim=args.embedding_dim)
        config.logger.info(
            "→ GloVe Embeddings:\n"
            f"{embedding_matrix.shape}")

    # Initialize model
    model = models.TextCNN(
        vocab_size=vocab_size, embedding_dim=args.embedding_dim,
        filter_sizes=args.filter_sizes, num_filters=args.num_filters,
        hidden_dim=args.hidden_dim, dropout_p=args.dropout_p,
        num_classes=len(y_tokenizer.classes_),
        freeze_embeddings=args.freeze_embeddings)
    model.summary(input_shape=(10,)) # build it

    # Set GloVe embeddings
    if args.use_glove:
        model.layers[0].set_weights([embedding_matrix])

    # Model dir
    experiment_id = f'{int(time.time())}-{petname.Generate(2)}'
    experiment_dir = os.path.join(config.EXPERIMENTS_DIR, experiment_id)
    utils.create_dirs(dirpath=experiment_dir)
    model_path = os.path.join(experiment_dir, 'model/cp.ckpt')

    # Callbacks
    tb_log_dir = f'{experiment_dir}/tensorboard/'
    callbacks = [EarlyStopping(monitor='val_loss', patience=args.patience, verbose=1, mode='min'),
                 ModelCheckpoint(filepath=model_path, monitor='val_loss', mode='min',
                                 verbose=0, save_best_only=True, save_weights_only=True),
                 ReduceLROnPlateau(patience=1, factor=0.1, verbose=0),
                 TensorBoard(log_dir=tb_log_dir, histogram_freq=1, update_freq='epoch')]

    # Compile
    model.compile(optimizer=Adam(lr=args.learning_rate),
                  loss=SparseCategoricalCrossentropy(),
                  metrics=[SparseCategoricalAccuracy()])

    # Training
    training_history = model.fit(
        x=training_generator, epochs=args.num_epochs, validation_data=validation_generator,
        callbacks=callbacks, shuffle=False, class_weight=class_weights, verbose=1)

    # Write hyperparamters to TensorBoard
    hparams = {
        'embedding_dim': args.embedding_dim,
        'use_glove': args.use_glove,
        'freeze_embeddings': args.freeze_embeddings,
        'num_filters': args.num_filters,
        'hidden_dim': args.hidden_dim,
        'dropout_p': args.dropout_p,
        'learning_rate': args.learning_rate
    }
    with tf.summary.create_file_writer(tb_log_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial

    # Evaluation
    test_history = model.evaluate(x=testing_generator, verbose=1)
    y_pred = model.predict(x=testing_generator, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)
    config.logger.info(
        "→ Test performance:\n"
        f"  test_loss: {test_history[0]:.2f}, test_acc: {test_history[1]:.1f}")

    # Per-class performance analysis
    performance = get_performance(y_pred, y_test, classes)
    plot_confusion_matrix(
        y_pred=y_pred, y_target=y_test, classes=classes,
        fp=os.path.join(experiment_dir, 'confusion_matrix.png'))
    utils.save_dict(performance, filepath=os.path.join(
        experiment_dir, 'performance.json'))
    config.logger.info(json.dumps(performance, indent=4, sort_keys=False))

    # Save
    utils.save_dict(args.__dict__, filepath=os.path.join(
        experiment_dir, 'config.json'))
    with open(os.path.join(experiment_dir, 'X_tokenizer.json'), 'w') as fp:
        json.dump(X_tokenizer.to_json(), fp, indent=4, sort_keys=False)
    np.save(os.path.join(experiment_dir, 'y_tokenizer.npy'), y_tokenizer.classes_)
