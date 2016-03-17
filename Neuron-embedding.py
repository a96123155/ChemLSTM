#!/usr/bin/env python2.7

from __future__ import print_function

import numpy as np
import json
import sys
import theano
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Graph
from keras.layers.core import Dense, TimeDistributedDense, Activation, Dropout, AutoEncoder, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.layers import containers
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.utils import generic_utils

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

batch_size = 20
n_d = 100

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: {} [training reactions json] [chemical embeddings]".format(sys.argv[0]))
        quit(1)

    # Read training data
    training_data_fn = sys.argv[1]
    print("Reading input...", end=" ")
    with open(training_data_fn) as f:
        training_data = json.load(f)
    print("done.")

    # Read embeddings
    embeddings_fn = sys.argv[2]
    print("Reading embeddings...", end=" ")
    with open(embeddings_fn) as f:
        embeddings = pickle.load(f)
    print("done.")

    # Fit the countvectorizer to the corpus
    token_regex = r"(?u)([\(\)\[\]]|\b\w+\b)"
    cv = CountVectorizer(token_pattern=token_regex, min_df=1)
    an = cv.build_analyzer()

    print("Training length: {}, embedding vector width: {}, batch_size: {}".format(len(training_data), n_d, batch_size))

    def word2embedding(w):
        tokens = an(w.strip())
        word_embeddings = np.zeros((batch_size, n_d), dtype=theano.config.floatX)

        # (pre-)pad out tokens
        if len(tokens) < batch_size:
            tokens = [""] * (batch_size - len(tokens)) + tokens

        for i, token in enumerate(tokens[:batch_size]):
            if token in embeddings:
                word_embeddings[i] = embeddings[token]
        return word_embeddings

    # Build the training vectors
    X = np.zeros((len(training_data), batch_size, 2*n_d), dtype=theano.config.floatX)
    Y = np.zeros((len(training_data)), dtype=np.bool)

    np.random.shuffle(training_data)

    for i, (c1, c2, reacts) in enumerate(training_data):
        X[i] = np.concatenate((word2embedding(c1), word2embedding(c2)), axis=1)
        Y[i] = reacts

    # Create training/development split
    split = int(len(training_data) * 0.9)

    X_train, X_dev = X[:split], X[split:]
    Y_train, Y_dev = Y[:split], Y[split:]

    def generate_batches(number):
        count = 0
        while count < X_train.shape[0]:
            yield X_train[count:count+number], Y_train[count:count+number].reshape((-1, 1))
            count += number

    print("Constructing model...", end=" ")

    dropout = 0.5

    model = Graph()
    model = Sequential()

    model.add(LSTM(output_dim=2*n_d, input_shape=(batch_size, 2*n_d), activation="tanh", return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss="binary_crossentropy", optimizer="adam", class_mode="binary")

    print("done.")

    ts = []
    ds = []

    data_file = "performance/data.tsv"

    def score(iteration):
        train_preds = model.predict_classes(X_train)
        dev_preds = model.predict_classes(X_dev)

        train_acc = accuracy_score(train_preds, Y_train)
        dev_acc = accuracy_score(dev_preds, Y_dev)

        print("Train accuracy: {:.4f}".format(train_acc))
        print("Dev accuracy  : {:.4f}".format(dev_acc))

        ts.append(train_acc)
        ds.append(dev_acc)

        with open(data_file, "a+") as f:
            f.write("{}\t{}\t{}\n".format(iteration, train_acc, dev_acc))

    # Train the model
    print("Training...")

    with open(data_file, "w") as f:
        f.write("iteration\ttraining\tdevelopment\n")

    iterations = 200
    nb_epochs = 10
    try:
        for iteration in xrange(iterations):
            print("ITERATION", iteration)
            for epoch in xrange(nb_epochs):
                print("Epoch", epoch)
                progbar = generic_utils.Progbar(len(X_train))
                for x_train, y_train in generate_batches(128):
                    loss = model.train_on_batch(x_train, y_train)
                    progbar.add(x_train.shape[0], values=[("train loss", loss[0])])

            # Score the model
            score(iteration)

    except KeyboardInterrupt:
        pass

    print("\ndone.")

    score(iteration)

    print("Previous training scores:", ts)
    print("Previous development scores:", ds)
