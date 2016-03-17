#!/usr/bin/env python2.7

from __future__ import print_function

import os
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
minibatch_size = 64

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} [input file]".format(sys.argv[0]))
        quit(1)

    print("Reading input...", end=" ")
    with open(sys.argv[1]) as f:
        data = json.load(f)

    print("done.")

    corpus = set()
    for c1, c2, reacts in data:
        corpus.add(c1)
        corpus.add(c2)

    # Fit the countvectorizer to the corpus
    token_regex = r"(?u)([\(\)\[\]]|\b\w+\b)"
    cv = CountVectorizer(token_pattern=token_regex, min_df=1)
    an = cv.build_analyzer()
    cv.fit(corpus)
    vocabulary_size = len(cv.vocabulary_) + 1

    vocab_file = "neural_vocabulary.p"

    with open(vocab_file, "w") as f:
        pickle.dump(cv.vocabulary_, f)
        print("Wrote vocabulary to:", vocab_file)

    print("Input length: {}, feature vector width: {}, batch_size: {}".format(len(corpus), vocabulary_size, batch_size))

    def word2seq(w):
        tokens = an(w.strip())
        token_ids = np.zeros(len(tokens))
        for i, token in enumerate(tokens):
            tid = cv.vocabulary_.get(token)
            if tid is not None:
                token_ids[i] = tid + 1
        return pad_sequences([token_ids], maxlen=batch_size)[0]

    def word2onehot(w):
        seq = word2seq(w)
        onehots = np.zeros((batch_size, vocabulary_size), dtype=np.bool)
        for i, tid in enumerate(seq):
            onehots[i, tid] = 1
        return onehots

    # Build the training vectors
    X = np.zeros((len(corpus), batch_size), dtype=np.int32)
    Y = np.zeros((len(corpus), batch_size, vocabulary_size), dtype=np.bool)

    for i, chemical in enumerate(corpus):
        X[i] = word2seq(chemical)
        Y[i] = word2onehot(chemical)

    # Create training/development split
    split = int(len(corpus) * 0.9)

    X_train, X_dev = X[:split], X[split:]
    Y_train, Y_dev = Y[:split], Y[split:]

    print(X_train.shape)
    print(Y_train.shape)

    def generate_batches(number):
        count = 0
        while count < X_train.shape[0]:
            yield X_train[count:count+number], Y_train[count:count+number]
            count += number

    def generate_dev_batches(number):
        count = 0
        while count < X_dev.shape[0]:
            yield X_dev[count:count+number], Y_dev[count:count+number]
            count += number

    print("Constructing model...", end=" ")

    # embedding layer
    encoder = containers.Sequential([
        Embedding(input_dim=vocabulary_size, output_dim=n_d, input_length=batch_size, init="glorot_normal"),
        Dropout(0.5),
        LSTM(output_dim=n_d, input_shape=(batch_size, n_d), activation="tanh", return_sequences=True)
    ])

    # rnn layer and output layers
    decoder = containers.Sequential([
        Dropout(0.5, input_shape=(batch_size, n_d)),
        TimeDistributedDense(output_dim=vocabulary_size, input_dim=n_d, input_length=batch_size),
        Activation("softmax")
    ])

    model = Sequential()
    model.add(AutoEncoder(encoder, decoder, output_reconstruction=False))

    # Compile the model
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    json_string = model.to_json()
    with open("autoencoder_architecture.json", "w") as f:
        f.write(json_string)

    print("done.")

    ts = []
    ds = []

    data_file = "autoencoder_accuracy.tsv"

    def preds2seq(predictions):
        result = np.zeros((len(predictions), batch_size))
        for i, prediction in enumerate(predictions):
            seq = [np.argmax(pred) for pred in prediction]
            result[i] = seq
        return result

    def score(iteration):
        train_correct = 0
        dev_correct = 0

        for x_train, y_train in generate_batches(minibatch_size):
            train_preds = model.predict_on_batch(x_train)[0]
            train_preds = preds2seq(train_preds)
            train_correct += sum([1 if p == t else 0 for p, t in zip(train_preds.flatten(), y_train.flatten())])

        for x_dev, y_dev in generate_dev_batches(minibatch_size):
            dev_preds = model.predict_on_batch(x_dev)[0]
            dev_preds = preds2seq(dev_preds)
            dev_correct += sum([1 if p == t else 0 for p, t in zip(dev_preds.flatten(), y_dev.flatten())])

        train_acc = float(train_correct) / (Y_train.shape[0] * batch_size)
        dev_acc = float(dev_correct) / (Y_dev.shape[0] * batch_size)

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

    def preserve_weights(x, name_prefix, iteration):
        x.save_weights("{}_weights_{}.h5".format(name_prefix, iteration), overwrite=True)
        old_weights = "{}_weights_{}.h5".format(name_prefix, iteration - 2)
        if os.path.exists(old_weights):
            os.remove(old_weights)
            print("Removed old weights file:", old_weights)

    iterations = 100
    nb_epochs = 1
    try:
        for iteration in xrange(iterations):
            print("ITERATION", iteration)
            progbar = generic_utils.Progbar(len(X_train))
            for epoch in xrange(nb_epochs):
                print("Epoch", epoch)
                for x_train, y_train in generate_batches(minibatch_size):
                    loss = model.train_on_batch(x_train, y_train)
                    progbar.add(x_train.shape[0], values=[("train loss", loss[0])])

            preserve_weights(model, "autoencoder_3", iteration)

            # Score the model
            score(iteration)

    except KeyboardInterrupt:
        pass

    print("\ndone.")

    # Score the model
    score(iteration)

    print("Previous training scores:", ts)
    print("Previous development scores:", ds)

    # Derive embeddings
    print("Deriving embeddings...")
    embeddings = {}
    for word, idx in cv.vocabulary_.iteritems():
        embeddings[word] = np.asarray(embedding_layer.W.eval()[idx])

    with open("embeddings_neural.p", "w") as f:
        pickle.dump(embeddings, f)
