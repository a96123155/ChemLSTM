#!/usr/bin/env python2.7

from __future__ import print_function

import h5py
import os
import numpy as np
import json
import sys
import pickle

import theano
import theano.tensor as T

from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Graph
from keras.layers.core import Dense, TimeDistributedDense, Activation, Dropout, AutoEncoder, Flatten, Merge, Lambda
from keras.layers.embeddings import Embedding
from keras.layers import containers
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.utils import generic_utils

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# cap on number of tokens per chemical per training sample
batch_size = 20

# number of hidden units in LSTM (also number of embedding dimensions)
n_d = 100

# number of samples passed to training at a time
minibatch_size = 128

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} [input file]".format(sys.argv[0]))
        quit(1)

    print("Reading input...", end=" ")
    with open(sys.argv[1]) as f:
        data = json.load(f)

    print("done.")

    corpus = []
    for c1, c2, c3 in data:
        corpus.append(c1)
        corpus.append(c2)
        if c3 is not None:
            corpus.append(c3)

    # Fit the countvectorizer to the corpus
    token_regex = r"(?u)([\(\)\[\]]|\b\w+\b)"
    cv = CountVectorizer(token_pattern=token_regex, min_df=1)
    an = cv.build_analyzer()
    cv.fit(corpus)
    vocabulary_size = len(cv.vocabulary_) + 2

    print("Input length: {}, feature vector width: {}, batch_size: {}".format(len(corpus), vocabulary_size, batch_size))

    def word2seq(w):
        tokens = an(w.strip())
        token_ids = np.zeros(len(tokens))
        for i, token in enumerate(tokens):
            tid = cv.vocabulary_.get(token)
            if tid is not None:
                token_ids[i] = tid + 2
        return pad_sequences([token_ids], maxlen=batch_size)[0]

    def seq2onehot(seq):
        one_hots = np.zeros((batch_size, vocabulary_size))
        for i, index in enumerate(seq):
            one_hots[i, index] = 1
        return one_hots

    def word2onehot(w):
        return seq2onehot(word2seq(w))

    # Select shuffled indices of reactions
    indices = np.arange(len(data))
    np.random.shuffle(indices)
    split = int(len(data) * 0.9)
    train_indices = indices[:split]
    dev_indices = indices[split:]

    null_reaction = seq2onehot(np.full(batch_size, 1))

    def generate_batches(number):
        count = 0
        while count < split:
            curr_batch_size = min(number, split - count)
            xa_train = np.zeros((curr_batch_size, batch_size), dtype=np.int32)
            xb_train = np.zeros((curr_batch_size, batch_size), dtype=np.int32)
            y_train = np.zeros((curr_batch_size, batch_size, vocabulary_size), dtype=np.bool)

            for i, ti in enumerate(train_indices[count:count+curr_batch_size]):
                c1, c2, c3 = data[ti]
                xa_train[i] = word2seq(c1)
                xb_train[i] = word2seq(c2)
                if c3 is not None:
                    y_train[i] = word2onehot(c3)
                else:
                    y_train[i] = null_reaction

            yield xa_train, xb_train, y_train
            count += number

    def generate_dev_batches(number):
        count = 0
        while count < (len(data) - split):
            curr_batch_size = min(number, (len(data) - split) - count)
            xa_dev = np.zeros((curr_batch_size, batch_size), dtype=np.int32)
            xb_dev = np.zeros((curr_batch_size, batch_size), dtype=np.int32)
            y_dev = np.zeros((curr_batch_size, batch_size, vocabulary_size), dtype=np.bool)

            for i, di in enumerate(dev_indices[count:count+curr_batch_size]):
                c1, c2, c3 = data[di]
                xa_dev[i] = word2seq(c1)
                xb_dev[i] = word2seq(c2)
                if c3 is not None:
                    y_dev[i] = word2onehot(c3)
                else:
                    y_dev[i] = null_reaction

            yield xa_dev, xb_dev, y_dev
            count += number

    print("Constructing model...", end=" ")

    model = Graph()

    # inputs
    model.add_input(name="chemA", input_shape=(batch_size,), dtype="int")
    model.add_input(name="chemB", input_shape=(batch_size,), dtype="int")

    dropout = 0.75

    # parallel embedding layers
    embedding_layer = Embedding(output_dim=n_d, input_dim=vocabulary_size, init="glorot_normal")
    model.add_shared_node(embedding_layer, "embed", inputs=["chemA", "chemB"], outputs=["embedA", "embedB"])
    model.add_shared_node(Dropout(dropout), "dropout", inputs=["embedA", "embedB"], outputs=["dropoutA", "dropoutB"])
    model.add_shared_node(LSTM(output_dim=n_d, activation="tanh", return_sequences=True), "lstm", inputs=["dropoutA", "dropoutB"], outputs=["lstmA", "lstmB"])
    
    # merge
    model.add_node(Dropout(dropout), name="dropoutM", inputs=["lstmA", "lstmB"], merge_mode="concat")
    model.add_node(LSTM(output_dim=n_d, activation="tanh", return_sequences=True), "lstmM", input="dropoutM")
    model.add_node(Dropout(dropout), name="dropoutM2", input="lstmM")
    model.add_node(TimeDistributedDense(vocabulary_size, activation="tanh"), name="denseM", input="dropoutM2")
    model.add_node(Activation("softmax"), name="softmax", input="denseM")
    model.add_output(name="output", input="softmax")

    model.compile("adam", {"output": "categorical_crossentropy"})

    get_chemical_repr = theano.function([model.inputs["chemA"].input], model.nodes["lstmA"].get_output(train=False))

    print("done.")

    ts = []
    ds = []

    data_file = "performance.tsv"

    def preds2seq(predictions):
        result = np.zeros((len(predictions), batch_size))
        for i, prediction in enumerate(predictions):
            seq = [np.argmax(pred) for pred in prediction]
            result[i] = seq
        return result

    def score(iteration):
        train_correct = 0
        dev_correct = 0

        for xa_train, xb_train, y_train in generate_batches(minibatch_size):
            train_preds = model.predict_on_batch({"chemA": xa_train, "chemB": xb_train})[0]
            train_preds = preds2seq(train_preds)
            train_correct += sum([1 if p == t else 0 for p, t in zip(train_preds.flatten(), y_train.flatten())])

        for xa_dev, xb_dev, y_dev in generate_dev_batches(minibatch_size):
            dev_preds = model.predict_on_batch({"chemA": xa_dev, "chemB": xb_dev})[0]
            dev_preds = preds2seq(dev_preds)
            dev_correct += sum([1 if p == t else 0 for p, t in zip(dev_preds.flatten(), y_dev.flatten())])

        train_acc = float(train_correct) / (split * batch_size)
        dev_acc = float(dev_correct) / ((len(data) - split) * batch_size)

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

    iterations = 1000
    nb_epochs = 1
    try:
        for iteration in xrange(iterations):
            print("ITERATION", iteration)
            progbar = generic_utils.Progbar(split)
            for epoch in xrange(nb_epochs):
                print("Epoch", epoch)
                for xa_train, xb_train, y_train in generate_batches(minibatch_size):
                    loss = model.train_on_batch({"chemA": xa_train, "chemB": xb_train, "output": y_train})
                    progbar.add(xa_train.shape[0], values=[("train loss", loss[0])])

            preserve_weights(model, "neural_with_outputs", iteration)

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

    with open("embeddings_neural_with_outputs.p", "w") as f:
        pickle.dump(embeddings, f)

    # Derive chemical-level embeddings
    corpus_set = set(corpus)
    print("Deriving chemical-level embeddings... ({} of them)".format(len(corpus_set)))
    embeddings_chemical = {}
    for count, chemical in enumerate(corpus_set):
        seq = word2seq(chemical)
        embeddings_chemical[chemical] = np.asarray(get_chemical_repr([seq])[0][-1])
        if count % 1000 == 0:
            print(count)

    with open("embeddings_neural_with_outputs_chemicals.p", "w") as f:
        pickle.dump(embeddings_chemical, f)
