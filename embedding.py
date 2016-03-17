#!/usr/bin/env python2.7

from __future__ import print_function

import numpy as np
import sys
import os
import json

import theano
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, TimeDistributedDense, Activation, Dropout, AutoEncoder
from keras.layers.embeddings import Embedding
from keras.layers import containers
from keras.layers.recurrent import SimpleRNN
from keras.utils import generic_utils

import cPickle as pickle

from sklearn.feature_extraction.text import CountVectorizer

# Embedding dimensions
batch_size = 20

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: {} [input file] [model json file] [weights file]".format(sys.argv[0]))
        quit(1)

    path = sys.argv[1]
    model_json_file = sys.argv[2]
    weights = sys.argv[3]

    print("Reading input...")
    token_regex = r"(?u)([\(\)\[\]]|\b\w+\b)"
    # cv = CountVectorizer(ngram_range=(1,ngrams), token_pattern=token_regex)
    cv = CountVectorizer(token_pattern=token_regex, min_df=2)
    an = cv.build_analyzer()

    corpus = []
    with open(path) as f:
        for line in f:
            corpus.append(line.strip())

    # vectorize and n-gram-ize the corpus
    X = cv.fit_transform(corpus)

    print("Building vectors...")
    # vocabulary size, including padding element
    vocabulary_size = len(cv.vocabulary_) + 1

    print("Vocabulary size: {}  Corpus size: {}".format(vocabulary_size, len(corpus)))

    print("Rebuilding model...")
    # embedding layer
    with open(model_json_file) as f:
        model = model_from_json(f.read())
    model.load_weights(weights)

    def generate_chem_x2(number):
        sequence_names = []
        sequence_tokens = []
        for line in corpus:
            line = line.strip()
            tokens = an(line)
            token_ids = [0] * len(tokens)
            for i, token in enumerate(tokens):
                tid = cv.vocabulary_.get(token)
                if tid is not None:
                    token_ids[i] = tid + 1
                    
            sequence_tokens.append(token_ids)
            sequence_names.append(line)

            if len(sequence_tokens) > number:
                padded_sequence_tokens = pad_sequences(sequence_tokens, maxlen=batch_size)
                yield sequence_names, padded_sequence_tokens
                sequence_tokens = []
                sequence_names = []

        if len(sequence_tokens) > 0:
            padded_sequence_tokens = pad_sequences(sequence_tokens, maxlen=batch_size)
            yield sequence_names, padded_sequence_tokens

    print("Producing embeddings...")
    embeddings = {}
    try:
        for chemicals, x2s in generate_chem_x2(1000):
            preds = model.predict(x2s)
            for i, name in enumerate(chemicals):
                embeddings[name] = preds[i][-1]
                
    except KeyboardInterrupt:
        pass

    with open("embeddings.p", "w") as f:
        pickle.dump(embeddings, f)
