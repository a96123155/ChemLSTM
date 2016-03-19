#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-
from __future__ import print_function

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans, MeanShift
from sklearn.decomposition import PCA
import json
import pickle
import random
import sys

n_d = 100

# with open("embeddings_tao_99.json") as f:
#     embeddings = json.load(f)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: {} [input file]".format(sys.argv[0]))
        quit(0)

    with open(sys.argv[1]) as f:
        embeddings = pickle.load(f)

    number = min(len(embeddings), 20000)

    # copy up to 20k elements
    X = np.ndarray((number, n_d), dtype=np.float32)
    if number < len(embeddings):
        keys = random.sample(embeddings.keys(), number)
    else:
        keys = embeddings.keys()

    for i, word in enumerate(keys):
        X[i] = embeddings[word]

    print(X.shape)

    # use PCA to reduce dimensionality
    num_pca_comps = 20
    pca = PCA(n_components=num_pca_comps)
    pca.fit(X)
    print("PCA explains:", pca.explained_variance_ratio_)

    # perform k-means to obtain clusters
    num_clusters = 100
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_ids = kmeans.fit_predict(X)

    clusters = [{} for i in set(cluster_ids)]
    for key, cid, x in zip(keys, cluster_ids, X):
        clusters[cid][key] = x

    # select a cluster and plot it

    def view_cluster(cid, subcid=0):
        # smallest_cid = np.argmin(len(cluster) for cluster in clusters)
        smallest_cid = cid
        wanted = subcid
        print("Smallest cluster: {} ({} elements)".format(smallest_cid, len(clusters[smallest_cid])))
        X2 = np.ndarray((len(clusters[smallest_cid]), num_pca_comps), dtype=np.float32)
        for i, word in enumerate(clusters[smallest_cid].values()):
            X2[i] = pca.transform(X[i])

        # further reduce to three dimensions
        pca2 = PCA(n_components=3)
        Y = pca2.fit_transform(X2)

        # build 3d scatter plot
        x, y, z = zip(*Y)

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection="3d")
        # ax.scatter(x, y, z)

        for i, word in enumerate(clusters[smallest_cid]):
            xx, yy, zz = x[i], y[i], z[i]
            # ax.text(xx, yy, zz, word, bbox=dict(facecolor='white', alpha=0.5), zorder=-1)

        # plt.show()

        kmeans2 = KMeans(n_clusters=10)
        cluster_ids2 = kmeans2.fit_predict(Y)
        clusters2 = [list() for i in xrange(10)]
        for cid, key in zip(cluster_ids2, clusters[smallest_cid]):
            clusters2[cid].append(key)
            for cid, cluster in enumerate(clusters2):
                print("CLUSTER {}".format(cid))
                print("\n".join(cluster))

        X3 = np.ndarray((len(clusters2[wanted]), 3), dtype=np.float32)
        j = 0
        texts = []
        for i, cluster in enumerate(cluster_ids2):
            if cluster == wanted:
                X3[j] = Y[i]
                texts.append(clusters[smallest_cid].keys()[i])
                j += 1

        pca3 = PCA(n_components=2)
        Y3 = pca3.fit_transform(X3)

        x, y = zip(*Y3)
        plt.scatter(x, y)
        for i, word in enumerate(texts):
            plt.text(x[i], y[i], word)
        plt.show()

    view_cluster(0, 0)
