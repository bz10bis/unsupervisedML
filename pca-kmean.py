# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from matplotlib import pyplot as plt
from keras.datasets import mnist

"""## Chargement des données"""


def load_mnist_dataset(nbr_sample):
    (x_train, y_train), (_, _) = mnist.load_data()
    dataset = (np.array([x.flatten() for x in x_train[:nbr_sample]])).T
    return x_train, y_train, dataset


def load_hand_dataset(path, nbr_sample):
    filelist = os.listdir(path)
    x_train = np.array([np.array(Image.open(os.getcwd() + '\\Hands\\normalized_32\\' + fname))[:, :, 0] for fname in
                        filelist[:nbr_sample]])
    y_train = []
    dataset = np.array(np.array([x.flatten() for x in x_train]).T, dtype=int)

    return x_train, y_train, dataset


class PCA_custom(object):
    def __init__(self, t_dim, t_newDim, t_nbr_samples, mode):

        self.dim = t_dim
        self.new_dim = t_newDim
        self.nbr_sample = t_nbr_samples

        if mode == 'mnist':
            self.x_train, self.y_train, self.dataset = load_mnist_dataset(self.nbr_sample)

        if mode == 'hand':
            self.x_train, self.y_train, self.dataset = load_hand_dataset("Hands/normalized_32", self.nbr_sample)

        # process the covariance matrix
        self.cov_mat = np.cov(self.dataset)

        # get the eig values/vectors
        self.eig_val, self.eig_vec = np.linalg.eig(self.cov_mat)
        self.eig_val = np.real(self.eig_val)
        self.eig_vec = np.real(self.eig_vec)

        self.sorted_vectors = np.array([x[1] for x in sorted(zip(self.eig_val, self.eig_vec), key=lambda x: x[0])])

        # extract the "t_newDim" best eig values
        self.sorted_vectors = self.sorted_vectors[:t_newDim].T

        # transform the matrix
        self.transformed_dataset = np.dot(self.dataset.T, np.transpose(self.sorted_vectors).T)

        # decompress data
        self.decompressed = np.dot(self.transformed_dataset, self.sorted_vectors.T)


data = PCA_custom(32, 1024, 10000, mode='hand').decompressed
print('data from PCA shape : ', data.shape)
data = data.reshape(10000, 1024)
label = np.arange(data.shape[0])

"""## Déclaration des parametres"""

# @title Parametres
epochs = 200  # @param {type:"integer"}
clusters = 10  # @param {type:"slider", min:0, max:20, step:1}
nombre_exemples = 50  # @param {type:"integer"}

"""## Définition de la fonction Kmeans"""


def kmean(data, k, epochs, labels):
    start_time = time.time()
    print("#" * 20 + " KMEANS " + "#" * 20)
    print("data shape\tnb clusters\tnbepochs")
    print("-" * 50)
    print("{}\t{}\t\t{}".format(data.shape, k, epochs))
    print("-" * 50)
    n = data.shape[0]
    nf = data.shape[1]
    rows = np.arange(n)
    c_idx = np.random.choice(n, k)
    centroids = data[c_idx].T
    print(centroids.shape)
    repeated_data = np.stack([data] * k, axis=-1)
    for i in range(epochs):
        local_time = time.time()
        distances = np.sqrt(np.sum(np.square(repeated_data - centroids), axis=1))
        assignments = np.argmin(distances, axis=-1)
        counts = np.bincount(assignments)
        concat_matx = np.zeros([n, k, nf])
        concat_matx[rows, assignments] = data
        centroids = concat_matx.sum(axis=0).T / counts.clip(min=1).T
        print("iteration: {} {} -- t:{} s".format(i, counts, round(time.time() - local_time, 2)))
    print("-" * 50)
    print("FINAL {}".format(counts))
    print("Elapse time: {} s".format(round(time.time() - start_time, 2)))
    hist = []
    for c in range(k):
        hist.append([])
    for i in range(len(labels)):
        hist[labels[i]].append(assignments[i])
    plt.hist(hist)
    plt.show()


kmean(data, clusters, epochs, label)
