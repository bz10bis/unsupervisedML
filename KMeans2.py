import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from keras.datasets import mnist
import progressbar
import time


class KMeans(object):

    def __init__(self, data, epoch, k):
        self.data = data
        self.epoch = epoch
        self.k = k
        self.n = data.shape[0]
        self.nbr_features = data.shape[1]

        self.duplicate_data = np.stack([data] * k, axis=-1)
        self.all_row = np.arange(self.n)

        first_centroid_index = self.get_random_centroid_index()
        self.centroids = data[first_centroid_index].T
        self.iterate()

    def get_random_centroid_index(self):
        print("Getting random centroid")
        return np.random.choice(range(self.n), self.k)

    def calc_dist(self):
        print("Compute distances")
        return np.linalg.norm(self.duplicate_data - self.centroids, axis=1)

    def assign_cluster(self):
        print("Assignment")
        return np.argmin(self.distances, axis=1)

    def compute_clusters(self):
        o_encode = np.zeros([self.n, self.k, self.nbr_features])
        o_encode[self.all_row, self.assignment] = self.data
        zero = np.zeros([1, 1, 2])
        self.counts = np.sum(o_encode, axis=0)
        print(o_encode[1][0])
        print(self.counts.shape)

    def iterate(self):
        print("Start iterations:")
        self.distances = self.calc_dist()
        self.assignment = self.assign_cluster()
        self.compute_clusters()


if __name__ == "__main__":
    (data, labels), (_, _) = mnist.load_data()
    data = data.reshape(60000, 784)
    KMeans(data=data[:500], epoch=200, k=10)
