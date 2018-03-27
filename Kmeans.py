import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
import random
import matplotlib.cm as cm
from keras.datasets import mnist
import progressbar
import time


class Point(object):
    def __init__(self, t_features, t_label, true_label):
        self.features = t_features
        self.label = t_label
        self.true_label = true_label

    def __eq__(self, other):
        for i, f in enumerate(self.features):
            if f != other.features[i]:
                return False
        return True

    def __ne__(self, other):
        for i, f in enumerate(self.features):
            if f != other.features[i]:
                return True
        return False


class KMeans(object):
    def __init__(self, t_nbr_cluster, t_min_x, t_max_x):
        self.nbr_cluster = t_nbr_cluster
        self.min_x = t_min_x
        self.max_x = t_max_x
        # self.color_dict = cm.rainbow(np.linspace(0, 1, self.nbr_cluster))
        self.color_dict = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow',
                           4: 'cyan', 5: 'black', 6: 'grey',
                           7: 'orange', 8: 'magenta', 9: 'lawngreen', 10: 'darkblue'}
        # self.X = self.load_data()
        self.X = self.load_mnist_data()
        self.nbr_data = len(self.X)
        print(self.nbr_data)
        self.centroid_index = self.get_random_centroid_index()
        self.centroid = self.get_centroid()
        # self.update_colors()

    def load_data(self):
        points = list()
        generated_points = sklearn.datasets.samples_generator.make_blobs(n_samples=self.nbr_data,
                                                                         centers=self.nbr_cluster,
                                                                         n_features=2, random_state=0,
                                                                         center_box=(self.min_x, self.max_x))
        for i, x in enumerate(generated_points[0]):
            points.append(Point(x, random.choice(generated_points[1])))
        return points

    def load_mnist_data(self):
        points = list()
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        for i, x in enumerate(x_train[:5000]):
            x = x.flatten()
            points.append(Point(x, y_train[i], y_train[i]))
        return points

    def draw(self):
        for point in self.X:
            plt.scatter(point.features[0], point.features[1], color='blue')
        for x in self.centroid_index:
            plt.scatter(self.X[x].features[0], self.X[x].features[1], color='red')

    def get_random_centroid_index(self):
        return [np.random.randint(0, self.nbr_data) for x in range(self.nbr_cluster)]

    def get_centroid(self):
        return [self.X[i] for i in self.centroid_index]

    def calc_distance(self, point, centroid):
        return np.sqrt(np.linalg.norm(point.features - centroid.features, 2, 0))

    def update_centroids(self):
        for c in self.centroid:
            for i, f in enumerate(c.features):
                feat_sum = 0
                points = [x for x in self.X if x.label == c.label]
                for p in points:
                    feat_sum += p.features[i]
                c.features[i] = feat_sum / len(points)

    def update_labels(self):
        distances = list()
        for i, x in enumerate(self.X):
            current_label = x.label
            centroid_list = list()
            for c in self.centroid:
                centroid_list.append(self.calc_distance(x, c))
            distances.append(centroid_list)
            if current_label != distances[i].index(min(distances[i])):
                x.label = distances[i].index(min(distances[i]))

    def update_colors(self):
        for point in self.X:
            plt.scatter(point.features[0], point.features[1], color=self.color_dict[point.label])
        for c in self.centroid:
            plt.scatter(c.features[0], c.features[1], marker='+', color=self.color_dict[c.label], s=200)

    def iterate(self, max_val=10, freq=5):
        with progressbar.ProgressBar(max_value=max_val) as bar:
            continue_iteration = True
            i = 0
            while continue_iteration:
                old_centroid = self.centroid
                self.update_centroids()
                self.update_labels()
                i += 1
                bar.update(i)
                if i % freq == 0:
                    self.draw_hist(i)
                if i >= max_val:
                    print("Timeout too much iteration")
                    continue_iteration = False

    def draw_hist(self, cpt):
        plt.clf()
        plt.figure(1)
        cpt = 1
        for c in self.centroid:
            tohist = [x.true_label for x in self.X if x.label == c.label]
            ax = plt.subplot(2, 5, cpt)
            ax.set_title(c.label)
            cpt += 1
            plt.hist(tohist)
        plt.savefig("res/10mean_{}_{}".format(str(int(time.time())), cpt))


if __name__ == '__main__':
    testkm = KMeans(10, 0, 50)
    testkm.iterate(200,10)