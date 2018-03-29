from keras.layers import Input, Dense
from keras.models import Model

encoding_dim= 32
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation="relu")(input_img)
decoded = Dense(784, activation="sigmoid")(encoded)
autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
	epochs=50,
	batch_size=256,
	shuffle=True,
	validation_data=(x_test, x_test))

data = encoder.predict(x_train)

import time
import matplotlib.pyplot as plt

epochs = 10 
clusters = 10

def kmean(data, k, epochs):
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
		print("iteration: {} {} -- t:{} s".format(i, counts, round(time.time()-local_time,2)))
	print("-"*50)  
	print("FINAL {}".format(counts))
	print("Elapse time: {} s".format(round(time.time() - start_time, 2)))

	plt.figure(1)
	for c in range(k):
		ax = plt.subplot(2, 5, c+1)
		ax.set_title(c)
		plt.hist([x for x in np.where(assignments == c)])
	plt.show()
kmean(data, clusters, epochs)