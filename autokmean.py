from keras.layers import Input, Dense
from keras.models import Model, load_model

encoding_dim= 30
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation="relu")(input_img)
decoded = Dense(784, activation="sigmoid")(encoded)
autoencoder = load_model('save/autoencoder_50e_30dim.h5')
# autoencoder = Model(input_img, decoded)

encoder = Model(input_img, encoded)

encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

# autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

from keras.datasets import mnist
import numpy as np
(x_train, labels), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)
e = 50
# autoencoder.fit(x_train, x_train,
# 	epochs=e,
# 	batch_size=256,
# 	shuffle=True,
# 	validation_data=(x_test, x_test))
# autoencoder.save("save/autoencoder_{}e_{}dim.h5".format(e,encoding_dim))
data = encoder.predict(x_train)

import time
import matplotlib.pyplot as plt
import numpy as np
# (data, _), (_,_) = mnist.load_data()
# data = data.reshape(60000, 784)
epochs = 200
clusters = 10

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
	repeated_data = np.stack([data] * k, axis=-1)
	continue_training = True
	i=0
	old_count = []
	while continue_training:
	#for i in range(epochs):
		local_time = time.time()
		distances = np.sum(np.square(repeated_data - centroids), axis=1)
		assignments = np.argmin(distances, axis=-1)
		counts = np.bincount(assignments)
		concat_matx = np.zeros([n, k, nf])
		concat_matx[rows, assignments] = data
		centroids = concat_matx.sum(axis=0).T / counts.clip(min=1).T
		print("iteration: {} {} -- t:{} s".format(i, counts, round(time.time()-local_time,2)))
		if i >= epochs:
			continue_training = False
		if np.array_equal(old_count, counts):
			continue_training = False
		old_count = counts
		i += 1
	print("-"*50)  
	print("FINAL {}".format(counts))
	print("Elapse time: {} s".format(round(time.time() - start_time, 2)))


	# plt.figure(1)
	hist = []
	for c in range(k):
		hist.append([])
	for i in range(len(labels)):
		hist[labels[i]].append(assignments[i])
	plt.hist(hist)
	plt.show()
		# ax = plt.subplot(2, 5, c+1)
		# ax.set_title(c)
		# plt.hist([x for x in np.where(assignments == c)])
		# ax.set_ylim([0, 2000])
	plt.show()
kmean(data, clusters, epochs, labels)

def plot_graph(k, labels, predicted_labels):
    hist = []
    for i in range(k):
        hist.append([])
    
    for i in range(len(labels)):
        hist[labels[i]].append(predicted_labels[i])

    plt.hist(hist)