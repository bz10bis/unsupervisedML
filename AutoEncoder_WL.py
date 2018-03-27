from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

(x_train, _), (x_test, label) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

encoding_dim = 3
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='tanh')(input_img)
decoded = Dense(784, activation='linear')(encoded)
autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)
encoded_input = Input(shape=(encoding_dim,))
autoencoder.compile(optimizer='sgd', loss='mse')
autoencoder.fit(x_train, x_train,
                epochs=1000,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
autoencoder.save('model.h5')

encoded_labels = []
for idx, i in enumerate(encoded_imgs.tolist()):
    encoded_labels.append([i,label[idx]])

x0 = []
y0 = []
z0 = []
x1 = []
y1 = []
z1 = []
x2 = []
y2 = []
z2 = []
x3 = []
y3 = []
z3 = []
x4 = []
y4 = []
z4 = []
x5 = []
y5 = []
z5 = []
x6 = []
y6 = []
z6 = []
x7 = []
y7 = []
z7 = []
x8 = []
y8 = []
z8 = []
x9 = []
y9 = []
z9 = []
#
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')

plt.rcParams['legend.fontsize'] = 10
for line in encoded_labels:
    if line[1] == 0:
        x0.append(line[0][0])
        y0.append(line[0][1])
        z0.append(line[0][2])
    elif line[1] == 1:
        x1.append(line[0][0])
        y1.append(line[0][1])
        z1.append(line[0][2])
    elif line[1] == 2:
        x2.append(line[0][0])
        y2.append(line[0][1])
        z2.append(line[0][2])
    elif line[1] == 3:
        x3.append(line[0][0])
        y3.append(line[0][1])
        z3.append(line[0][2])
    elif line[1] == 4:
        x4.append(line[0][0])
        y4.append(line[0][1])
        z4.append(line[0][2])
    elif line[1] == 5:
        x5.append(line[0][0])
        y5.append(line[0][1])
        z5.append(line[0][2])
    elif line[1] == 6:
        x6.append(line[0][0])
        y6.append(line[0][1])
        z6.append(line[0][2])
    elif line[1] == 7:
        x7.append(line[0][0])
        y7.append(line[0][1])
        z7.append(line[0][2])
    elif line[1] == 8:
        x8.append(line[0][0])
        y8.append(line[0][1])
        z8.append(line[0][2])
    elif line[1] == 9:
        x9.append(line[0][0])
        y9.append(line[0][1])
        z9.append(line[0][2])
    else:
        print("c'est mort")

plt.scatter(x0, y0, s=50)
plt.scatter(x1, y1, s=50)
plt.scatter(x2, y2, s=50)
plt.scatter(x3, y3, s=50)
plt.scatter(x4, y4, s=50)
plt.scatter(x5, y5, s=50)
plt.scatter(x6, y6, s=50)
plt.scatter(x7, y7, s=50)
plt.scatter(x8, y8, s=50)
plt.scatter(x9, y9, s=50)

# plt.title('Samples for class 1 and class 2')
# ax.legend(loc='upper right')
plt.savefig('colorful_4.png')

plt.show()