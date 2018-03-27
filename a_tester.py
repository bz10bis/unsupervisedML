from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse
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

encoding_dim = 2
input_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='linear')(input_img)
decoded = Dense(784, activation='linear')(encoded)

model = Model(input_img, [decoded, encoded])

model.compile(optimizer='sgd', loss=[mse,mse], loss_weights=[1,0])

model.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = model.predict(x_test)
model.save('model.h5')  # creates a HDF5 file 'my_model.h5'

X = [x[0] for x in encoded_imgs]
Y = [x[1] for x in encoded_imgs]
Z = [x[2] for x in encoded_imgs]
print(X)

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10
ax.plot(X, Y, Z, 'o', markersize=8, color='blue', alpha=0.5, label='class1')
#ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')

plt.show()