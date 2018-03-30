from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

def get_labels(file):
    origin = open(file,"r")
    classes = []
    choices = ['dorsal right', 'dorsal left', 'palmar right', 'palmar left']
    for idx, line in enumerate(origin):
        if idx == 0:
            continue
        else:
            try:
                id, age, gender, skinColor, accessories, nailPolish, aspectOfHand, imageName, irregularities = line.split(',')
                if aspectOfHand.lower().strip() in choices:
                    classes.append(choices.index(aspectOfHand))
            except Exception as e:
                print(e)
                continue
    return np.asarray(classes)

a = get_labels("HandInfo.txt")
print(a)
print(np.unique(a))

#
# (x_train, _), (x_test, label) = mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
# x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#
# encoding_dim = 32
# nbcluster = 10
# input_img = Input(shape=(784,))
# encoded = Dense(encoding_dim, activation='tanh')(input_img)
# decoded = Dense(784, activation='linear')(encoded)
# autoencoder = Model(input_img, decoded)
# encoder = Model(input_img, encoded)
# encoded_input = Input(shape=(encoding_dim,))
# autoencoder.compile(optimizer='sgd', loss='mse')
# autoencoder.fit(x_train, x_train,
#                 epochs=10,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))
#
# encoded_imgs = encoder.predict(x_test)
# autoencoder.save('model.h5')
# encoded_labels = []
# for idx, i in enumerate(encoded_imgs.tolist()):
#     encoded_labels.append([i,label[idx]])
# truc = list()
# for i in range(nbcluster*3):
#     truc.append([])
#
# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(111, projection='3d')
#
# plt.rcParams['legend.fontsize'] = 10
# for line in encoded_labels:
#     if line[1] == 0:
#         truc[0].append(line[0][0])
#         truc[1].append(line[0][1])
#         truc[2].append(line[0][2])
#     elif line[1] == 1:
#         truc[3].append(line[0][0])
#         truc[4].append(line[0][1])
#         truc[5].append(line[0][2])
#     elif line[1] == 2:
#         truc[6].append(line[0][0])
#         truc[7].append(line[0][1])
#         truc[8].append(line[0][2])
#     elif line[1] == 3:
#         truc[9].append(line[0][0])
#         truc[10].append(line[0][1])
#         truc[11].append(line[0][2])
#     elif line[1] == 4:
#         truc[12].append(line[0][0])
#         truc[13].append(line[0][1])
#         truc[14].append(line[0][2])
#     elif line[1] == 5:
#         truc[15].append(line[0][0])
#         truc[16].append(line[0][1])
#         truc[17].append(line[0][2])
#     elif line[1] == 6:
#         truc[18].append(line[0][0])
#         truc[19].append(line[0][1])
#         truc[20].append(line[0][2])
#     elif line[1] == 7:
#         truc[21].append(line[0][0])
#         truc[22].append(line[0][1])
#         truc[23].append(line[0][2])
#     elif line[1] == 8:
#         truc[24].append(line[0][0])
#         truc[25].append(line[0][1])
#         truc[26].append(line[0][2])
#     elif line[1] == 9:
#         truc[27].append(line[0][0])
#         truc[28].append(line[0][1])
#         truc[29].append(line[0][2])
#     else:
#         print("c'est mort")
#
# ax.plot(truc[0], truc[1], truc[2], 'o', label='Classe 0', c='blue')
# ax.plot(truc[3], truc[4], truc[5], 'o', label='Classe 1', c='red')
# ax.plot(truc[6], truc[7], truc[8], 'o', label='Classe 2', c='green')
# ax.plot(truc[9], truc[10], truc[11], 'o', label='Classe 3', c='yellow')
# ax.plot(truc[12], truc[13], truc[14], 'o', label='Classe 4', c='pink')
# ax.plot(truc[15], truc[16], truc[17], 'o', label='Classe 5', c='purple')
# ax.plot(truc[18], truc[19], truc[20], 'o', label='Classe 6', c='cyan')
# ax.plot(truc[21], truc[22], truc[23], 'o', label='Classe 7', c='lightblue')
# ax.plot(truc[24], truc[25], truc[26], 'o', label='Classe 8', c='lightgreen')
# ax.plot(truc[27], truc[28], truc[29], 'o', label='Classe 9', c='grey')
# plt.legend()
#
# # plt.title('Samples for class 1 and class 2')
# # ax.legend(loc='upper right')
# plt.savefig('colorful_6.png')
#
# plt.show()