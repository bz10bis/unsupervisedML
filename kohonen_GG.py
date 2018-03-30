from sklearn.cluster import KMeans
import numpy as np
import random
import matplotlib.pyplot as plt
from math import sqrt, exp

gamma = 0.7
alpha = 0.4
nb_clusters = 9
clusters = list()
iter = 10000

def distance(point, cluster):
    ptX, ptY = point
    clustX = cluster['coord'][0]
    clustY = cluster['coord'][1]
    val = sqrt((clustX - ptX) ** 2 + (clustY - ptY) ** 2)
    return val

def find_nearest_clust(point, allclust):
    values = []
    for clust in allclust:
        values.append(distance(point, clust))
    return allclust[values.index(min(values))]

def norme(clust1, clust2, truc):
    x1, y1 = clust1
    x2, y2 = clust2
    if truc == 'plus':
        val = sqrt((x1**2)+(y1**2)) + sqrt((x2**2)+(y2**2))
    else:
        val = sqrt((x1 ** 2) + (y1 ** 2)) - sqrt((x2 ** 2) + (y2 ** 2))
    return val


#DATASET JOUET
rep_vis = [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2], [2,0], [2,1], [2,2]]
X = [0,2,0,0,1,2,3,1,1,5,1,10,9,7,7,8,9,10,9,8,6,7,8,9]
Y = [1,0,1,3,4,1,2,4,0,10,8,7,9,10,7,8,10,0,2,3,1,4,2,4]
datas = []
for idx, i in enumerate(X):
    datas.append([i,Y[idx]])

# INITIALISATION DES CLUSTERS
for i in range(9):
    clusters.append(dict({'coord': datas[random.randint(0,len(datas)-1)], 'coordVis': rep_vis[i]}))

# ITERATIONS
for i in range(iter):
    point = np.asarray(datas[random.randint(0,len(datas)-1)])
    clust = find_nearest_clust(point, clusters)
    for j in clusters:
        H = exp(-gamma) * norme(j['coordVis'], clust['coordVis'], truc='plus')
        j['coord'] = np.asarray(j['coord'])
        j['coord'] = j['coord'] + alpha * H * (point  - j['coord'])

AX = []
BX = []

for i in clusters:
    AX.append(i['coord'][0])
    BX.append(i['coord'][1])

plt.scatter(AX, BX, s=50)
plt.scatter(X,Y, s=50)
plt.title(str(iter) + ' iterations')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('test.png')
plt.show()