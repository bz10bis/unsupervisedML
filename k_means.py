from sklearn.cluster import KMeans
import numpy as np
import random
import matplotlib.pyplot as plt
from math import sqrt

nbClusters = 3
datas_clust = []
clust_1 = []
clust_2 = []
clust_3 = []

X = [0,2,0,0,1,1,1,10,9,8,9,10,9,8,6,7,8,9]
Y = [0,1,3,1,2,4,0,10,8,9,10,8,10,0,1,4,2,4]
datas = []
for idx, i in enumerate(X):
    datas.append([i,Y[idx]])

# pour chaque point, lui attribuer un cluster au hasard
for i in datas:
    a = random.randint(0, nbClusters-1)
    datas_clust.append([i,a])

def find_clust(datas_clust):
    # Trouver le centroid des représentants
    for i in datas_clust:
        a, b = i
        if b == 0:
            clust_1.append(a)
        if b == 1:
            clust_2.append(a)
        if b == 2:
            clust_3.append(a)
    return clust_1, clust_2, clust_3

def generate_rep(clust):
    print("GENERATION REP")
    varx = 0
    vary = 0
    rep = []
    lol = len(clust)
    for i, j in clust:
        varx += i
        vary += j
    rep = [varx / lol, vary / lol]
    print(rep)
    return rep

def redefinition_cluster(rep1, rep2, rep3, data):
    #calculer la distance entre le représentant et les points
    rep1X, rep1Y = rep1
    rep2X, rep2Y = rep2
    rep3X, rep3Y = rep3
    datas_clust = []
    for pt in data:
        aX, aY = pt
        d1 = sqrt((rep1X - aX) ** 2 + (rep1Y - aY) ** 2)
        d2 = sqrt((rep2X - aX) ** 2 + (rep2Y - aY) ** 2)
        d3 = sqrt((rep3X - aX) ** 2 + (rep2Y - aY) ** 2)
        if min(d1,d2,d3) == d1:
            datas_clust.append([pt, 0])
        elif min(d1,d2,d3) == d2:
            datas_clust.append([pt, 1])
        elif min(d1, d2, d3) == d3:
            datas_clust.append([pt, 2])
        else:
            print("chelou")
    return datas_clust

def main():

    clust_1, clust_2, clust_3 = find_clust(datas_clust)
    rep1 = generate_rep(clust_1)
    rep2 = generate_rep(clust_2)
    rep3 = generate_rep(clust_3)
    p = redefinition_cluster(rep1,rep2,rep3,datas)
    print('premier iter OK')

    for i in range(0,9):
        print('iter ' + str(i))
        clust_1, clust_2, clust_3 = find_clust(p)
        rep1 = generate_rep(clust_1)
        rep2 = generate_rep(clust_2)
        rep3 = generate_rep(clust_3)
        p = redefinition_cluster(rep1,rep2,rep3,datas)

    aX, aY = rep1
    bX, bY = rep2
    cX, cY = rep3

    print(p)

    plt.scatter(X, Y, s=50)
    plt.scatter(aX, aY, s=50)
    plt.scatter(bX, bY, s=50)
    plt.scatter(cX, cY, s=50)
    plt.title('Nuage de points avec Matplotlib')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('Test.png')
    plt.show()


main()


# Triche

# kmeans = KMeans(n_clusters=3, random_state=0).fit(datas)
# print(kmeans.labels_)
# print(kmeans.predict([[0, 0], [4, 4], [8,8]]))
# print(kmeans.cluster_centers_)

