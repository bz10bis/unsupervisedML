import numpy as np
import matplotlib.pyplot as plt
import random

N = 10
test1 = random.sample(range(100, 250), 10)
test2 = random.sample(range(100, 250), 10)
test3 = random.sample(range(100, 250), 10)
test4 = random.sample(range(100, 250), 10)
test5 = random.sample(range(100, 250), 10)
test6 = random.sample(range(100, 250), 10)
test7 = random.sample(range(100, 250), 10)
test8 = random.sample(range(100, 250), 10)
test9 = random.sample(range(100, 250), 10)
test10 = random.sample(range(100, 250), 10)

fig, ax = plt.subplots()

ind = np.arange(N)
width = 0.08

p1 = ax.bar(ind, test1, width, color='r', bottom=0)
p2 = ax.bar(ind + width, test2, width, color='y', bottom=0)
p3 = ax.bar(ind + 2 * width, test3, width, color='b', bottom=0)
p4 = ax.bar(ind + 3 * width, test4, width, color='g', bottom=0)
p5 = ax.bar(ind + 4 * width, test5, width, color='black', bottom=0)
p6 = ax.bar(ind + 5 * width, test6, width, color='cyan', bottom=0)
p7 = ax.bar(ind + 6 * width, test7, width, color='orange', bottom=0)
p8 = ax.bar(ind + 7 * width, test8, width, color='magenta', bottom=0)
p9 = ax.bar(ind + 8 * width, test9, width, color='purple', bottom=0)
p10 = ax.bar(ind + 9 * width, test10, width, color='lightblue', bottom=0)

ax.set_title('Score par cluster')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5', 'Cluster 6', 'Cluster 7', 'Cluster 8', 'Cluster 9'))

ax.autoscale_view()

plt.show()