from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pandas import DataFrame
import numpy as np
from collections import defaultdict

# triangle 1: 0.1 0.7 - 0.35 0.1 - 0.75 0.85
f1 = lambda x: (-2.4 * x + 0.94)
f2 = lambda x: (0.23 * x + 0.68)
f3 = lambda x: (1.87 * x - 0.55)

# generate random 2d classification dataset
X, y = make_blobs(n_samples=40, centers=2, n_features=2)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'green', 1:'blue'}
fig, ax = plt.subplots()
grouped = df.groupby('label')
newValues = defaultdict();
# plot points and rectangles
for key, group in grouped:
    # add points
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    # rectangle
    x_min = group.x.agg(np.min)
    x_max = group.x.agg(np.max)
    x_mean = group.x.agg(np.mean)
    x_std = group.x.agg(np.std)
    y_min = group.y.agg(np.min)
    y_max = group.y.agg(np.max)
    y_mean = group.y.agg(np.mean)
    y_std = group.y.agg(np.std)
    # add rectangle classification to plot
    ax.add_patch(
      patches.Rectangle(
        (x_min, y_min),   # (x,y)
        (x_max - x_min),          # width
        (y_max - y_min),          # height
        fill=False,
        zorder=2
      ))
    # add new values
    newValues[key] = [np.random.normal(x_mean, x_std, 3), np.random.normal(y_mean, y_std, 3)]

# suggest 3 more values for every classification
print(newValues[0])
print(newValues[1])
plt.scatter(newValues[0][0][0], newValues[0][1][0],s=50)
plt.scatter(newValues[0][0][1], newValues[0][1][1],s=50)
plt.scatter(newValues[0][0][2], newValues[0][1][2],s=50)
plt.scatter(newValues[1][0][0], newValues[1][1][0],s=50)
plt.scatter(newValues[1][0][1], newValues[1][1][1],s=50)
plt.scatter(newValues[1][0][2], newValues[1][1][2],s=50)

x = np.arange(0, 10, 0.1)
plt.plot(x, f1(x), color='red', label='triangle 1')
plt.plot(x, f2(x), color='red')
plt.plot(x, f3(x), color='red')

# show plot
plt.show()