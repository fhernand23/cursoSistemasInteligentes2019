from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot
import matplotlib.patches as patches
from pandas import DataFrame
import numpy as np

# generate random 2d classification dataset
X, y = make_blobs(n_samples=20, centers=2, n_features=2)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'green', 1:'blue'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
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

# show plot
pyplot.show()