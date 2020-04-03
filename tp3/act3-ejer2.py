# Plot linear nuSVM classifier in the iris dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


# load iris dataset
iris = datasets.load_iris()

# Take the first two features (sepal & petal) separated to simplify.
Xsepal = iris.data[:, :2]
Xpetal = iris.data[:, 2:]
y = iris.target

# we create an instance of SVM and fit out data.
# 1 model with sepal & 1 model with petal
datas = (Xsepal, Xpetal)
models = (svm.NuSVC(nu=0.5, kernel='poly', degree=3, gamma='auto').fit(data, y) for data in datas)
# Set-up 1x2 grid for plotting.
fig, sub = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# title for the plots
titles = ('Nu-Support Vector Classif. Sepal',
          'Nu-Support Vector Classif. Petal')
for clf, data, title, ax in zip(models, datas, titles, sub.flatten()):
    # Create points to plot in
    X0, X1 = data[:, 0], data[:, 1]
    h = .02
    x_min, x_max = X0.min() - 1, X0.max() + 1
    y_min, y_max = X1.min() - 1, X1.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Plot the decision boundaries for a classifier
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    # plot points
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Length')
    ax.set_ylabel('Width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()