import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

from keras import backend as K

points = 3000
points_test = 100
np.random.seed(0)

# triangle 1: 0.1 0.7 - 0.35 0.1 - 0.75 0.85
f1 = lambda x: (-2.4 * x + 0.94)
f2 = lambda x: (0.23 * x + 0.68)
f3 = lambda x: (1.87 * x - 0.55)

# generate random numbers
x1 = np.random.random(points)
y1 = np.random.random(points)
lab1 = np.zeros(points)
# apply right labels
f1y = [f1(x) for x in x1]
f2y = [f2(x) for x in x1]
f3y = [f3(x) for x in x1]
for i in range(points):
    if f1y[i] <= y1[i] and f2y[i] >= y1[i] and f3y[i] <= y1[i]:
        lab1[i] = 1
# generate test
x1_test = np.random.random(points_test)
y1_test = np.random.random(points_test)
lab1_test = np.zeros(points_test)
for i in range(points_test):
    if f1y[i] <= y1_test[i] and f2y[i] >= y1_test[i] and f3y[i] <= y1_test[i]:
        lab1_test[i] = 1

def draw_triangles_points():
    # scatter plot, dots colored by class value
    df = DataFrame(dict(x=x1, y=y1, label=lab1))
    colors = {0: 'green', 1: 'blue'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')

    x = np.arange(0, 10, 0.1)

    plt.plot(x, f1(x), color='red', label='triangle 1')
    plt.plot(x, f2(x), color='red')
    plt.plot(x, f3(x), color='red')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(loc='best')

    axes = plt.gca()
    axes.set_xlim([0, 1])
    axes.set_ylim([0, 1])

    # plot points
    for key, group in grouped:
        # add points
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])

    plt.show()


# split data intro train and test
X = np.transpose(np.array((x1, y1)))
y = np.array(lab1)
X_test = np.transpose(np.array((x1_test, y1_test)))
y_test = np.array(lab1_test)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print("Shape of train images is: ", X_train.shape)
print("Shape of validation images is: ", X_val.shape)
print("Shape of train labels is: ", y_train.shape)
print("Shape of validation labels is: ", y_val.shape)

K.tensorflow_backend._get_available_gpus()

# Initialising the ANN
model = Sequential()
# Adding the Single Perceptron or Shallow network
model.add(Dense(units=128, activation='relu', input_dim=2))
# Adding dropout to prevent overfitting
model.add(Dropout(p=0.1))
# Adding the output layer
model.add(Dense(units=1, activation='sigmoid'))

# criterion loss and optimizer
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size=100, epochs=64)

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

#print("Test accuracy is {}%".format(((110/114)*100)))
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# draw triangles
draw_triangles_points()
