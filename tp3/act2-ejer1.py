# Bayesian optimization over Cnn images shapes clasiffication
# Curso Sistemas Inteligentes - Trabajo práctico 3 - Actividad 2
# Red neuronal convolucional para identificar figuras geométricas simples

# import packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import gc  # garbage collector
from keras import layers, models, optimizers, losses
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
from GPyOpt.methods import BayesianOptimization
from sklearn.model_selection import KFold


# read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images, nrows=64, ncolumns=64):
    # images
    X = []
    # labels
    y = []

    # label - class
    # 0 - 'circle', 1 - 'ellipse', 2 - 'rectangle', 3 - 'square', 4 - 'triangle'
    for image in list_of_images:
        # read image
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (nrows, ncolumns), interpolation=cv2.INTER_CUBIC))
        # get the labels
        if 'circle' in image:
            y.append(0)
        elif 'ellipse' in image:
            y.append(1)
        elif 'rectangle' in image:
            y.append(2)
        elif 'square' in image:
            y.append(3)
        elif 'triangle' in image:
            y.append(4)

    return X, y


# conv model, 4 hidden layers with relu activation
def get_conv_model(units):
    model = models.Sequential()
    model.add(layers.Conv2D(units[0], (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(units[1], (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(units[2], activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units[3], activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units[4], activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units[5], activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=5, activation='softmax'))

    # compile model: RMSprop optimizer with a learning rate of 0.0001 sparse_categorical_crossentropy loss
    model.compile(loss=losses.sparse_categorical_crossentropy,
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    return model


# train de model with configuration in parameter units
def train_model(units):
    print("Start training")

    # cast variables to integers
    values = [int(param) for param in units[0]]

    # kfold cross-validation
    kf = KFold(n_splits=4, shuffle=False)
    # KFold performance history
    KF_history = []

    # Split our data into train and validation set guided by fold
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # get the length of the train and validation data
        ntrain = len(X_train)
        nval = len(X_val)
        batch_size = 32
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        val_datagen = ImageDataGenerator(rescale=1. / 255)

        # create the image generators
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

        model = get_conv_model(values)
        # the training part - 50 epochs - 100 steps per epoch
        history = model.fit_generator(train_generator, steps_per_epoch=ntrain // batch_size, epochs=50,
                                      validation_data=val_generator, validation_steps=nval // batch_size)

        if np.max(history.history['val_acc']) < 0.7:
            performance = np.max(history.history['val_acc'])
            print(units, performance)
            return performance

        KF_history.append(history)

    # return average of best validation by fold
    performance = np.average([np.max(history.history['val_acc']) for history in KF_history])
    print(units, performance)
    return performance


# Load data set
train_dir = './shapes/train'
train_imgs = ['./shapes/train/{}'.format(i) for i in os.listdir(train_dir)]
random.shuffle(train_imgs)

# preprocess images
X, y = read_and_process_image(train_imgs)

del train_imgs
gc.collect()

# convert list to numpy array
X = np.array(X)
y = np.array(y)

# parameters for bayesian optimization
units = [
    {'name': 'conv2d1', 'type': 'discrete', 'domain': np.arange(20, 100, 1)},
    {'name': 'conv2d2', 'type': 'discrete', 'domain': np.arange(20, 100, 1)},
    {'name': 'dense1', 'type': 'discrete', 'domain': np.arange(100, 1024, 1)},
    {'name': 'dense2', 'type': 'discrete', 'domain': np.arange(100, 1024, 1)},
    {'name': 'dense3', 'type': 'discrete', 'domain': np.arange(100, 1024, 1)},
    {'name': 'dense4', 'type': 'discrete', 'domain': np.arange(100, 1024, 1)}
]

# adquisition functions: Expected Improbement, Maximum probability of improvement
for acquisition in ['EI', 'MPI']:
    print('Run with Acquisition function: ' + acquisition)

    # bayesian optimizer definition
    optimizer = BayesianOptimization(f=train_model,
                                     domain=units,
                                     model_type='GP',
                                     acquisition_type=acquisition,
                                     acquisition_jitter=0.05,
                                     verbosity=True,
                                     maximize=True)

    # 20 optimization iterations
    optimizer.run_optimization(max_iter=20)

    # results
    y_bo = np.maximum.accumulate(-optimizer.Y).ravel()
    plt.plot(y_bo, 'bo-', label='Bayesian Optimization')
    plt.xlabel('Iteration')
    plt.ylabel('Performance')
    plt.title('20 iterations')
    plt.legend()
    plt.show()