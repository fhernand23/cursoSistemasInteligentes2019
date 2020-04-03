# Cnn images shapes clasiffication
# Curso Sistemas Inteligentes - Trabajo práctico 2 - Actividad 3
# Red neuronal convolucional para identificar figuras geométricas simples(círculos, elipses, triángulos, cuadrados
# y rectángulos)
# Set de datos generados con 100 figuras(60 train, 15 test, 20 validation)
# Red neuronal convolucional 4 capas ocultas(Relu) y Dropout

# import packages
import cv2
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# to see our directory
import os
import random
import gc  # garbage collector

# Load data set
train_dir = './shapes/train'
test_dir = './shapes/test'

# get train images
train_imgs = ['./shapes/train/{}'.format(i) for i in os.listdir(train_dir)]
# get test images
test_imgs = ['./shapes/test/{}'.format(i) for i in os.listdir(test_dir)]
# shuffle train randomly
random.shuffle(train_imgs)

# preprocess images
# declare image dimension
nrows = 64
ncolumns = 64
# 3 for colors, 1 for grayscale
channels = 3


# read and process the images to an acceptable format for our model
def read_and_process_image(list_of_images):
    """
    Return two arrays:
        X is an array of resized images
        x is an array of labels
    """
    # images
    X = []
    # labels
    y = []

    # label - class
    # 0 - 'circle'
    # 1 - 'ellipse'
    # 2 - 'rectangle'
    # 3 - 'square'
    # 4 - 'triangle'
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


X, y = read_and_process_image(train_imgs)

del train_imgs
gc.collect()

# convert list to numpy array
X = np.array(X)
y = np.array(y)

# label - class
# 0 - 'circle'
# 1 - 'ellipse'
# 2 - 'rectangle'
# 3 - 'square'
# 4 - 'triangle'
class_names = ['circle', 'ellipse', 'rectangle', 'square', 'triangle']

# Check the shape of our data
print("Shape of train images is: ", X.shape)
print("Shape of labels is: ", y.shape)

# Split our data into train and validation set

# split data into train and test
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=2)

print("Shape of train images is: ", X_train.shape)
print("Shape of validation images is: ", X_val.shape)
print("Shape of train labels is: ", y_train.shape)
print("Shape of validation labels is: ", y_val.shape)

# clear memory
del X
del y
gc.collect()
# get the length of the train and validation data
ntrain = len(X_train)
nval = len(X_val)
batch_size = 32

from keras import layers, models, optimizers, losses
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# conv + 4 hidden layers with relu activation
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(units=5, activation='softmax'))
model.summary()

# compile model: RMSprop optimizer with a learning rate of 0.0001 sparse_categorical_crossentropy loss
model.compile(loss=losses.sparse_categorical_crossentropy,
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# create augmented configuration (prevent overfitting in small dataset)
# alternative code to improve data set quality adding alternatives
# train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2,
#                                  height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,
#                                  horizontal_flip=True,)
train_datagen = ImageDataGenerator(rescale=1. / 255)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# create the image generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

# the training part - 64 epochs - 100 steps per epoch
history = model.fit_generator(train_generator, steps_per_epoch=ntrain // batch_size, epochs=64,
                              validation_data=val_generator, validation_steps=nval // batch_size)

# Plot graphs
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

# train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Test: predict on the first 10 images of the test set
random.shuffle(test_imgs)
X_test, y_test = read_and_process_image(test_imgs[0:10])
x = np.array(X_test)
test_datagen = ImageDataGenerator(rescale=1. / 255)

i = 0
text_labels = []
plt.figure(figsize=(30, 20))

columns = 5
for batch in test_datagen.flow(x, batch_size=1):
    pred = model.predict(batch)
    pred_label = np.argmax(pred)
    pred_perc = 100 * np.max(pred)
    print("prediction {} {} {:2.0f}%".format(pred, class_names[pred_label], pred_perc))
    text_labels.append(str(pred))
    plt.subplot(5 / columns + 1, columns, i + 1)
    plt.title("Prediction {} {:2.0f}%".format(class_names[pred_label], pred_perc))
    imgplot = plt.imshow(batch[0])
    i += 1
    if i % 10 == 0:
        break
plt.show()
