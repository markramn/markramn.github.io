# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 07:00:48 2022

@author: nicholas.markram
"""

import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.distutils.system_info import x11_info
from sklearn.preprocessing import LabelEncoder
from keras.wrappers.scikit_learn import KerasClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras.preprocessing import image
from numpy.random import seed
import os
import random as rn
import tensorflow as tf
import cv2
import imageio
import PIL

def train_model():

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(30)
    rn.seed(1234)
    tf.random.set_seed(6)

    labels = ['Elephant', 'ElephantC', 'Giraffe', "Hyena", 'Hippo', 'Wildebeest', 'HyenaC', 'Rhino', 'RhinoC']

    img_size = 18

    def get_data(data_dir):
        data = []
        for label in labels:
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img))[..., ::-1]  # convert BGR to RGB format
                    resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Reshaping images to preferred size
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(e)
        return np.array(data)

    # Fetch train and 80/20 split into training_data and validation_data
    train_full = get_data('FinalTrainSet')
    train, val = train_test_split(train_full, test_size=0.3, random_state=30)

    # Data Preprocessing and Data Augmentation
    x_train = []
    y_train = []
    x_val = []
    y_val = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in val:
        x_val.append(feature)
        y_val.append(label)

    train_images = np.array(x_train)
    train_labels = np.array(y_train)

    test_images = np.array(x_val)
    test_labels = np.array(y_val)

    train_images.shape
    test_images.shape

    train_images = train_images.astype('float32')
    test_images = test_images.astype('float32')

    train_images /= 255
    test_images /= 255

    # Better CNN
    model = keras.Sequential([
        keras.layers.Conv2D(64, 2, activation='relu', input_shape=(18, 18, 3)),
        keras.layers.Conv2D(64, 2, activation='relu'),
        keras.layers.Conv2D(64, 2, activation='relu'),
        keras.layers.Conv2D(128, 2, activation='relu'),
        keras.layers.Conv2D(128, 2, activation='relu'),
        keras.layers.Conv2D(128, 2, activation='relu'),
        keras.layers.MaxPool2D(2, 2),
        keras.layers.Dropout(0.5),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(9, activation='softmax')
    ])

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    history = model.fit(train_images, train_labels, epochs=10, batch_size=32,
                        validation_data=(test_images, test_labels))


    return model


#model = train_model()
#model.save('CCF_Model')

def predict(X, model):

    labels = ['Elephant', 'ElephantC', 'Giraffe', "Hyena", 'Hippo', 'Wildebeest', 'HyenaC', 'Rhino', 'RhinoC']

    #X = image.img_to_array(img)

    X = np.expand_dims(X, axis=0)

    val = model.predict(X)

    predicted = labels[np.argmax(val[0])]
    sftmax = max(val[0])

    return predicted,sftmax