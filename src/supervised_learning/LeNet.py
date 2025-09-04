import numpy as np
from supervised_learning.ANN import myANN, DenseLayer, Conv2DLayer, MaxPooling2DLayer, FlattenLayer, CrossEntropyLoss, Adam
from supervised_learning.ANN import Sigmoid, ReLU, Softmax
import tensorflow as tf
from tensorflow.keras import layers, models

def create_lenet_model():
    model = myANN()
    # LeNet-5 Architecture
    model.add(Conv2DLayer(n_filters=6, kernel_size=5, use_padding=False, activation=Sigmoid))
    model.add(MaxPooling2DLayer(pool_size=2, strides=2))
    model.add(Conv2DLayer(n_filters=16, kernel_size=5, use_padding=False, activation=Sigmoid))
    model.add(MaxPooling2DLayer(pool_size=2, strides=2))
    model.add(FlattenLayer())
    model.add(DenseLayer(16 * 5 * 5, 120, activation=Sigmoid))
    model.add(DenseLayer(120, 84, activation=Sigmoid))
    model.add(DenseLayer(84, 10, activation=Softmax))
    
    return model

def create_lenet_keras(input_shape=(32, 32, 1)):

    model = models.Sequential()

    model.add(layers.Conv2D(filters=6, kernel_size=(5, 5), activation='sigmoid', input_shape=input_shape))

    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(filters=16, kernel_size=(5, 5), activation='sigmoid'))

    model.add(layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(units=120, activation='sigmoid'))

    model.add(layers.Dense(units=84, activation='sigmoid'))

    model.add(layers.Dense(units=10, activation='softmax'))
    
    return model