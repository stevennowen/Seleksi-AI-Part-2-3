import numpy as np
from supervised_learning.ANN import myANN, DenseLayer, Conv2DLayer, MaxPooling2DLayer, FlattenLayer, CrossEntropyLoss, Adam
from supervised_learning.ANN import Sigmoid, ReLU, Softmax
import tensorflow as tf
from tensorflow.keras import layers, models

def create_vgg_model():
    model = myANN()
    
    # VGG Architecture
    model.add(Conv2DLayer(n_filters=32, kernel_size=3, use_padding=True, activation=ReLU))
    model.add(Conv2DLayer(n_filters=32, kernel_size=3, use_padding=True, activation=ReLU))
    model.add(MaxPooling2DLayer(pool_size=2, strides=2))
    
    model.add(Conv2DLayer(n_filters=64, kernel_size=3, use_padding=True, activation=ReLU))
    model.add(Conv2DLayer(n_filters=64, kernel_size=3, use_padding=True, activation=ReLU))
    model.add(MaxPooling2DLayer(pool_size=2, strides=2))
    
    model.add(Conv2DLayer(n_filters=128, kernel_size=3, use_padding=True, activation=ReLU))
    model.add(Conv2DLayer(n_filters=128, kernel_size=3, use_padding=True, activation=ReLU))
    model.add(MaxPooling2DLayer(pool_size=2, strides=2))
    
    model.add(FlattenLayer())
    model.add(DenseLayer(128 * 4 * 4, 512, activation=ReLU))
    model.add(DenseLayer(512, 512, activation=ReLU))
    model.add(DenseLayer(512, 10, activation=Softmax))
    
    return model

def create_vgg_keras(input_shape=(32, 32, 1)):

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dense(10, activation='softmax'))
    
    return model