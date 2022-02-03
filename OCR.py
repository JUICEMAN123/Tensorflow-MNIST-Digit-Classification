import tensorflow as tf
from PIL import Image
import os
import numpy as np

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

SAVES = 'saves/'
DATA = 'data/'

def create_and_build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(56, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(56, (3, 3), activation='relu'))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(56, activation='relu'))
    model.add(layers.Dense(10))
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

def construct_and_train_model(train_images, train_labels, test_images, test_labels, epoch_count):
    model = create_and_build_model()
    history = model.fit(train_images, train_labels, epochs=epoch_count, validation_data=(test_images, test_labels))
    return model, history

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')

def read_image_data_in_grayscale(path): #matrix form
    img_gs = Image.open(path).convert('LA')
    img_gs_data = [pixel[0] for pixel in img_gs.getdata()]
    w, h = img_gs.size
    return np.reshape(img_gs_data, (w, h, 1))

def read_image_data_in_grayscale_normalized(path):
    normalized = read_image_data_in_grayscale(path) / 255.0
    return normalized.tolist()


def MAIN():
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    train_images, test_images = train_images / 255.0, test_images / 255.0
    
    train_images = train_images.reshape((60000,28,28,1))
    test_images = test_images.reshape((10000,28,28,1))

    model, history = construct_and_train_model(train_images, train_labels, test_images, test_labels, 10)
    model.save(SAVES)
    model = models.load_model(SAVES)
    
    predict_images = [read_image_data_in_grayscale_normalized(DATA + filename) for filename in os.listdir(DATA)]
    predict_outcomes = model.predict(predict_images)
    
    print(predict_outcomes)
  
    
MAIN()