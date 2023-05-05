# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 20:25:59 2023

@author: 20100
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense, BatchNormalization, Dropout, Flatten, Conv2D, MaxPooling2D,Activation
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import os
os.environ['PYTHONHASHSEED']= '0'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
np.random.seed(0)
import random as rn
rn.seed(0)
tf.random.set_seed(0)
import FoodGUI as K

# set the paths for the training, validation, and testing sets
train_dir = 'C:/Users/Lenovo/Desktop/Projects/Selected-2/dataset/training'
val_dir = 'C:/Users/Lenovo/Desktop/Projects/Selected-2/dataset/validation'
test_dir = 'C:/Users/Lenovo/Desktop/Projects/Selected-2/dataset/evaluation'

# define the data augmentation strategy
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

# set the batch size and image size
batch_size = 32
img_height = 224
img_width = 224

# generate the training, validation, and testing sets
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(img_height, img_width),
                                               batch_size=batch_size,
                                               class_mode='categorical')

val_data = val_datagen.flow_from_directory(val_dir,
                                           target_size=(img_height, img_width),
                                           batch_size=batch_size,
                                           class_mode='categorical')

test_data = val_datagen.flow_from_directory(test_dir,
                                             target_size=(img_height, img_width),
                                             batch_size=batch_size,
                                             class_mode='categorical')
nb_train_samples =train_data.samples
nb_validation_samples = val_data.samples
nb_test_samples = test_data.samples
classes = list(train_data.class_indices.keys())
print('Classes: '+str(classes))
num_classes  = len(classes)

# build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# compile the model
model.compile(optimizer=optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()]
              )

# train the model
epochs = 50
history = model.fit(train_data,
                    epochs=epochs,
                    verbose=1,
                    validation_data=val_data)

results = model.evaluate(test_data,verbose=0)
print(model.metrics_names)
print(results)
 
from tensorflow.keras.preprocessing import image
from tkinter import*
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import cv2
root=Tk()
b=K.window(root)   

# set the path for the test image
img_path=b.filename


# load the image and preprocess it
img = image.load_img(img_path, target_size=(img_height, img_width))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.

# make a prediction using the trained model
preds = model.predict(x)
class_idx = np.argmax(preds)

# get the class label of the predicted class
class_labels = list(train_data.class_indices.keys())
class_label = class_labels[class_idx]

b.out_label = class_label
print(f"Predicted class: {class_label}")
root.mainloop() 



