from typing import re

import pandas as pd
import numpy as np
import os
import keras
import keras.backend as K
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNetV2
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Conv1D,Flatten,MaxPooling1D,Bidirectional,Reshape,LSTM,Dropout,Embedding,TimeDistributed,MaxPool2D
from sklearn.metrics import confusion_matrix,plot_confusion_matrix


model = Sequential()

model.add((Convolution2D(32, (3, 3), input_shape=(50, 50, 3), padding='valid', activation= 'relu')) )

model.add((MaxPooling2D(pool_size=(2, 2))))

model.add((Convolution2D(48, (3, 3), padding='valid',activation= 'relu')))
model.add((MaxPooling2D(pool_size=(2, 2))))

model.add((Convolution2D(64, (3, 3), padding='valid',activation= 'relu')))
model.add((MaxPooling2D(pool_size=(2, 2))))

model.add((Convolution2D(96, (3, 3), padding='valid',activation= 'relu')))
model.add((MaxPooling2D(pool_size=(2, 2))))

model.add((Flatten()))
model.add(Dense(256))
model.add(Dropout(0.25))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))
model.compile(optimizer='adam', loss='mse')
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
label = pd.read_csv("AAPL.csv", delimiter=',', names=['id', 'y_val'],header= 0)
label_t = pd.read_csv("AAPL_t.csv", delimiter=',', names=['id', 'y_val'],header= 0)
print(label)
train_generator=train_datagen.flow_from_dataframe(dataframe = label , directory='Train', x_col="id", y_col="y_val", has_ext=True,
                                              class_mode="other", target_size=(50, 50 ),
                                              batch_size=32,shuffle=True)
test_generator=test_datagen.flow_from_dataframe(dataframe = label_t , directory='Test', x_col="id", y_col="y_val", has_ext=True,
                                          class_mode="other", target_size=(50, 50 ),
                                              batch_size=32,shuffle=False)


model.compile(optimizer='Adam',loss='mse')
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train=train_generator.n//train_generator.batch_size
step_size_test=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,validation_data=test_generator,validation_steps=step_size_test,
                   epochs=200)
