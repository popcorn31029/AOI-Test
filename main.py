import matplotlib
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
import keras
from keras.utils import np_utils
import random
from  PIL import Image
from keras.preprocessing import image
import csv
from keras.models import load_model
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

#load train data
TRAIN_DATA = "C:/Users/User/Desktop/temp/abc/ml_aoi/train_images"
filenames_image = os.listdir(TRAIN_DATA)
total = len(filenames_image)
train_data = np.array([])
train_label = np.array([])
train_files = filenames_image[0:6]
#filename = 0,1,2,3,4,5
for filename in train_files:
    path = TRAIN_DATA + '/' + filename
    subfile = os.listdir(path)
    for tr in subfile:
        train_label = np.append(train_label, int(filename[0]))
        path1 = os.path.join(path, tr)
        img = image.load_img(path1, target_size=(112, 112))
        img = np.mean(img, axis=2)
        train_data = np.append(train_data, img)

#random
mapIndexPosition = list(zip(train_data, train_label))
random.shuffle(mapIndexPosition)
a, b = zip(*mapIndexPosition)
train_data = np.array(a)
train_label = np.array(b)
print(train_data.shape)
print(train_label.shape)

#train data pre-processing
train_data4D = train_data.reshape(train_data.shape[0], 112, 112, 1).astype('float32')
train_data_nor = train_data4D / 255
train_label_onehot = np_utils.to_categorical(train_label)
print(train_data4D.shape)

#build CNN model and training
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', input_shape=(112,112,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same', input_shape=(112,112,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
train_history=model.fit(x=train_data_nor, y=train_label_onehot, validation_split=0.1, epochs=100, batch_size=32, verbose=1)

#show training process
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel(train_acc)
    plt.xlabel('Epoch')
    plt.legend([train_acc, test_acc], loc='upper left')
    plt.show()
show_train_history('accuracy', 'val_accuracy')
show_train_history('loss','val_loss')

#prediction
TEST_DATA = "C:/Users/User/Desktop/temp/abc/ml_aoi/test_images"
filenames_image = os.listdir(TEST_DATA)
path = TEST_DATA

with open('test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    fieldNames = ['ID', 'Label']
    writer = csv.DictWriter(csvfile, fieldNames)
    writer.writeheader()
    
    for te in filenames_image:
        path1 = os.path.join(path, te)
        img = image.load_img(path1, target_size=(112, 112))
        img = np.mean(img, axis=2)
        test_data4D = img.reshape(1, 112, 112, 1).astype('float32')
        test_data_nor = test_data4D / 255
        prediction=model.predict_classes(test_data_nor)
        writer.writerow({'ID':te,'Label':prediction[0]})

model.summary()
