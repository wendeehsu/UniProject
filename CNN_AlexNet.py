import keras
from keras.models import Sequential
from keras.layers import Activation,Dropout
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

train_path = 'Image/pancake1022/train'
target_size = (224,224)
batch_size = 20

# ImageDataGenerator()可以做一些影像處理的動作 
datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2)

# 以 batch 的方式讀取資料
train_batches = datagen.flow_from_directory(
        train_path,  
        target_size = target_size,  
        batch_size = batch_size,
        color_mode = 'grayscale',
        classes = ['Ok','NotYet'],
        subset='training')  

valid_batches = datagen.flow_from_directory(
        train_path,
        target_size = target_size,
        batch_size = batch_size,
        color_mode = 'grayscale',
        classes = ['Ok','NotYet'],
        subset='validation')

nb_train_samples = 1040
nb_validation_samples = 260

# build model
model=Sequential()
model.add(Conv2D(96, kernel_size=(3,3), strides=(2,2), input_shape=(224,224,1), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(256, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Conv2D(384, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu'))
model.add(Conv2D(256, kernel_size=(3,3), strides=(2,2), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(2, activation='sigmoid'))

model.summary()

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, 
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data = valid_batches, 
                    validation_steps = nb_validation_samples // batch_size, 
                    epochs = 10)

model.save("cnn")

# ========  testing  =========

test_path = "Image/pancake1022/test"
predict_batches = datagen.flow_from_directory(
        test_path,
        target_size = target_size,
        batch_size = 1,
        color_mode = 'grayscale',
        classes = ['Ok','NotYet'])

filenames = predict_batches.filenames
nb_samples = len(filenames)
predict = model.predict_generator(predict_batches,steps = nb_samples)
y_pred = np.argmax(predict, axis=1)
print('Confusion Matrix')
print(confusion_matrix(predict_batches.classes, y_pred))
target_names = ['Ok','NotYet']
print(classification_report(predict_batches.classes, y_pred, target_names=target_names))

import numpy as np
import cv2
import matplotlib.pyplot as plt

def GetFN(real,predicted):
    names = []
    for i in range(len(real)):
        if real[i] == 0 and predicted[i] == 1:
            names.append(filenames[i])
    return names
            
def GetFP(real,predicted):
    names = []
    for i in range(len(real)):
        if real[i] == 1 and predicted[i] == 0:
            names.append(filenames[i])
    return names

def GetTN(real,predicted):
    names = []
    for i in range(len(real)):
        if real[i] == 1 and predicted[i] == 1:
            names.append(filenames[i])
    return names
            
def GetTP(real,predicted):
    names = []
    for i in range(len(real)):
        if real[i] == 0 and predicted[i] == 0:
            names.append(filenames[i])
    return names

def show_images(images):
    fig=plt.figure(figsize=(60, 60))
    columns = 4
    rows = (len(images) // 4) + 1
    for i in range(1, len(images) +1):
        img = cv2.imread("Image/pancake1012/"+images[i-1])
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()

print(GetFN(predict_batches.classes, y_pred))
