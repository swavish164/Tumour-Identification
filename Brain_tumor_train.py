import cv2
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import utils
from tensorflow.keras.utils import normalize 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Activation,Dropout,Flatten,Dense
from tensorflow.keras.utils import to_categorical

#from sklearn.model_selection import train_test_split
image_directory = "Brain Tumor Data Set/Brain Tumor Data Set/"
#image_directory = 'Brain Tumor Data Set\Brain Tumor Data Set\Brain Tumor'
dataset = []
label = []
tumor_images = os.listdir(image_directory+'Brain Tumor')
healthy_images = os.listdir(image_directory+'Healthy')
input_size = 64

def preprocess(images,tag):
  for i, image in enumerate(images):
    if(image.split('.')[1]=='jpg' or image.split('.')[1]=='JPG'):
      image_path = image_directory+tag+image
      image_array = cv2.imread(image_path)
      image_array = Image.fromarray(image_array,'RGB')
      image_array = image_array.resize((input_size,input_size))
      dataset.append(np.array(image_array))
      if(image == 'Brain Tumor'):
        label.append(1)
      else:
        label.append(0)

preprocess(tumor_images,'Brain Tumor/')
preprocess(healthy_images,'Healthy/')
dataset = np.array(dataset)
label = np.array(label)
print(len(dataset))

x_train = dataset[int(len(dataset)*0.2):]
x_test = dataset[:int(len(dataset)*0.2)]
y_train = label[int(len(label)*0.2):]
y_test = label[:int(len(label)*0.2)]

x_train = normalize(x_train,axis = 1)
x_test = normalize(x_test,axis = 1)

#y_train = to_categorical(y_train,num_classes = 2)
#y_test = to_categorical(y_test, num_classes = 2)


model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(input_size,input_size,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model = Sequential()
model.add(Conv2D(32,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model = Sequential()
model.add(Conv2D(64,(3,3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1)) # using binary classification so dense value of 1
model.add(Activation('sigmoid'))


model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
#model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',metrics = ['accuracy'])
model.fit(x_train,y_train,batch_size = 16,verbose = 1,epochs = 5,validation_data= (x_test,y_test),shuffle = False)
model.save('Brain_Tumor10Epochs.keras')