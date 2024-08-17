#import cv2
import os
from PIL import Image
import numpy as np
#from sklearn.model_selection import train_test_split
image_directory = 'Brain Tumor Data Set/'
dataset = []
label = []
tumor_images = os.listdir(image_directory+'Brain Tumor')
healthy_images = os.listdir(image_directory+'Healthy')

def preprocess(images,tag):
  for i, image in enumerate(images):
    if(image.split('.')[1]=='jpg' or image.split('.')[1]=='JPG'):
      image_path = image_directory+tag+image
      image_array = cv2.imread(image_path)
      image_array = Image.fromarray(image_array,'RGB')
      image_array = image_array.resize((64,64))
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

x_train = dataset[len(dataset)*0.2:]
x_test = dataset[:len(dataset)*0.2]
y_train = label[len(label)*0.2:]
y_test = label[:len(label)*0.2]
