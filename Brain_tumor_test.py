import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '3'

model = load_model('Brain_Tumor10Epochs.keras')
input_size = 64

image= cv2.imread('Brain Tumor Data Set\Brain Tumor Data Set\Healthy\\Not Cancer  (1).jpg')
img = Image.fromarray(image)

img = img.resize((input_size,input_size))
img= np.array(img)
input_imgs = np.expand_dims(img,axis=0,)

predictions = model.predict(input_imgs)
result = np.where(predictions > 0.5, 1,0)
print(result)