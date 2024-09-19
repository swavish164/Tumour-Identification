import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
import os
import json
import tensorflow as tf
from flask import Flask, request,jsonify,render_template
from werkzeug.utils import secure_filename

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

model = load_model('Brain_Tumor10Epochs.keras')

def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"
    
def get_result(img):
    image = cv2.imread(img)
    image = Image.fromarray(image,'RGB')
    image = image.resize((64,64))
    image = np.array(image)
    input_img = np.expand_dims(image,axis = 0)
    result = model.predict(input_img)
    print(result)
    nonCancerAccuracy = result[0][0]
    print(nonCancerAccuracy)
    if(nonCancerAccuracy > 0.5):
        accuracy = nonCancerAccuracy
        result = 1
    else:
        accuracy = 1- nonCancerAccuracy
        result = 0
    return result

@app.route('/',methods = ['GET'])
def index():
    return render_template('index.html')


@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'uploads',secure_filename(f.filename))
        f.save(file_path)
        value = get_result(file_path)
        result = get_className(value)
        return (result)
    return None

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))