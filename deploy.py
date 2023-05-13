from flask import Flask, flash, request, redirect, url_for
from flask_restful import Resource, Api
import tensorflow as tf
import numpy as np
from model.LoadModel import LoadModel
from werkzeug.utils import secure_filename
import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2

UPLOAD_FOLDER = 'data/Test/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
@app.route('/')
def index():
    return "<h1>Hello</h1>"


app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
api = Api(app)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def processData():
    # Test model
    height = 30
    width = 30
    y_test = pd.read_csv("data/Test.csv")
    labels = y_test['Path'].as_matrix()
    y_test = y_test['ClassId'].values

    data = []

    for f in labels:
        image = cv2.imread('data/Test/' + f.replace('Test/', ''))
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((height, width))
        data.append(np.array(size_image))

    X_test = np.array(data)
    X_test = X_test.astype('float32') / 255
    return X_test


class UploadWavAPI(Resource):
    def post(self):
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                x_test = processData()
                load_model = LoadModel('model/final_model.h5', x_test)
                return load_model.predict()
        return 'Oki'


api.add_resource(UploadWavAPI, '/upload')

if __name__ == '__main__':
    app.debug = True
    app.run()
