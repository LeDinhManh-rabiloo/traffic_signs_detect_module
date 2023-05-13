import tensorflow as tf
import cv2
from flask_restful import Resource, Api
from PIL import Image


class LoadModel():
    def __init__(self, model, X_test):
        self.model = model
        self.X_test = X_test

    def predict(self):
        new_model = tf.keras.models.load_model(self.model)
        pre = new_model.predict_classes(self.X_test)
        return pre
