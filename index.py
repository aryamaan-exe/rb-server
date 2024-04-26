from flask import Flask, request
import tensorflow as tf
import io
import secrets
import numpy as np
import os
import base64
import cv2

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5", compile=False)

@app.route('/', methods=["POST"])
def main():
    image_data = request.files["image"].read()
    image_data = base64.b64decode(image_data)
    arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    img = img.reshape(1, 128, 128, 3)
    result = model.predict(img)
    return str(result)
