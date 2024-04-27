from flask import Flask, request
import tensorflow as tf
import io
import secrets
import numpy as np
import os
import base64
import cv2
import json

app = Flask(__name__)
model = tf.keras.models.load_model("model.h5", compile=False)

import json
from datetime import date

open("stats", "a")

try:
    stats=json.load(open("stats").read())
except:
    stats={}
stats={}

def save():
    open("stats", "w").write(json.dumps(stats))

@app.route('/', methods=["POST"])
def main():
    global stats
    if request.get_json()["route"]=="/":
        image_data = request.get_json()["image"]
        image_data = base64.b64decode(image_data)
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128))
        img = img.reshape(1, 128, 128, 3)
        result = model.predict(img)
        # print(type(result))
        try:
            stats[str(date.today())]+=1
        except:
            stats[str(date.today())]=1
        save()
        return json.dumps([float(x) for x in result[0]])
    elif request.get_json()["route"]=="/stats":
        return json.dumps(stats)
    elif request.get_json()["route"]=="/stats_increment":
        try:
            stats[str(date.today())]+=1
        except:
            stats[str(date.today())]=1
        save()
        return "true"

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
