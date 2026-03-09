from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load("leather_classifier.pkl")
classes = joblib.load("class_names.pkl")


def extract_features(img):

    img = img.resize((64,64))
    img = np.array(img)/255.0
    features = img.flatten()

    return features.reshape(1,-1)


@app.route("/")
def home():
    return "LeatherLens API running"


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error":"No image uploaded"})

    file = request.files["image"]
    img = Image.open(file).convert("RGB")

    features = extract_features(img)

    prediction = model.predict(features)[0]

    label = classes[prediction]

    return jsonify({
        "prediction": label
    })
