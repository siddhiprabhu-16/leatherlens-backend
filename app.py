from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import joblib
from flask_cors import CORS
CORS(app)

app = Flask(__name__)

# load model
model = joblib.load("leather_classifier.pkl")

# load class labels
classes = joblib.load("class_names.pkl")

def extract_features(img):

    img = img.resize((64,64))

    img = np.array(img)/255.0

    features = img.flatten()

    return features.reshape(1,-1)


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"]

    img = Image.open(file).convert("RGB")

    features = extract_features(img)

    prediction = model.predict(features)[0]

    label = classes[prediction]

    return jsonify({
        "prediction": label
    })


app.run(host="0.0.0.0", port=5000)
