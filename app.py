from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import joblib
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Load model and class labels
model = joblib.load("leather_classifier.pkl")
classes = joblib.load("class_names.pkl")


def extract_features(img):

    img = img.resize((64, 64))
    img = np.array(img) / 255.0

    features = img.flatten()

    return features.reshape(1, -1)


@app.route("/")
def home():
    return "LeatherLens API running"


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:

        img = Image.open(BytesIO(file.read())).convert("RGB")

        features = extract_features(img)

        prediction = model.predict(features)[0]

        probabilities = model.predict_proba(features)

        confidence = float(max(probabilities[0]))

        label = classes[prediction]

        return jsonify({
            "prediction": label,
            "confidence": confidence
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
