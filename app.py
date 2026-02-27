from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model files
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
classes = joblib.load("classes.pkl")

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
   img = Image.open(io.BytesIO(file.read()))
img = img.resize((224, 224))
img = np.array(img)

features = extract_features(img)
features = features.reshape(1, -1)

features = scaler.transform(features)
features = pca.transform(features)

    prediction = model.predict_proba(img)[0]
    class_index = np.argmax(prediction)

    return jsonify({
        "class": classes[class_index],
        "confidence": float(prediction[class_index])
    })

if __name__ == "__main__":

    app.run(host="0.0.0.0", port=8080)
