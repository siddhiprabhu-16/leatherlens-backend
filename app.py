from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from PIL import Image
import io
import os

# Texture / ML imports
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------- APP INIT -------------------

app = Flask(__name__)
CORS(app)

# ------------------- LOAD MODELS -------------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
classes = joblib.load("classes.pkl")

# Load CNN once
cnn_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# ------------------- FEATURE FUNCTIONS -------------------

def preprocess_texture(img):
    return cv2.bilateralFilter(img, 9, 75, 75)


def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []
    for r in [1, 2, 3]:
        n = 8 * r
        lbp = local_binary_pattern(gray, n, r, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n + 2,
            range=(0, n + 2)
        )
        hist = hist / (hist.sum() + 1e-6)
        feats.extend(hist)
    return np.array(feats)


def extract_glcm(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(
        gray,
        distances=[1, 2, 3],
        angles=[0, np.pi/4, np.pi/2],
        levels=256,
        symmetric=True,
        normed=True
    )
    props = ["contrast", "dissimilarity", "homogeneity",
             "energy", "correlation", "ASM"]
    feats = []
    for p in props:
        feats.extend(graycoprops(glcm, p).flatten())
    return np.array(feats)


def extract_gabor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []
    for theta in np.arange(0, np.pi, np.pi/6):
        for freq in [0.1, 0.2]:
            kernel = cv2.getGaborKernel(
                (21, 21),
                5,
                theta,
                10 * freq,
                0.5,
                0
            )
            f = cv2.filter2D(gray, cv2.CV_32F, kernel)
            feats.append(f.mean())
            feats.append(f.var())
    return np.array(feats)

# ------------------- ROOT ROUTE -------------------

@app.route("/")
def home():
    return jsonify({
        "status": "LeatherLens Backend Running 🚀"
    })

# ------------------- PREDICT ENDPOINT -------------------

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
        # Load image safely
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        img = img.resize((224, 224))
        img = np.array(img)

        # Extract features
        features = extract_features(img)
        features = features.reshape(1, -1)

        # Apply scaler and PCA
        features = scaler.transform(features)
        features = pca.transform(features)

        # Predict
        prediction = model.predict_proba(features)[0]
        class_index = np.argmax(prediction)

        return jsonify({
            "prediction": classes[class_index],
            "confidence": float(prediction[class_index])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- RUN -------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

