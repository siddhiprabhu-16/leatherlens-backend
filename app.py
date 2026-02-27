from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from PIL import Image
import io
import os
import cv2

from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# ------------------- APP INIT -------------------

app = Flask(__name__)
CORS(app)

# ------------------- LOAD MODELS -------------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
classes = joblib.load("classes.pkl")

# ------------------- LOAD CNN (IDENTICAL TO NOTEBOOK) -------------------

base = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)
base.trainable = False
x = GlobalAveragePooling2D()(base.output)
cnn_model = Model(base.input, x)

# ------------------- FEATURE FUNCTIONS -------------------

def preprocess_texture(img):
    # EXACTLY like notebook
    return cv2.bilateralFilter(img, 9, 75, 75)


def extract_lbp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []

    for r in [1, 2, 3]:
        n = 8 * r
        lbp = local_binary_pattern(gray, n, r, "uniform")
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
    # Ensure distances and angles match the training exactly
    glcm = graycomatrix(gray, [1, 2, 3], [0, np.pi/4, np.pi/2], 256, True, True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    
    feats = []
    for p in props:
        # .flatten() ensures we get all 9 values (3x3) per property
        feats.extend(graycoprops(glcm, p).flatten())
    return np.array(feats)


def extract_gabor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feats = []

    for theta in np.arange(0, np.pi, np.pi/6):  # 6 orientations
        for freq in [0.1, 0.2]:                 # EXACT match
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


def extract_cnn(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, 0)
    features = cnn_model.predict(img, verbose=0).flatten()
    # print(f"CNN Feature length: {len(features)}") # Should be 1280
    return features


def extract_features(img):
    img_proc = preprocess_texture(img)
    
    f_lbp = extract_lbp(img_proc)
    f_glcm = extract_glcm(img_proc)
    f_gabor = extract_gabor(img_proc)
    f_cnn = extract_cnn(img_proc)
    
    # This will show up in your Flask logs/terminal
    print(f"DEBUG: LBP={len(f_lbp)}, GLCM={len(f_glcm)}, Gabor={len(f_gabor)}, CNN={len(f_cnn)}")
    print(f"DEBUG: Total={len(f_lbp) + len(f_glcm) + len(f_gabor) + len(f_cnn)}")
    
    return np.concatenate([f_lbp, f_glcm, f_gabor, f_cnn])

# FINAL SAFETY CHECK
    if len(full_features) != 1412:
        print(f"⚠️ Warning: Feature mismatch! Got {len(full_features)}, need 1412")
        # Pad with zeros to prevent StandardScaler crash
        if len(full_features) < 1412:
            full_features = np.pad(full_features, (0, 1412 - len(full_features)))
            
    return full_features[:1412]


# ------------------- ROOT ROUTE -------------------

@app.route("/")
def home():
    return jsonify({"status": "LeatherLens Backend Running 🚀"})


# ------------------- PREDICT ENDPOINT -------------------

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    try:
       file_bytes = np.frombuffer(file.read(), np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))

        features = extract_features(img)
        features = features.reshape(1, -1)
print("Feature length BEFORE scaler:", features.shape)
        # Apply SAME scaler + PCA as training
        features = scaler.transform(features)
        features = pca.transform(features)

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




