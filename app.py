from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from PIL import Image
import io

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

# Load CNN once (very important)
cnn_model = EfficientNetB0(weights="imagenet", include_top=False, pooling="avg")

# ------------------- FEATURE FUNCTIONS -------------------

def preprocess_texture(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img

def extract_lbp(img):
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(),
                           bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_glcm(img):
    glcm = graycomatrix(img, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return np.array([contrast, correlation, energy, homogeneity])

def extract_gabor(img):
    features = []
    for theta in np.arange(0, np.pi, np.pi / 4):
        kernel = cv2.getGaborKernel((9, 9), 4.0, theta,
                                     10.0, 0.5, 0, ktype=cv2.CV_32F)
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
        features.append(fimg.mean())
        features.append(fimg.var())
    return np.array(features)

def extract_cnn(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_rgb = cv2.resize(img_rgb, (224, 224))
    img_rgb = img_to_array(img_rgb)
    img_rgb = np.expand_dims(img_rgb, axis=0)
    img_rgb = preprocess_input(img_rgb)
    features = cnn_model.predict(img_rgb, verbose=0)
    return features.flatten()

def extract_features(img):
    img_proc = preprocess_texture(img)
    return np.concatenate([
        extract_lbp(img_proc),
        extract_glcm(img_proc),
        extract_gabor(img_proc),
        extract_cnn(img_proc)
    ])

# ------------------- PREDICT ENDPOINT -------------------

@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # Load image
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((224, 224))
    img = np.array(img)

    # Extract SAME training features
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

# ------------------- RUN -------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

