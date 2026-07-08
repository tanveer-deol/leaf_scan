from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

IMG_SIZE = 256
CONFIDENCE_THRESHOLD = 0.70

model = tf.keras.models.load_model("leaf_disease_model_final.h5")

class_names = [
    "Cassava_Bacterial Blight (CBB)",
    "Cassava_Brown Streak Disease (CBSD)",
    "Cassava_Green Mottle (CGM)",
    "Cassava_Healthy",
    "Cassava_Mosaic Disease (CMD)",
    "Rice_BrownSpot",
    "Rice_Healthy",
    "Rice_Hispa",
    "Rice_LeafBlast",
    "apple_apple scab",
    "apple_black rot",
    "apple_cedar apple rust",
    "apple_healthy",
    "cherry (including sour)_healthy",
    "cherry (including sour)_powdery mildew",
    "corn (maize)_cercospora leaf spot gray leaf spot",
    "corn (maize)_common rust",
    "corn (maize)_healthy",
    "corn (maize)_northern leaf blight",
    "grape_black rot",
    "grape_esca (black measles)",
    "grape_healthy",
    "grape_leaf blight (isariopsis leaf spot)",
    "orange_haunglongbing (citrus greening)",
    "peach_bacterial spot",
    "peach_healthy",
    "pepper, bell_bacterial spot",
    "pepper, bell_healthy",
    "potato_early blight",
    "potato_healthy",
    "potato_late blight",
    "squash_powdery mildew",
    "strawberry_healthy",
    "strawberry_leaf scorch",
    "tomato_bacterial spot",
    "tomato_early blight",
    "tomato_healthy",
    "tomato_late blight",
    "tomato_leaf mold",
    "tomato_septoria leaf spot",
    "tomato_spider mites two-spotted spider mite",
    "tomato_target spot",
    "tomato_tomato mosaic virus",
    "tomato_tomato yellow leaf curl virus"
]


def predict(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    image = np.expand_dims(image, axis=0)

    image = tf.keras.applications.efficientnet.preprocess_input(image)

    prediction = model.predict(image, verbose=0)

    idx = np.argmax(prediction)
    confidence = float(np.max(prediction))
    if confidence < CONFIDENCE_THRESHOLD:
        return "Uncertain — please retake the photo with a clear leaf image.", confidence
    return class_names[idx], confidence


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/learn")
def learn():
    return render_template("learn.html")

@app.route("/predict", methods=["POST"])
def predict_api():

    file = request.files["image"]

    image_bytes = np.frombuffer(file.read(), np.uint8)

    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    disease, confidence = predict(image)

    return jsonify({
        "prediction": disease,
        "confidence": round(confidence * 100, 2)
    })


import os

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )