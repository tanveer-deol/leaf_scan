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

disease_info = {

    "Cassava_Bacterial Blight (CBB)": {
        "cause": "Bacterial disease caused by Xanthomonas axonopodis pv. manihotis.",
        "prevention": "Use disease-free cuttings, practice crop rotation, and remove infected plants.",
        "treatment": "No effective chemical cure. Remove infected plants and use resistant cassava varieties."
    },

    "Cassava_Brown Streak Disease (CBSD)": {
        "cause": "Viral disease caused by Cassava brown streak viruses (CBSV and UCBSV).",
        "prevention": "Plant certified disease-free cuttings and control whitefly populations.",
        "treatment": "Remove infected plants and grow resistant or tolerant varieties."
    },

    "Cassava_Green Mottle (CGM)": {
        "cause": "Viral disease causing green mottling and leaf distortion.",
        "prevention": "Use healthy planting material and control insect vectors.",
        "treatment": "There is no cure; remove infected plants and plant resistant varieties."
    },

    "Cassava_Healthy": {
        "cause": "No disease detected.",
        "prevention": "Maintain proper nutrition, irrigation, and pest management.",
        "treatment": "No treatment required."
    },

    "Cassava_Mosaic Disease (CMD)": {
        "cause": "Viral disease caused by Cassava mosaic begomoviruses.",
        "prevention": "Use resistant varieties, healthy cuttings, and control whiteflies.",
        "treatment": "Remove infected plants and replant with certified healthy cuttings."
    },

    "Rice_BrownSpot": {
        "cause": "Fungal disease caused by Bipolaris oryzae.",
        "prevention": "Use certified seeds, balanced fertilization, and proper irrigation.",
        "treatment": "Apply appropriate fungicides and improve field nutrition."
    },

    "Rice_Healthy": {
        "cause": "No disease detected.",
        "prevention": "Maintain proper irrigation, fertilization, and weed control.",
        "treatment": "No treatment required."
    },

    "Rice_Hispa": {
        "cause": "Damage caused by the rice hispa insect (Dicladispa armigera).",
        "prevention": "Monitor fields regularly and maintain field sanitation.",
        "treatment": "Apply recommended insecticides if infestation is severe."
    },

    "Rice_LeafBlast": {
        "cause": "Fungal disease caused by Magnaporthe oryzae.",
        "prevention": "Use resistant varieties, balanced nitrogen application, and proper spacing.",
        "treatment": "Apply recommended fungicides at early stages."
    },

    "apple_apple scab": {
        "cause": "Fungal disease caused by Venturia inaequalis.",
        "prevention": "Prune trees, remove fallen leaves, and improve air circulation.",
        "treatment": "Apply recommended fungicides during the growing season."
    },

    "apple_black rot": {
        "cause": "Fungal disease caused by Botryosphaeria obtusa.",
        "prevention": "Remove infected fruit and branches and maintain orchard sanitation.",
        "treatment": "Prune infected areas and apply fungicides."
    },

    "apple_cedar apple rust": {
        "cause": "Fungal disease caused by Gymnosporangium juniperi-virginianae.",
        "prevention": "Remove nearby juniper hosts and plant resistant varieties.",
        "treatment": "Apply fungicides before symptoms become severe."
    },

    "apple_healthy": {
        "cause": "No disease detected.",
        "prevention": "Maintain proper pruning, irrigation, and nutrition.",
        "treatment": "No treatment required."
    },

    "cherry (including sour)_healthy": {
        "cause": "No disease detected.",
        "prevention": "Provide proper pruning, watering, and fertilization.",
        "treatment": "No treatment required."
    },

    "cherry (including sour)_powdery mildew": {
        "cause": "Fungal disease caused by Podosphaera species.",
        "prevention": "Ensure good air circulation and avoid excessive nitrogen.",
        "treatment": "Apply sulfur or other recommended fungicides."
    },

    "corn (maize)_cercospora leaf spot gray leaf spot": {
        "cause": "Fungal disease caused by Cercospora zeae-maydis.",
        "prevention": "Rotate crops, use resistant hybrids, and manage crop residue.",
        "treatment": "Apply fungicides when disease pressure is high."
    },

    "corn (maize)_common rust": {
        "cause": "Fungal disease caused by Puccinia sorghi.",
        "prevention": "Plant resistant hybrids and monitor fields regularly.",
        "treatment": "Apply fungicides if severe infection occurs."
    },

    "corn (maize)_healthy": {
        "cause": "No disease detected.",
        "prevention": "Maintain balanced fertilization and irrigation.",
        "treatment": "No treatment required."
    },

    "corn (maize)_northern leaf blight": {
        "cause": "Fungal disease caused by Exserohilum turcicum.",
        "prevention": "Use resistant hybrids and rotate crops.",
        "treatment": "Apply recommended fungicides."
    },

    "grape_black rot": {
        "cause": "Fungal disease caused by Guignardia bidwellii.",
        "prevention": "Prune vines, remove infected fruit, and improve airflow.",
        "treatment": "Apply fungicides during the growing season."
    },

    "grape_esca (black measles)": {
        "cause": "Complex fungal disease affecting grapevine wood.",
        "prevention": "Avoid pruning wounds and sanitize pruning tools.",
        "treatment": "Remove infected wood and maintain vine health."
    },

    "grape_healthy": {
        "cause": "No disease detected.",
        "prevention": "Maintain proper pruning and vineyard sanitation.",
        "treatment": "No treatment required."
    },

    "grape_leaf blight (isariopsis leaf spot)": {
        "cause": "Fungal disease caused by Pseudocercospora vitis.",
        "prevention": "Improve airflow and remove infected leaves.",
        "treatment": "Apply recommended fungicides."
    },

    "orange_haunglongbing (citrus greening)": {
        "cause": "Bacterial disease caused by Candidatus Liberibacter species and spread by psyllids.",
        "prevention": "Control psyllid vectors and use certified disease-free trees.",
        "treatment": "No cure exists; remove infected trees and manage insect vectors."
    },

    "peach_bacterial spot": {
        "cause": "Bacterial disease caused by Xanthomonas arboricola pv. pruni.",
        "prevention": "Plant resistant varieties and avoid overhead irrigation.",
        "treatment": "Apply copper-based bactericides when recommended."
    },

    "peach_healthy": {
        "cause": "No disease detected.",
        "prevention": "Maintain proper pruning and fertilization.",
        "treatment": "No treatment required."
    },

    "pepper, bell_bacterial spot": {
        "cause": "Bacterial disease caused by Xanthomonas species.",
        "prevention": "Use disease-free seeds, rotate crops, and avoid overhead watering.",
        "treatment": "Apply copper-based bactericides and remove infected plants."
    },

    "pepper, bell_healthy": {
        "cause": "No disease detected.",
        "prevention": "Maintain proper irrigation and nutrition.",
        "treatment": "No treatment required."
    },

    "potato_early blight": {
        "cause": "Fungal disease caused by Alternaria solani.",
        "prevention": "Rotate crops, avoid overhead watering, and remove infected foliage.",
        "treatment": "Apply recommended fungicides."
    },

    "potato_healthy": {
        "cause": "No disease detected.",
        "prevention": "Maintain proper irrigation and soil fertility.",
        "treatment": "No treatment required."
    },

    "potato_late blight": {
        "cause": "Disease caused by Phytophthora infestans.",
        "prevention": "Use certified seed potatoes and avoid prolonged leaf wetness.",
        "treatment": "Apply appropriate fungicides and remove infected plants."
    },

    "squash_powdery mildew": {
        "cause": "Fungal disease caused by Podosphaera xanthii and related fungi.",
        "prevention": "Ensure proper spacing and avoid excessive humidity.",
        "treatment": "Apply sulfur or other recommended fungicides."
    },

    "strawberry_healthy": {
        "cause": "No disease detected.",
        "prevention": "Maintain proper irrigation and remove weeds.",
        "treatment": "No treatment required."
    },

    "strawberry_leaf scorch": {
        "cause": "Fungal disease caused by Diplocarpon earlianum.",
        "prevention": "Remove infected leaves and improve air circulation.",
        "treatment": "Apply fungicides if necessary."
    },

    "tomato_bacterial spot": {
        "cause": "Bacterial disease caused by Xanthomonas species.",
        "prevention": "Use certified seeds, avoid overhead watering, and rotate crops.",
        "treatment": "Apply copper-based bactericides and remove infected plants."
    },

    "tomato_early blight": {
        "cause": "Fungal disease caused by Alternaria solani.",
        "prevention": "Avoid overhead watering, rotate crops, and remove infected leaves.",
        "treatment": "Apply a recommended fungicide and improve air circulation."
    },

    "tomato_healthy": {
        "cause": "No disease detected.",
        "prevention": "Maintain proper irrigation, fertilization, and pest management.",
        "treatment": "No treatment required."
    },

    "tomato_late blight": {
        "cause": "Disease caused by Phytophthora infestans.",
        "prevention": "Use resistant varieties, avoid prolonged leaf wetness, and ensure good airflow.",
        "treatment": "Apply recommended fungicides and remove infected plants."
    },

    "tomato_leaf mold": {
        "cause": "Fungal disease caused by Passalora fulva.",
        "prevention": "Reduce humidity and improve greenhouse ventilation.",
        "treatment": "Apply recommended fungicides."
    },

    "tomato_septoria leaf spot": {
        "cause": "Fungal disease caused by Septoria lycopersici.",
        "prevention": "Avoid overhead watering and remove infected leaves.",
        "treatment": "Apply fungicides and improve air circulation."
    },

    "tomato_spider mites two-spotted spider mite": {
        "cause": "Infestation by the two-spotted spider mite (Tetranychus urticae).",
        "prevention": "Maintain adequate humidity and monitor plants regularly.",
        "treatment": "Use miticides or introduce beneficial predatory mites."
    },

    "tomato_target spot": {
        "cause": "Fungal disease caused by Corynespora cassiicola.",
        "prevention": "Rotate crops and improve air circulation.",
        "treatment": "Apply appropriate fungicides."
    },

    "tomato_tomato mosaic virus": {
        "cause": "Viral disease caused by Tomato mosaic virus (ToMV).",
        "prevention": "Use virus-free seeds, disinfect tools, and avoid handling plants after tobacco use.",
        "treatment": "Remove infected plants; there is no chemical cure."
    },

    "tomato_tomato yellow leaf curl virus": {
        "cause": "Viral disease caused by Tomato yellow leaf curl virus (TYLCV), spread by whiteflies.",
        "prevention": "Control whiteflies and use resistant tomato varieties.",
        "treatment": "Remove infected plants and manage whitefly populations."
    }
}
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

    info = disease_info.get(
        disease,
    {
        "cause": "Information unavailable.",
        "prevention": "Consult an agricultural expert.",
        "treatment": "Follow local recommendations."
    }
)

    return jsonify({
    "prediction": disease,
    "confidence": round(confidence * 100, 2),
    "cause": info["cause"],
    "prevention": info["prevention"],
    "treatment": info["treatment"]
})




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)