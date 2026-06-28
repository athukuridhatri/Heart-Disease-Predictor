"""
app.py
======
Flask backend for the Heart Disease Risk Predictor.

Routes:
  GET  /              — Main page (prediction form)
  POST /predict       — JSON prediction endpoint
  GET  /model-info    — Model metadata and evaluation metrics
"""

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load Saved Model Artefact
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    _data = pickle.load(f)

model     = _data["model"]
scaler    = _data["scaler"]
accuracy  = _data["accuracy"]
precision = _data["precision"]
recall    = _data["recall"]
f1_score  = _data["f1_score"]
roc_auc   = _data["roc_auc"]
conf_matrix = _data["confusion_matrix"]

# ---------------------------------------------------------------------------
# Feature Definitions
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

FEATURE_LABELS = [
    "Age", "Sex", "Chest Pain Type", "Resting Blood Pressure",
    "Cholesterol", "Fasting Blood Sugar", "Resting ECG",
    "Max Heart Rate", "Exercise Angina", "ST Depression",
    "ST Slope", "Major Vessels", "Thalassemia",
]

# Valid value ranges for server-side input validation
FEATURE_RANGES = {
    "age":      (20,  80),
    "sex":      (0,   1),
    "cp":       (0,   3),
    "trestbps": (80,  220),
    "chol":     (100, 700),
    "fbs":      (0,   1),
    "restecg":  (0,   2),
    "thalach":  (60,  220),
    "exang":    (0,   1),
    "oldpeak":  (0.0, 7.0),
    "slope":    (0,   2),
    "ca":       (0,   3),
    "thal":     (1,   3),
}

# Short recommendations returned with each prediction
RECOMMENDATIONS = {
    "high": (
        "Consult a cardiologist promptly. Adopt a heart-healthy diet, "
        "engage in moderate physical activity as advised, and monitor "
        "blood pressure and cholesterol levels regularly."
    ),
    "low": (
        "Maintain your healthy lifestyle. Exercise regularly, eat a "
        "balanced diet, avoid smoking, and schedule routine check-ups "
        "with your physician."
    ),
}

# ---------------------------------------------------------------------------
# Helper: Confidence Label
# ---------------------------------------------------------------------------
def get_confidence_label(probability: float) -> str:
    """Return a human-readable confidence label for a given probability."""
    if probability >= 0.85:
        return "Very High"
    if probability >= 0.70:
        return "High"
    if probability >= 0.55:
        return "Moderate"
    return "Low"


# ---------------------------------------------------------------------------
# Helper: Validate Input
# ---------------------------------------------------------------------------
def validate_input(data: dict) -> list[str]:
    """
    Validate that all required features are present and within expected ranges.
    Returns a list of error messages (empty = valid).
    """
    errors = []

    for feature in FEATURE_NAMES:
        if feature not in data:
            errors.append(f"Missing field: '{feature}'.")
            continue

        try:
            value = float(data[feature])
        except (TypeError, ValueError):
            errors.append(f"Invalid value for '{feature}': must be a number.")
            continue

        low, high = FEATURE_RANGES[feature]
        if not (low <= value <= high):
            errors.append(
                f"'{feature}' = {value} is out of the expected range [{low}, {high}]."
            )

    return errors


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Render the main prediction form."""
    return render_template(
        "index.html",
        accuracy=round(accuracy * 100, 2),
        precision=round(precision * 100, 2),
        recall=round(recall * 100, 2),
        f1=round(f1_score * 100, 2),
        roc_auc=round(roc_auc * 100, 2),
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept JSON with 13 clinical features and return prediction results.

    Returns:
        JSON with prediction, probabilities, confidence, recommendation,
        and top contributing features.
    """
    raw = request.get_json(silent=True)

    # Guard: missing or malformed body
    if not raw:
        return jsonify({"error": "No JSON body received. Send Content-Type: application/json."}), 400

    # Validate input fields
    errors = validate_input(raw)
    if errors:
        return jsonify({"error": " | ".join(errors)}), 400

    # Build feature vector
    features = np.array([float(raw[f]) for f in FEATURE_NAMES]).reshape(1, -1)

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    prediction   = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]
    prob_disease  = round(float(probabilities[1]) * 100, 1)
    prob_no_disease = round(float(probabilities[0]) * 100, 1)

    # Confidence is based on the winning class probability
    winning_prob    = max(probabilities)
    confidence      = get_confidence_label(winning_prob)

    # Top-5 contributing features (|scaled_value × coefficient|)
    coefs = model.coef_[0]
    contributions = []
    for label, raw_val, scaled_val, coef in zip(
        FEATURE_LABELS, features[0], features_scaled[0], coefs
    ):
        contributions.append({
            "name":        label,
            "value":       round(float(raw_val), 1),
            "contribution": round(float(scaled_val * coef), 4),
            "abs_contrib": round(abs(float(scaled_val * coef)), 4),
        })

    contributions.sort(key=lambda x: x["abs_contrib"], reverse=True)

    return jsonify({
        "prediction":           prediction,
        "risk_label":           "High Risk" if prediction == 1 else "Low Risk",
        "probability_disease":  prob_disease,
        "probability_no_disease": prob_no_disease,
        "confidence":           confidence,
        "recommendation":       RECOMMENDATIONS["high" if prediction == 1 else "low"],
        "top_features":         contributions[:5],
        "message":              "Heart disease detected" if prediction == 1 else "No heart disease detected",
    })


@app.route("/model-info")
def model_info():
    """Return model metadata and evaluation metrics as JSON."""
    return jsonify({
        "algorithm":      "Logistic Regression",
        "dataset":        "UCI Cleveland Heart Disease (278 clean samples)",
        "features":       FEATURE_LABELS,
        "train_test_split": "80% / 20% (stratified)",
        "metrics": {
            "accuracy":  round(accuracy * 100, 2),
            "precision": round(precision * 100, 2),
            "recall":    round(recall * 100, 2),
            "f1_score":  round(f1_score * 100, 2),
            "roc_auc":   round(roc_auc * 100, 2),
        },
        "confusion_matrix": conf_matrix,
    })


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 55)
    print("  Heart Disease Predictor — Flask Server")
    print(f"  Accuracy  : {accuracy:.2%}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")
    print("  Open      : http://127.0.0.1:5000")
    print("=" * 55)
    app.run(debug=True)
