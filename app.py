import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

app = Flask(__name__)

_model = None
_feature_cols = None

CONTRACT_MAP = {"Month-to-month": 0, "One year": 1, "Two year": 2}
INTERNET_MAP = {"DSL": 0, "Fiber optic": 1, "No": 2}
TECH_SUPPORT_MAP = {"No": 0, "No internet service": 1, "Yes": 2}


def load_artifacts():
    global _model, _feature_cols
    if _model is None:
        _model = joblib.load(os.path.join(MODEL_DIR, "churn_model.pkl"))
        _feature_cols = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))


def encode_input(data: dict) -> np.ndarray:
    load_artifacts()
    row = {}
    for col in _feature_cols:
        val = data.get(col)
        if col == "contract":
            row[col] = CONTRACT_MAP.get(val, 0)
        elif col == "internet_service":
            row[col] = INTERNET_MAP.get(val, 0)
        elif col == "tech_support":
            row[col] = TECH_SUPPORT_MAP.get(val, 0)
        else:
            row[col] = float(val) if val is not None else 0.0
    return np.array([[row[c] for c in _feature_cols]])


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict churn probability for a customer"
        },
        "example_curl": (
            'curl -X POST http://localhost:5000/predict '
            '-H "Content-Type: application/json" '
            '-d \'{"tenure": 3, "monthly_charges": 89.5, "total_charges": 268.5, '
            '"contract": "Month-to-month", "internet_service": "Fiber optic", '
            '"tech_support": "No", "senior_citizen": 0, "dependents": 0, '
            '"num_services": 3, "paperless_billing": 1}\''
        ),
        "feature_info": {
            "tenure": "int, months as customer (1-72)",
            "monthly_charges": "float, monthly bill amount",
            "total_charges": "float, total amount charged",
            "contract": "Month-to-month | One year | Two year",
            "internet_service": "DSL | Fiber optic | No",
            "tech_support": "Yes | No | No internet service",
            "senior_citizen": "0 or 1",
            "dependents": "0 or 1",
            "num_services": "int, number of services (1-7)",
            "paperless_billing": "0 or 1"
        }
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    required = ["tenure", "monthly_charges", "contract", "internet_service", "tech_support"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {missing}"}), 400

    try:
        X = encode_input(data)
        load_artifacts()
        pred = int(_model.predict(X)[0])
        proba = float(_model.predict_proba(X)[0][1])

        if proba >= 0.65:
            risk_level = "High"
        elif proba >= 0.35:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return jsonify({
            "prediction": "Churn" if pred == 1 else "Retain",
            "churn_probability": round(proba, 4),
            "risk_level": risk_level,
        })

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    load_artifacts()
    print("Churn Prediction API running on http://localhost:5000")
    app.run(host="0.0.0.0", port=8080, debug=True)
