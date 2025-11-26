from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = joblib.load("house_price_model_minimal.joblib")

NUMERIC_FEATURES = ["land_size_m2", "building_size_m2", "bedrooms", "bathrooms", "building_age"]
CATEGORICAL_FEATURES = ["district", "city", "property_type", "certificate", "furnishing"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    for f in ALL_FEATURES:
        if f not in data:
            return jsonify({"error": f"Missing field: {f}"}), 400

    df = pd.DataFrame([data])
    pred_log = model.predict(df)[0]
    pred_price = np.expm1(pred_log) 

    return jsonify({
        "predicted_price": float(pred_price),
        "formatted_price": f"Rp {pred_price:,.0f}"
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
