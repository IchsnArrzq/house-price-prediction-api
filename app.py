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
LOCATION_COLUMNS = ["lat", "long", "district", "city", "price_in_rp", "title"]

try:
    location_df = (
        pd.read_csv("jabodetabek_house_price.csv", usecols=LOCATION_COLUMNS)
        .dropna(subset=["lat", "long", "price_in_rp"])
    )
    location_df["lat"] = location_df["lat"].astype(float)
    location_df["long"] = location_df["long"].astype(float)
    location_df["price_in_rp"] = location_df["price_in_rp"].astype(float)
except FileNotFoundError:
    # Keep running even if the CSV is missing so other endpoints keep working.
    location_df = pd.DataFrame(columns=LOCATION_COLUMNS)

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


@app.route("/api/locations", methods=["GET"])
def list_locations():
    """
    Returns latitude/longitude points for plotting on a map.
    Optional query params:
      - city: filter by city name (case insensitive, partial match allowed)
      - limit: max number of rows to return (defaults to 500, capped at 1000)
    """
    if location_df.empty:
        return jsonify({"error": "Location dataset unavailable"}), 500

    limit = request.args.get("limit", default=500, type=int)
    limit = max(1, min(limit or 500, 1000))

    city = request.args.get("city", type=str)
    filtered = location_df
    if city:
        filtered = filtered[filtered["city"].str.contains(city, case=False, na=False)]

    payload = filtered.head(limit).to_dict(orient="records")
    return jsonify(payload)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
