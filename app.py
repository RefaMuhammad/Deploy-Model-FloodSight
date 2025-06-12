from flask import Flask, request, jsonify
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import tifffile as tiff
import io
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from utils import preprocess_tabular_data, load_image_as_array  # Pastikan ini tersedia

app = Flask(__name__)

# Load model dan preprocessor saat startup
model = tf.keras.models.load_model("saved_model/")
preprocessor = joblib.load("preprocessor.pkl")


def load_tif_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    img = tiff.imread(io.BytesIO(response.content))
    img = img.astype(np.float32) / 255.0
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/predict", methods=["GET"])
def predict():
    year = int(request.args.get("year"))
    month = int(request.args.get("month"))
    lon = float(request.args.get("longitude"))
    lat = float(request.args.get("latitude"))

    now = datetime.now()
    current_year = now.year
    current_month = now.month

    # Pengecekan masa depan untuk data citra
    imagery_year = year
    if year > current_year:
        imagery_year = current_year
    imagery_year -= 1

    # Ambil data citra
    image_url = f"http://gee.up.railway.app/api/imagery/{imagery_year}?longitude={lon}&latitude={lat}"
    image_response = requests.get(image_url).json()

    if not image_response["success"]:
        return jsonify({"error": "Data citra tidak tersedia"}), 400

    image_download_url = image_response["imagery"]["download_url"]

    # Ambil data tabular dari API
    tabular_url = f"http://gee.up.railway.app/api/data/{year}/{month}?longitude={lon}&latitude={lat}"
    tabular_response = requests.get(tabular_url).json()

    if not tabular_response["success"] or len(tabular_response["data"]) == 0:
        return jsonify({"error": "Data tabular tidak ditemukan"}), 400

    tabular_data = tabular_response["data"][0]

    # Jika input tahun/bulan di masa depan, ganti untuk data rainfall & soil moisture
    fallback_year = current_year - 1
    if year > current_year or (year == current_year and month > current_month):
        fallback_url = f"http://gee.up.railway.app/api/data/{fallback_year}/{month}?longitude={lon}&latitude={lat}"
        fallback_response = requests.get(fallback_url).json()
        if fallback_response["success"] and len(fallback_response["data"]) > 0:
            fallback_data = fallback_response["data"][0]
            # Ganti hanya kolom yang relevan
            for col in ["avg_rainfall", "max_rainfall", "soil_moisture"]:
                tabular_data[col] = fallback_data.get(col, 0.0)

    # Konversi ke DataFrame
    tabular_df = pd.DataFrame([tabular_data])

    # Preprocessing tabular
    try:
        X_tabular = preprocess_tabular_data(tabular_df, preprocessor)
    except Exception as e:
        return jsonify({"error": f"Preprocessing gagal: {str(e)}"}), 500

    # Preprocessing citra
    try:
        image_array = load_image_as_array(image_download_url)
    except Exception as e:
        return jsonify({"error": f"Gagal load citra: {str(e)}"}), 500

    # Prediksi
    try:
        prediction = model.predict([image_array, X_tabular])
        result = int(np.round(prediction[0][0]))
    except Exception as e:
        return jsonify({"error": f"Prediksi gagal: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "prediction": result,
        "metadata": {
            "district": tabular_response.get("district", "Unknown"),
            "coordinates": {"latitude": lat, "longitude": lon},
            "imagery_year": imagery_year
        }
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default ke 5000 untuk development lokal
    app.run(host="0.0.0.0", port=port)
