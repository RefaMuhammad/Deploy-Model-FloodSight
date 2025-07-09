from flask import Flask, request, jsonify
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import tifffile as tiff
import io
from datetime import datetime
import os

from utils import preprocess_tabular_data, load_image_as_array  # Pastikan ini tersedia

app = Flask(__name__)

# Load model dengan signature
loaded = tf.saved_model.load("saved_model/")
infer = loaded.signatures["serving_default"]

# Load preprocessor
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

    # Gunakan imagery tahun sebelumnya jika future
    imagery_year = year
    if year > current_year:
        imagery_year = current_year
    imagery_year -= 1

    # Ambil citra
    image_url = f"http://suciihtisabi-datafloodsight.hf.space/api/imagery/{imagery_year}?longitude={lon}&latitude={lat}"
    image_response = requests.get(image_url).json()

    if not image_response["success"]:
        return jsonify({"error": "Data citra tidak tersedia"}), 400

    image_download_url = image_response["imagery"]["download_url"]

    # Kurangi bulan untuk API GEE
    api_month = month - 1
    api_year = year
    if api_month <= 0:
        api_month = 12
        api_year -= 1

    # Ambil data tabular
    tabular_url = f"http://suciihtisabi-datafloodsight.hf.space/api/data/{api_year}/{api_month}?longitude={lon}&latitude={lat}"
    tabular_response = requests.get(tabular_url).json()

    if not tabular_response["success"] or len(tabular_response["data"]) == 0:
        return jsonify({"error": "Data tabular tidak ditemukan"}), 400

    tabular_data = tabular_response["data"][0]

    # Data fallback jika masa depan
    fallback_year = current_year - 1
    if year > current_year or (year == current_year and month > current_month):
        # Fallback juga dikurangi sebulan
        fallback_month = month - 1
        fallback_year_adjusted = fallback_year
        if fallback_month <= 0:
            fallback_month = 12
            fallback_year_adjusted -= 1
            
        fallback_url = f"http://suciihtisabi-datafloodsight.hf.space/api/data/{fallback_year_adjusted}/{fallback_month}?longitude={lon}&latitude={lat}"
        fallback_response = requests.get(fallback_url).json()
        if fallback_response["success"] and len(fallback_response["data"]) > 0:
            fallback_data = fallback_response["data"][0]
            for col in ["avg_rainfall", "max_rainfall", "soil_moisture"]:
                tabular_data[col] = fallback_data.get(col, 0.0)

    # Konversi tabular ke dataframe
    tabular_df = pd.DataFrame([tabular_data])
    tabular_df.drop(columns=['NAME_2', 'long', 'lat'], inplace=True)

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

    # Prediksi menggunakan signature
    try:
        # Ganti "input_1" dan "input_2" sesuai input signature model Anda
        inputs = {
            "image_input": tf.convert_to_tensor(image_array, dtype=tf.float32),
            "tabular_input": tf.convert_to_tensor(X_tabular, dtype=tf.float32)
        }


        output = infer(**inputs)
        prediction = list(output.values())[0].numpy()
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
