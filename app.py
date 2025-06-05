from flask import Flask, request, jsonify
import requests
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import tifffile as tiff
import io

from utils import preprocess_tabular_data, load_image_as_array   # Asumsikan ada fungsi ini di utils.py

app = Flask(__name__)

# Load model dan preprocessor saat startup
model = tf.keras.models.load_model("saved_model/")
preprocessor = joblib.load("preprocessor.pkl")


def load_tif_image_from_url(url):
    """
    Download dan load gambar .tif dari URL menggunakan tifffile.
    Output: numpy array dengan shape (1, height, width, channels)
    """
    response = requests.get(url)
    response.raise_for_status()  # Error kalau gagal download

    # Baca isi file tiff dari bytes
    img = tiff.imread(io.BytesIO(response.content))

    # Normalisasi ke [0, 1]
    img = img.astype(np.float32) / 255.0

    # Pastikan image punya channel, jika grayscale tambah dimensi channel
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)

    # Tambah batch dimensi
    img = np.expand_dims(img, axis=0)

    return img


@app.route("/predict", methods=["GET"])
def predict():
    year = int(request.args.get("year"))
    month = int(request.args.get("month"))
    lon = float(request.args.get("longitude"))
    lat = float(request.args.get("latitude"))

    # Ambil data tabular
    tabular_url = f"http://gee.up.railway.app/api/data/{year}/{month}?longitude={lon}&latitude={lat}"
    tabular_response = requests.get(tabular_url).json()

    if not tabular_response["success"] or len(tabular_response["data"]) == 0:
        return jsonify({"error": "Data tabular tidak ditemukan"}), 400

    tabular_data = pd.DataFrame([tabular_response["data"][0]])

    # Ambil data citra (gunakan tahun citra = tahun - 1)
    image_url = f"http://gee.up.railway.app/api/imagery/{year - 1}?longitude={lon}&latitude={lat}"
    image_response = requests.get(image_url).json()

    if not image_response["success"]:
        return jsonify({"error": "Data citra tidak tersedia"}), 400

    image_download_url = image_response["imagery"]["download_url"]

    # Preprocess tabular
    try:
        X_tabular = preprocess_tabular_data(tabular_data, preprocessor)  # shape (1, n_features)
    except Exception as e:
        return jsonify({"error": f"Preprocessing gagal: {str(e)}"}), 500

    # Preprocess citra
    try:
        image_array = load_image_as_array(image_download_url)  # shape (1, 128, 128, 3)
    except Exception as e:
        return jsonify({"error": f"Gagal load citra: {str(e)}"}), 500

    # Prediksi
    try:
        # Pastikan urutan input sesuai model: [image_input, tabular_input] atau sebaliknya
        prediction = model.predict([image_array, X_tabular])
        result = int(np.round(prediction[0][0]))
    except Exception as e:
        return jsonify({"error": f"Prediksi gagal: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "prediction": result,
        "metadata": {
            "district": tabular_response.get("district", "Unknown"),
            "coordinates": {
                "latitude": lat,
                "longitude": lon
            },
            "imagery_year": year - 1
        }
    })

if __name__ == "__main__":
    app.run(debug=True)