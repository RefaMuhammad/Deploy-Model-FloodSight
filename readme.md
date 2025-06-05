# 🌊 Flood Prediction API (Multimodal)

API Flask untuk memprediksi risiko banjir berdasarkan **data tabular** dan **citra satelit `.tif` (RGB)**. Model deep learning menggabungkan dua jenis input untuk memberikan prediksi yang lebih akurat.

---

## 📦 Fitur

- ✅ Input: koordinat (latitude, longitude), tahun, dan bulan
- ✅ Mendapatkan data tabular dari REST API eksternal
- ✅ Memproses dan resize citra satelit `.tif`
- ✅ Prediksi menggunakan model multimodal TensorFlow
- ✅ Output JSON (prediksi + metadata)

---

## 🧠 Teknologi

- Python 3.9+
- Flask
- TensorFlow
- Rasterio, tifffile (citra `.tif`)
- Pandas, NumPy, scikit-learn
- joblib, requests

---

## 🚀 Instalasi Lokal

1. **Clone repositori:**

```bash
git clone https://github.com/username/flood-prediction-api.git
cd flood-prediction-api
```

2. **Aktifkan virtual environment (opsional):**

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. **Install dependensi:**

```bash
pip install -r requirements.txt
```

4. **Jalankan API:**

```bash
python app.py
```

API berjalan di: `http://localhost:5000/predict`

---

## 📤 Deploy ke Railway (Gratis dan Mudah)

1. **Login/daftar Railway:**  
   https://railway.app/

2. **Buat project baru → Deploy from GitHub Repo**

3. **Upload model dan file penting:**

   - `saved_model/` (folder TensorFlow SavedModel)
   - `preprocessor.pkl`
   - `app.py`, `utils.py`, `requirements.txt`, `README.md`

4. **Tambahkan file `Procfile` di root:**

```
web: python app.py
```

5. **Railway otomatis menginstall dependensi dan menjalankan server.**

6. **Set `PORT` di app.py:**

```python
port = int(os.environ.get("PORT", 5000))
app.run(debug=False, host="0.0.0.0", port=port)
```

7. **API online! Coba via browser atau Postman:**

```
https://your-project-name.up.railway.app/predict?year=2024&month=3&longitude=106.82&latitude=-6.2
```

---

## 📎 Contoh Request

```http
GET /predict?year=2024&month=3&longitude=106.82&latitude=-6.2
```

### ✅ Contoh Respon

```json
{
  "success": true,
  "prediction": 1,
  "metadata": {
    "district": "Ciledug",
    "coordinates": {
      "latitude": -6.2,
      "longitude": 106.82
    },
    "imagery_year": 2023
  }
}
```

---

## 🗂 Struktur Direktori

```
flood-prediction-api/
├── app.py
├── utils.py
├── preprocessor.pkl
├── saved_model/
│   └── [TensorFlow SavedModel files]
├── requirements.txt
├── Procfile
└── README.md
```

---

## 📜 Lisensi

MIT License

---

## 🙋‍♂️ Kontak

Refa Muhammad – [refamuhammad.pcommtech.com](https://refamuhammad.pcommtech.com)
