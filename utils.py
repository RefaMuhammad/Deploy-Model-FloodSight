import numpy as np
import pandas as pd
from sklearn.utils import estimator_html_repr
from PIL import Image
import requests
import rasterio
import tensorflow as tf
import io
from io import BytesIO
from scipy.stats.mstats import winsorize

def preprocess_tabular_data(df_tabular, preprocessor):

    df_tabular.drop(columns=['NAME_2', 'long', 'lat'], inplace=True)
    
    # Winsorize outliers
    outlier_columns = ['avg_rainfall', 'max_rainfall', 'avg_temperature',
                       'elevation', 'slope', 'soil_moisture']
    for col in outlier_columns:
        df_tabular[col] = winsorize(df_tabular[col], limits=[0.01, 0.01])

    # Transform with pre-fitted preprocessor
    X_processed = preprocessor.transform(df_tabular)
    return X_processed

def load_image_as_array(image_url, target_size=(128, 128)):
    # Download gambar dari URL dan baca pakai rasterio dari buffer
    response = requests.get(image_url)
    response.raise_for_status()
    file_bytes = io.BytesIO(response.content)

    with rasterio.MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            img = src.read([1, 2, 3])  # Ambil channel RGB
            img = np.nan_to_num(img, nan=0.0).astype(np.float32)

            # Normalisasi tiap channel (min-max)
            img_min, img_max = img.min(), img.max()
            img = (img - img_min) / (img_max - img_min + 1e-6)

            img = np.transpose(img, (1, 2, 0))  # (H, W, C)

    # Resize pakai TensorFlow ke target_size (default 128x128)
    img = tf.image.resize(img, target_size).numpy()

    # Tambahkan batch dimension (1, H, W, C)
    img = np.expand_dims(img, axis=0)
    return img
