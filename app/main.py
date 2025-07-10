from fastapi import FastAPI, File, UploadFile
from fastapi.responses import ORJSONResponse
import pandas as pd
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import random

app = FastAPI(default_response_class=ORJSONResponse)

MAP_DAMPAK = {
    "Sangat Sedikit Berpengaruh": 1,
    "Sedikit Berpengaruh": 2,
    "Cukup Berpengaruh": 3,
    "Berpengaruh": 4,
    "Sangat Berpengaruh": 5
}
MAP_KEMUNGKINAN = {
    "Sangat Jarang": 1,
    "Jarang": 2,
    "Kadang-kadang": 3,
    "Sering": 4,
    "Sangat Sering": 5
}

def hitung_nilai_iku_dan_ikt(isi):
    if pd.isna(isi) or not isinstance(isi, str):
        return 0
    bagian = isi.lower().split(',')
    jumlah_iku = sum(1 for item in bagian if 'iku' in item.strip())
    jumlah_ikt = sum(1 for item in bagian if 'ikt' in item.strip())
    return (jumlah_iku * 0.7) + (jumlah_ikt * 0.3)

def klasifikasi_risiko(nilai):
    if nilai <= 8:
        return 0  # rendah
    elif 9 <= nilai <= 12:
        return 1  # sedang
    else:
        return 2  # tinggi

def scale_to_1_5_int(series):
    if series.max() == series.min():
        return pd.Series([3] * len(series), index=series.index)
    scaled = 1 + 4 * (series - series.min()) / (series.max() - series.min())
    return scaled.round().astype(int)


@app.post("/uploadFile")
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls")):
        return {"error": "File harus berformat Excel"}

    contents = await file.read()
    df = pd.read_excel(BytesIO(contents))

    # Hapus duplikat berdasarkan nama kegiatan
    df = df.drop_duplicates(subset='nmKegiatan')

    # Kolom penting
    col_wajib = ['dampak', 'probaBilitas', 'nilRabUsulan']

    # Ubah string kosong / 'null' jadi NaN
    df[col_wajib] = df[col_wajib].replace(['', ' ', 'NaN', 'nan', 'NULL', 'null'], pd.NA)

    # Mapping ke kolom baru (tidak menimpa kolom asli)
    df['dampak_angka'] = df['dampak'].map(MAP_DAMPAK)
    df['probaBilitas_angka'] = df['probaBilitas'].map(MAP_KEMUNGKINAN)

    # Konversi kolom wajib ke numerik (untuk nilRabUsulan)
    df['nilRabUsulan'] = pd.to_numeric(df['nilRabUsulan'], errors='coerce')

    # Hitung tingkat risiko
    df['tingkat_risiko'] = df['dampak_angka'] * df['probaBilitas_angka']

    # Validasi: hapus baris yang kosong atau 0 di kolom wajib + tingkat risiko
    col_wajib_final = ['dampak_angka', 'probaBilitas_angka', 'nilRabUsulan', 'tingkat_risiko']
    df_tidak_lengkap = df[
        df[col_wajib_final].isnull().any(axis=1) | (df[col_wajib_final] == 0).any(axis=1)
    ]
    df = df.drop(index=df_tidak_lengkap.index)

    #random

    # Buat list string IKU, misal IKU 1 s/d IKU 5
    iku_labels = [f"IKU {i}" for i in range(1, 6)]

    # Generate IKU random, bisa lebih dari 1 (dipisah koma)
    def random_iku():
        jumlah = random.randint(1, 5)  # Banyaknya IKU, misalnya 1 sampai 3 IKU per baris
        pilihan = random.sample(iku_labels, jumlah)
        return ', '.join(pilihan)

    # Assign ke kolom 'iku'
    df['iku'] = [random_iku() for _ in range(len(df))]

    df['iku_angka'] = df['iku'].apply(hitung_nilai_iku_dan_ikt)

    median = df['iku_angka'].median()


    # Transform
    df['nilai_iku'] = (df['iku_angka'] > median).astype(int)

    # df['nilRabUsulan_scaled'] = scale_to_1_5_int(df['nilRabUsulan'])
    df['tingkat_risiko_scaled'] = df['tingkat_risiko'].apply(klasifikasi_risiko)

    # Fitur clustering
    fitur_clustering = ['nilai_iku', 'nilRabUsulan', 'tingkat_risiko_scaled']
    X = df[fitur_clustering]

    # Scaling fitur
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Tentukan jumlah cluster optimal dengan Elbow
    model = KMeans(random_state=42, n_init='auto')
    visualizer = KElbowVisualizer(model, k=(2, 10), metric='distortion', timings=False)
    visualizer.fit(X_scaled)
    k_optimal = visualizer.elbow_value_
    print(f"Jumlah cluster optimal berdasarkan Elbow: {k_optimal}")

    # Clustering
    kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init='auto')
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Ambil nilai centroid
    centroids = kmeans.cluster_centers_
    centroid_list = [
        {
            "cluster": i,
            "nilai_iku": round(c[0]),
            "nilai_anggaran": round(c[1]),
            "tingkat_risiko": round(c[2])
        }
        for i, c in enumerate(centroids)
    ]

    # Evaluasi Silhouette Score
    score = silhouette_score(X_scaled, df['cluster'])
    print(f"Silhouette Score (k={k_optimal}): {score:.4f}")

    # Ringkasan
    total_data = len(df) + len(df_tidak_lengkap)
    data_bersih = len(df)
    data_dibuang = len(df_tidak_lengkap)

    return {
        "status": "success",
        "nama_file": file.filename,
        "total_data": total_data,
        "data_bersih": data_bersih,
        "data_dibuang": data_dibuang,
        "centroids": centroid_list,
        "score": score,
        "kolom": df.columns.to_list(),
        "data": df.to_dict(orient='records'),
        "data_gagal": df_tidak_lengkap.to_dict(orient='records')
    }
