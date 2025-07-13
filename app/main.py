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
    return int((jumlah_iku * 0.7) + (jumlah_ikt * 0.3))

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


iku_labels = [f"IKU {i}" for i in range(1, 6)]
# Generate IKU random, bisa lebih dari 1 (dipisah koma)
def random_iku():
    jumlah = random.randint(1, 5)  # Banyaknya IKU, misalnya 1 sampai 3 IKU per baris
    pilihan = random.sample(iku_labels, jumlah)
    return ', '.join(pilihan)


@app.post("/uploadFile")
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls")):
        return {"error": "File harus berformat Excel"}

    contents = await file.read()
    df = pd.read_excel(BytesIO(contents))

    df_awal = df.copy()

    # Cleaning

    # 1. Hapus duplikat berdasarkan nmKegiatan
    df = df.drop_duplicates(subset='nmKegiatan', keep='first')

    # 2. Hapus baris dengan nilai 0 atau NaN pada kolom nilRabUsulan
    df = df[df['nilRabUsulan'].notna() & (df['nilRabUsulan'] != 0)]

    # 3. Bersihkan nilai kosong/NaN pada kolom 'dampak' dan 'probaBilitas'
    for col in ['dampak', 'probaBilitas']:
        df[col] = df[col].astype(str).str.strip()
        df = df[(df[col].notna()) & (df[col].str.lower() != 'nan') & (df[col] != '')]

    # 4. Cek kolom yang mengandung kata 'iku' (tidak case sensitive)
    iku_col = next((col for col in df.columns if 'iku' in col.lower()), None)

    # 5. Jika kolom 'iku' ditemukan, bersihkan nilainya juga
    if iku_col is not None:
        df[iku_col] = df[iku_col].astype(str).str.strip()
        df = df[(df[iku_col].notna()) & (df[iku_col].str.lower() != 'nan') & (df[iku_col] != '')]


    # 6. Reset index setelah semua filter
    df = df.reset_index(drop=True)


    # Jika kolom IKU tidak ada (case-insensitive)
    iku_col = next((col for col in df.columns if 'iku' in col.lower()), None)

    if not iku_col:
        df['iku'] = [random_iku() for _ in range(len(df))]
        iku_col = 'iku'  # agar bisa dipakai lagi nanti

    # Transform Numerik
    # Mapping dampak dan probabilitas
    df['dampak_numerik'] = df['dampak'].map(MAP_DAMPAK)
    df['probabilitas_numerik'] = df['probaBilitas'].map(MAP_KEMUNGKINAN)
    df['tingkat_risiko'] = df['dampak_numerik'] * df['probabilitas_numerik']
    # Transformasi IKU jika kolom IKU ada
    if iku_col:
        df['iku_numerik'] = df[iku_col].apply(hitung_nilai_iku_dan_ikt)

    # Normalisasi
    cols = ['iku_numerik', 'nilRabUsulan', 'tingkat_risiko']

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[cols]), columns=['normal_' + c for c in cols]).round(2)
    df = pd.concat([df.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

    # 4. Siapkan data untuk clustering
    fitur_clustering = df_scaled.columns.tolist()
    X = df[fitur_clustering]

    # Tentukan jumlah cluster optimal dengan Elbow
    # model = KMeans(random_state=42, n_init='auto')
    # visualizer = KElbowVisualizer(model, k=(2, 10), metric='distortion', timings=False)
    # visualizer.fit(X)
    # k_optimal = visualizer.elbow_value_
    # print(f"Jumlah cluster optimal berdasarkan Elbow: {k_optimal}")

    # Clustering
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    # Ambil nilai centroid
    centroids = kmeans.cluster_centers_
    centroid_list = [
        {
            "cluster": i,
            "c_iku": round(c[0], 2),
            "c_nilRabUsulan": round(c[1], 2),
            "c_tingkat_risiko": round(c[2], 2)
        }
        for i, c in enumerate(centroids)
    ]

    # Evaluasi Silhouette Score
    score = silhouette_score(X, df['cluster'])

    # Ringkasan
    total_data = len(df_awal) 
    data_bersih = len(df)
    data_dibuang = len(df_awal) - len(df)

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
    }
