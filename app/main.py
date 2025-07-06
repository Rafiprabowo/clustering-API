from fastapi import FastAPI, File, UploadFile
from fastapi.responses import ORJSONResponse
import pandas as pd
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score

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

def klasifikasi_tingkat_risiko(nilai):
    if nilai > 21:
        return 5  # Sangat Tinggi / Ekstrem
    elif nilai > 16:
        return 4  # Tinggi
    elif nilai > 11:
        return 3  # Menengah
    elif nilai > 5:
        return 2  # Rendah
    else:
        return 1  # Sangat Rendah


def hitung_skor_iku(iku, bobot_iku=0.7, bobot_ikt=0.3):
    iku = str(iku)  # Cast ke string biar aman
    items = [item.strip() for item in iku.split(',') if item.strip()]
    total_skor = 0
    for item in items:
        if item.upper().startswith("IKU"):
            total_skor += bobot_iku
        elif item.upper().startswith("IKT"):
            total_skor += bobot_ikt

    return int(round(total_skor))  

def normalize_to_1_5(series):
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([3] * len(series), index=series.index)  # Semua sama → skor tengah
    normalized = 1 + 4 * (series - min_val) / (max_val - min_val)
    return normalized.round().astype(int)  # Dibulatkan ke integer



def interpret_centroids(centroids):
    interpreted = []
    for i, c in enumerate(centroids):
        # Langsung ambil nilai tanpa normalisasi tambahan
        skor_iku, anggaran, tingkat_risiko = c
        # Skor prioritas: tinggi IKU + tinggi anggaran + rendah risiko
        score = skor_iku * 0.4 + anggaran * 0.3 + (1 - tingkat_risiko) * 0.3

        interpreted.append({
            "cluster": i,
            "centroid": {
                "skor_iku": round(skor_iku, 2),
                "anggaran": round(anggaran, 2),
                "tingkat_risiko": round(tingkat_risiko, 2),
            },
            "prioritas_score": round(score, 4),
            "interpretasi": None
        })

    # Urutkan dari skor tertinggi ke terendah
    sorted_clusters = sorted(interpreted, key=lambda x: x["prioritas_score"], reverse=True)

    # Label prioritas otomatis
    prioritas_labels = [
        "Prioritas Tinggi",
        "Prioritas Sedang",
        "Prioritas Rendah",
        "Prioritas Sangat Rendah"
    ]
    num_clusters = len(centroids)
    if num_clusters <= len(prioritas_labels):
        labels = prioritas_labels[:num_clusters]
    else:
        labels = prioritas_labels + [f"Prioritas Level {i+1}" for i in range(4, num_clusters)]

    # Assign interpretasi
    for idx, cluster in enumerate(sorted_clusters):
        cluster["interpretasi"] = labels[idx]

    # Kembalikan urut sesuai cluster index asli
    return sorted(sorted_clusters, key=lambda x: x["cluster"])

@app.post("/uploadFile")
async def upload_excel(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls")):
        return {"error": "File harus berformat Excel"}

    contents = await file.read()
    df = pd.read_excel(BytesIO(contents))

    # Simpan data mentah
    data_awal = df.to_dict(orient="records")

    # Cleaning
    col_wajib = ['kode_iku', 'nilRabUsulan', 'dampak', 'probaBilitas']
    df_cleaned = df.dropna(subset=col_wajib).copy()
    df_cleaned["index_awal"] = df_cleaned.index
    df_cleaned.reset_index(drop=True, inplace=True)

    # Transform
    skor_iku = df_cleaned["kode_iku"].apply(hitung_skor_iku)
    anggaran = df_cleaned["nilRabUsulan"].astype(float)
    skor_dampak = df_cleaned["dampak"].map(MAP_DAMPAK).astype(int)
    skor_kemungkinan = df_cleaned["probaBilitas"].map(MAP_KEMUNGKINAN).astype(int)
    tingkat_risiko = skor_dampak * skor_kemungkinan

    # Gabungkan jadi DataFrame fitur baru
    features = pd.DataFrame({
        'skor_iku': skor_iku,
        'anggaran': anggaran,
        'tingkat_risiko': tingkat_risiko
    })

    # Normalisasi 1_5
    features["skor_iku"] = normalize_to_1_5(features['skor_iku'])
    features['anggaran'] = normalize_to_1_5(features['anggaran'])
    features['tingkat_risiko'] = normalize_to_1_5(features['tingkat_risiko'])

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    centroids_denorm = scaler.inverse_transform(kmeans.cluster_centers_)

    # Build response
    response = {
        "filename": file.filename,
        "peta_awals": data_awal,
        "peta_cleaneds": df_cleaned.to_dict(orient="records"),
        "preprocessing": [
            {
                "index_cleaned": i,
                "transform": {
                    "iku": int(skor_iku[i]),
                    "dampak": int(skor_dampak[i]),
                    "probaBilitas": int(skor_kemungkinan[i]),
                    "tingkat_risiko": int(tingkat_risiko[i])
                },
                "normalisasi": dict(zip(
                    ["skor_iku", "anggaran", "tingkat_risiko"],
                    features_scaled[i]
                ))
            } for i in range(len(df_cleaned))
        ],

        "clustering_run": {
            "jumlah_cluster": 4,
            "nama_file": file.filename,
            "silhouette_score": silhouette_score(features_scaled, labels=labels),
        },
        "cluster_results": [
            {"index_cleaned": i, "cluster": int(label)} for i, label in enumerate(labels)
        ],
        "interpretasi_clusters": interpret_centroids(centroids_denorm)
    }

    return response
