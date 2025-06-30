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

def hitung_skor_iku(iku: str):
    items = [item.strip() for item in iku.split(',')]
    total_skor = 0
    for item in items:
        if item.startswith("IKU"):
            total_skor += 0.7
        elif item.startswith("IKT"):
            total_skor += 0.3   
    return round(total_skor, 1)

def interpret_centroids(centroids):
    import numpy as np

    # Pisahkan nilai
    skor_ikus = np.array([c[0] for c in centroids])
    anggarans = np.array([c[1] for c in centroids])
    risiko = np.array([c[2] for c in centroids])

    # Normalisasi ke rentang 0–1
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else np.zeros_like(x)

    norm_iku = normalize(skor_ikus)
    norm_anggaran = normalize(anggarans)
    norm_risiko = normalize(risiko)

    # Buat skor gabungan (dengan bobot)
    interpreted = []
    for i in range(len(centroids)):
        # Tinggi iku, tinggi anggaran, rendah risiko
        score = norm_iku[i] * 0.4 + norm_anggaran[i] * 0.3 + (1 - norm_risiko[i]) * 0.3
        interpreted.append({
            "cluster": i,
            "centroid": {
                "skor_iku": round(skor_ikus[i], 2),
                "anggaran": round(anggarans[i], 2),
                "tingkat_risiko": round(risiko[i], 2),
            },
            "prioritas_score": round(score, 4),
            "interpretasi": None
        })

    # Urutkan berdasarkan skor
    sorted_clusters = sorted(interpreted, key=lambda x: x["prioritas_score"], reverse=True)

    for idx, cluster in enumerate(sorted_clusters):
        cluster["interpretasi"] = [
            # "Prioritas Sangat Tinggi",
            "Prioritas Tinggi",
            "Prioritas Sedang",
            "Prioritas Rendah",
            # "Prioritas Sangat Rendah",
        ][idx] if idx < 3 else f"Prioritas Level {idx+1}"

    # Kembalikan dalam urutan cluster 0-n
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
    def normalize_to_1_5(series):
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([3] * len(series), index=series.index)  # Semua sama → beri skor tengah
        return 1 + 4 * (series - min_val) / (max_val - min_val)

    df_cleaned["skor_iku"] = df_cleaned["kode_iku"].apply(hitung_skor_iku)
    # df_cleaned["skor_iku"] = normalize_to_1_5(df_cleaned['skor_iku'])

    df_cleaned["skor_dampak"] = df_cleaned["dampak"].map(MAP_DAMPAK).fillna(0).astype(int)
    df_cleaned["skor_kemungkinan"] = df_cleaned["probaBilitas"].map(MAP_KEMUNGKINAN).fillna(0).astype(int)
    df_cleaned["tingkat_risiko"] = df_cleaned["skor_dampak"] * df_cleaned["skor_kemungkinan"]
    df_cleaned["anggaran"] = df_cleaned["nilRabUsulan"].astype(float)

    # Normalisasi
    features = df_cleaned[["skor_iku", "anggaran", "tingkat_risiko"]]
    # features = df_cleaned[["skor_iku", "anggaran", "skor_dampak", "skor_kemungkinan"]]
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(features_scaled)
    centroids_denorm = scaler.inverse_transform(kmeans.cluster_centers_)

    # Build response
    response = {
        "filename": file.filename,
        "peta_awals": data_awal,
        "peta_cleaneds": df_cleaned.drop(columns=["skor_iku", "skor_dampak", "skor_kemungkinan", "anggaran", "tingkat_risiko"]).to_dict(orient="records"),
        # "peta_cleaneds": df_cleaned.drop(columns=["skor_iku", "skor_dampak", "skor_kemungkinan", "anggaran"]).to_dict(orient="records"),
        "preprocessing": [
            {
                "index_cleaned": i,
                "transform": {
                    "iku": row['skor_iku'],
                    "dampak": row['skor_dampak'],
                    "probaBilitas": row['skor_kemungkinan']
                },
                "normalisasi": dict(zip(
                    ["skor_iku", "anggaran", "tingkat_risiko"],
                    # ["skor_iku", "anggaran", "skor_dampak", "skor_kemungkinan" ],
                    features_scaled[i]
                ))
            } for i, row in df_cleaned.iterrows()
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
