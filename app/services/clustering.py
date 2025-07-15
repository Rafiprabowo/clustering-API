# app/services/clustering.py
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime
from app.db import SessionLocal, ClusterPeta

def clustering_data(data: list, nama_file: str):
    db = SessionLocal()
    df = pd.DataFrame(data)

    # Ambil kolom fitur hasil normalisasi
    fitur_clustering = ['normal_iku_numerik', 'normal_nilRabUsulan', 'normal_tingkat_risiko']

    for col in fitur_clustering:
        if col not in df.columns:
            return {"status": "error", "message": f"Kolom {col} tidak ditemukan dalam data"}

    X = df[fitur_clustering]

    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    # Simpan hasil centroid
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

    score = silhouette_score(X, df['cluster'])

    # Simpan ke DB
    clustering = ClusterPeta(
        nama_file=nama_file,
        data=df.to_dict(orient="records"),
        centroids=centroid_list,
        score=round(score, 3),
        total_data=len(df),
        data_dibuang=0,  # bisa diisi jika ada perhitungan
        created_at=datetime.utcnow()
    )
    db.add(clustering)
    db.commit()
    db.close()

    return {
        "status": "success",
        "nama_file": nama_file,
        "total_data": len(df),
        "score": round(score, 3),
        "centroids": centroid_list,
        "data": df.to_dict(orient="records")
    }
