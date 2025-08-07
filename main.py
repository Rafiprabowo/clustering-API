from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import pandas as pd
import numpy as np
from pydantic import BaseModel, root_validator, ValidationError
from sklearn.metrics import silhouette_score

app = FastAPI()

class NormalisasiRequest(BaseModel):
    data: List[dict]  # Data list dengan key: iku, nilai_rab_usulan, skor_risiko

class DataItem(BaseModel):
    data_cleaned_id: int
    iku: float
    nilai_rab_usulan: float
    skor_risiko: float  

class ClusteringRequest(BaseModel):
    data: List[DataItem]
    n_clusters: int
    weight_iku: float = 0.4
    weight_nilai_rab_usulan: float = 0.3
    weight_skor_risiko: float = 0.3

class DataMentah(BaseModel):
    id_usulan: str
    iku: Optional[str]
    nama_kegiatan: Optional[str]
    nilai_rab_usulan: Optional[float]
    nama_unit: Optional[str]
    resiko: Optional[str]
    dampak: Optional[str]
    probabilitas: Optional[str]
    pernyataan_risiko: Optional[str]
    uraian_dampak: Optional[str]
    pengendalian: Optional[str]
    
@app.post("/cleaning")
def clean_data(data: List[dict]):
    # Ubah input jadi DataFrame
    df = pd.DataFrame([d.dict() if hasattr(d, 'dict') else d for d in data])

    required_fields = ['iku', 'nilai_rab_usulan', 'dampak', 'probabilitas']

    # 1. Drop baris yang kolom wajibnya null
    df_cleaned = df.dropna(subset=required_fields)

    # 2. Drop baris yang kolom wajibnya string kosong
    for col in required_fields:
        df_cleaned = df_cleaned[df_cleaned[col].astype(str).str.strip() != '']

    # 3. Drop jika nilai_rab_usulan == 0 (pastikan sebagai numeric dulu)
    df_cleaned['nilai_rab_usulan'] = pd.to_numeric(df_cleaned['nilai_rab_usulan'], errors='coerce')
    df_cleaned = df_cleaned[df_cleaned['nilai_rab_usulan'] != 0]

    # 4. Drop duplikat berdasarkan nama_kegiatan
    if 'nama_kegiatan' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop_duplicates(subset='nama_kegiatan', keep='first')

    return df_cleaned.to_dict(orient='records')

@app.post("/normalisasi")
def normalisasi(req: NormalisasiRequest):
    try:
        df = pd.DataFrame(req.data)
        print(df)

        required_columns = ['iku', 'nilai_rab_usulan', 'skor_risiko']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail="Data tidak memiliki kolom yang diperlukan.")

        # Proses normalisasi menggunakan MinMaxScaler
        scaler = MinMaxScaler()
        df_scaled = scaler.fit_transform(df[required_columns])

        df_normalized = pd.DataFrame(df_scaled, columns=required_columns)
        df_normalized['data_cleaned_id'] = df['data_cleaned_id']
        print(df_normalized)

        return df_normalized.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clustering")
def clustering(req: ClusteringRequest):
    try:
        # Data dan ID awal
        raw_data = req.data
        n_clusters = req.n_clusters

        ids = [item.data_cleaned_id for item in raw_data]
        X = np.array([[item.iku, item.nilai_rab_usulan, item.skor_risiko] for item in raw_data])

        # KMeans Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_

        # Interpretasi dengan bobot dinamis
        interpretations = []
        centroid_scores = []

        for i, c in enumerate(centroids):
            skor = (
                c[0] * req.weight_iku +
                (1 - c[1]) * req.weight_nilai_rab_usulan +
                (1 - c[2]) * req.weight_skor_risiko
            )
            centroid_scores.append((i, skor))

        # Urutkan skor tertinggi → Prioritas Tinggi
        centroid_scores.sort(key=lambda x: x[1], reverse=True)
        label_mapping = {}

        for i, (cluster_id, skor) in enumerate(centroid_scores):
            label = ["Prioritas Sangat Tinggi","Prioritas Tinggi", "Prioritas Sedang", "Prioritas Rendah", "Prioritas Sangat Rendah"]
            # if n_clusters > 3:
            #     # Tambahkan lebih banyak label jika perlu
            #     label += [f"Prioritas {i+4}" for i in range(n_clusters - 3)]
            label_mapping[cluster_id] = {
                "skor": skor,
                "label": label[i]
            }

        # Bangun response hasil
        clustered_data = []
        for i, item in enumerate(raw_data):
            cluster_id = int(labels[i])
            clustered_data.append({
                "data_cleaned_id": item.data_cleaned_id,
                "iku": item.iku,
                "nilai_rab_usulan": item.nilai_rab_usulan,
                "skor_risiko": item.skor_risiko,
                "cluster": cluster_id
            })

        centroid_data = []
        for i, center in enumerate(centroids):
            centroid_data.append({
                "cluster": i,
                "iku": center[0],
                "nilai_rab_usulan": center[1],
                "skor_risiko": center[2],
            })

        interpretation_data = []
        for cluster_id, info in label_mapping.items():
            interpretation_data.append({
                "cluster": cluster_id,
                "label": info["label"],
                "skor": info["skor"]
            })

        akurasi = silhouette_score(X, labels)

        return {
            "clustered": clustered_data,
            "centroids": centroid_data,
            "interpretation": interpretation_data,
            "akurasi": f"{akurasi:.4f}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))