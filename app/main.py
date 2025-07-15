# Simulasi struktur endpoint FastAPI berdasarkan tahapan: cleaning, transform, normalize, clustering

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import random

app = FastAPI()


class CleaningRequest(BaseModel):
    nama_file: str
    data: List[Dict[str, Any]]  # data tabular dari Laravel
    

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

@app.post("/cleaning")
async def cleaning(request: CleaningRequest):
    # Ubah list of dict → DataFrame
    df = pd.DataFrame(request.data)

    # Drop duplikat berdasarkan nama_kegiatan
    df = df.drop_duplicates(subset='nama_kegiatan')

    # Filter nilai_rab_usulan tidak null dan tidak 0
    df['nilai_rab_usulan'] = pd.to_numeric(df['nilai_rab_usulan'], errors='coerce')
    df = df[df['nilai_rab_usulan'].notna() & (df['nilai_rab_usulan'] != 0)]

    # Bersihkan kolom dampak dan probabilitas
    for col in ['dampak', 'probabilitas']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df = df[df[col].notna() & (df[col].str.lower() != 'nan') & (df[col] != '')]

    # Bersihkan kolom yang mengandung kata 'iku' (jika ada)
    iku_col = next((col for col in df.columns if 'iku' in col.lower()), None)
    if iku_col:
        df[iku_col] = df[iku_col].astype(str).str.strip()
        df = df[df[iku_col].notna() & (df[iku_col].str.lower() != 'nan') & (df[iku_col] != '')]

    df = df.reset_index(drop=True)

    # Kembalikan dalam format list of dict
    return JSONResponse(content={
        "status": "success",
        "data": df.to_dict(orient="records")
    })

   

@app.post("/transform")
async def transform(request: Request):
    payload = await request.json()
    df = pd.DataFrame(payload['data'])

    # Transformasi nilai IKU/IKT → numerik
    df['iku'] = df['iku'].apply(hitung_nilai_iku_dan_ikt)

    # Mapping dampak dan probabilitas → numerik (timpa kolom asli)
    df['dampak'] = df['dampak'].map(MAP_DAMPAK).fillna(0)
    df['probabilitas'] = df['probabilitas'].map(MAP_KEMUNGKINAN).fillna(0)

    return JSONResponse(content={
        "status": "success",
        "data": df.to_dict(orient="records")
    })

   

@app.post("/normalize")
async def normalize(request: Request):
    payload = await request.json()
    df = pd.DataFrame(payload['data'])

    # Hitung tingkat risiko = dampak * probabilitas
    df['tingkat_risiko'] = df['dampak'] * df['probabilitas']

    # Normalisasi fitur
    cols = ['iku', 'nilai_rab_usulan', 'tingkat_risiko']
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[cols]), columns=cols).round(2)

    # Timpa nilai aslinya
    df[cols] = df_scaled

    return JSONResponse({
        "status": "success",
        "data": df.to_dict(orient="records")
    })


@app.post("/clustering")
async def clustering(request: Request):
    payload = await request.json()
    df = pd.DataFrame(payload['data'])

    fitur = ['iku', 'nilai_rab_usulan', 'tingkat_risiko']
    X = df[fitur]

    # Clustering
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10)
    df['cluster'] = kmeans.fit_predict(X)

    print(len(df))

    # Ambil centroid
    centroids = kmeans.cluster_centers_

    # Interpretasi berdasarkan skor kustom
    centroid_list = []
    for i, c in enumerate(centroids):
        c_iku = round(c[0], 4)
        c_anggaran = round(c[1], 4)
        c_risiko = round(c[2], 4)

        # Hitung skor prioritas: makin kecil, makin tinggi prioritas
        skor = (c_iku * 0.4) + ((1 - c_anggaran) * 0.3) + ((1 - c_risiko) * 0.3)
        centroid_list.append({
            "cluster": i,
            "c_iku": c_iku,
            "c_anggaran": c_anggaran,
            "c_tingkat_risiko": c_risiko,
            "skor_prioritas": round(skor, 4)
        })

    # Urutkan skor: makin tinggi skor = prioritas makin tinggi
    sorted_by_score = sorted(centroid_list, key=lambda x: x['skor_prioritas'], reverse=True)

    interpretasi_labels = [
        'Prioritas Sangat Tinggi',
        'Prioritas Tinggi',
        'Prioritas Sedang',
        'Prioritas Rendah',
        'Prioritas Sangat Rendah'
    ]

    for i, item in enumerate(sorted_by_score):
        item['interpretasi'] = interpretasi_labels[i]
        item['tingkat_prioritas'] = i + 1

    # Gabungkan interpretasi ke centroid utama
    for centroid in centroid_list:
        match = next((x for x in sorted_by_score if x['cluster'] == centroid['cluster']), None)
        if match:
            centroid['interpretasi'] = match['interpretasi']
            centroid['tingkat_prioritas'] = match['tingkat_prioritas']

    # Hitung silhouette score
    score = silhouette_score(X, df['cluster'])

    return JSONResponse({
        "data": df.to_dict(orient='records'),
        "centroids": centroid_list,
        "score": round(score, 4)
    })

