# app/services/transform.py
import pandas as pd
from app.db import SessionLocal
from app.db import PetaTransformed
from datetime import datetime

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

def transform_data(data: list, nama_file: str):
    db = SessionLocal()
    df = pd.DataFrame(data)

    iku_col = next((col for col in df.columns if 'iku' in col.lower()), 'iku')

    df['dampak_numerik'] = df['dampak'].map(MAP_DAMPAK)
    df['probabilitas_numerik'] = df['probaBilitas'].map(MAP_KEMUNGKINAN)
    df['tingkat_risiko'] = df['dampak_numerik'] * df['probabilitas_numerik']
    df['iku_numerik'] = df[iku_col].apply(hitung_nilai_iku_dan_ikt)

    # Simpan hasil transformasi ke DB
    transformed = PetaTransformed(
        nama_file=nama_file,
        data=df.to_dict(orient="records"),
        created_at=datetime.utcnow()
    )
    db.add(transformed)
    db.commit()
    db.close()

    return {
        "status": "success",
        "nama_file": nama_file,
        "total_data": len(df),
        "data": df.to_dict(orient="records")
    }
