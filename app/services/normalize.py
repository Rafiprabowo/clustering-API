# app/services/normalize.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from app.db import SessionLocal, PetaNormalized

def normalize_data(data: list, nama_file: str):
    db = SessionLocal()
    df = pd.DataFrame(data)

    # Kolom yang akan dinormalisasi
    cols = ['iku_numerik', 'nilRabUsulan', 'tingkat_risiko']

    # Pastikan semua kolom tersedia
    missing_cols = [c for c in cols if c not in df.columns]
    if missing_cols:
        return {"status": "error", "message": f"Kolom tidak lengkap: {missing_cols}"}

    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[cols]), columns=['normal_' + c for c in cols])
    df_scaled = df_scaled.round(2)

    df = pd.concat([df.reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

    # Simpan hasil normalisasi ke DB
    normalized = PetaNormalized(
        nama_file=nama_file,
        data=df.to_dict(orient="records"),
        created_at=datetime.utcnow()
    )
    db.add(normalized)
    db.commit()
    db.close()

    return {
        "status": "success",
        "nama_file": nama_file,
        "total_data": len(df),
        "data": df.to_dict(orient="records")
    }
