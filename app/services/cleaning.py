# app/services/cleaning.py
import pandas as pd
from app.db import SessionLocal
from app.db import PetaCleaned
from datetime import datetime

def clean_data(data: list, nama_file: str):
    db = SessionLocal()
    df_awal = pd.DataFrame(data)

    df = df_awal.drop_duplicates(subset='nmKegiatan', keep='first')
    df = df[df['nilRabUsulan'].notna() & (df['nilRabUsulan'] != 0)]

    for col in ['dampak', 'probaBilitas']:
        df[col] = df[col].astype(str).str.strip()
        df = df[(df[col].notna()) & (df[col].str.lower() != 'nan') & (df[col] != '')]

    iku_col = next((col for col in df.columns if 'iku' in col.lower()), None)
    if iku_col:
        df[iku_col] = df[iku_col].astype(str).str.strip()
        df = df[(df[iku_col].notna()) & (df[iku_col].str.lower() != 'nan') & (df[iku_col] != '')]

    df = df.reset_index(drop=True)

    # Simpan hasil ke DB
    cleaned = PetaCleaned(
        nama_file=nama_file,
        data=df.to_dict(orient="records"),
        created_at=datetime.utcnow()
    )
    db.add(cleaned)
    db.commit()
    db.close()

    return {
        "status": "success",
        "nama_file": nama_file,
        "total_data": len(df_awal),
        "data_dibuang": len(df_awal) - len(df),
        "data": df.to_dict(orient="records")
    }