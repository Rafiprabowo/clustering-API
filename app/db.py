# app/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, JSON, Float, DateTime
from datetime import datetime

DATABASE_URL = "sqlite:///./clustering.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# Model untuk menyimpan hasil cleaning
class PetaCleaned(Base):
    __tablename__ = "peta_cleaneds"
    id = Column(Integer, primary_key=True, index=True)
    nama_file = Column(String, index=True)
    data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)


# Model untuk hasil transformasi
class PetaTransformed(Base):
    __tablename__ = "peta_transforms"
    id = Column(Integer, primary_key=True, index=True)
    nama_file = Column(String, index=True)
    data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# Model untuk hasil normalisasi
class PetaNormalized(Base):
    __tablename__ = "peta_normalizes"
    id = Column(Integer, primary_key=True, index=True)
    nama_file = Column(String, index=True)
    data = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

# Model untuk hasil clustering
class ClusterPeta(Base):
    __tablename__ = "cluster_petas"
    id = Column(Integer, primary_key=True, index=True)
    nama_file = Column(String, index=True)
    data = Column(JSON)  
    centroids = Column(JSON)
    score = Column(Float)
    total_data = Column(Integer)
    data_dibuang = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)

# Inisialisasi tabel saat pertama kali
Base.metadata.create_all(bind=engine)
