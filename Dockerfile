# Gunakan image dasar Python
FROM python:3.11.9-slim

# Tentukan working directory di dalam container
WORKDIR /app

# Salin file requirements dan install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file ke dalam container
COPY . .

# Jalankan FastAPI dengan Uvicorn saat container dimulai
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]
