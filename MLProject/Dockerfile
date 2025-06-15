# Dockerfile untuk proyek CI/CD Machine Learning
# Menggunakan image dasar Miniconda3 dari ContinuumIO
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Salin semua file ke dalam container
COPY . /app

# Buat environment dari conda.yaml
RUN conda env create -f conda.yaml

# Aktifkan shell default ke dalam environment saat build selesai
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Expose port
EXPOSE 5000

# Menetapkan perintah default untuk menjalankan pelatihan model saat container dijalankan
CMD ["conda", "run", "-n", "mlflow-env", "python", "modelling.py"]
