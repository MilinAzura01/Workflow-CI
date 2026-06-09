"""
prometheus_exporter.py
Script untuk mengekspos metrics model MLflow ke Prometheus
"""

import time
import requests
import pandas as pd
from prometheus_client import start_http_server, Counter, Histogram, Gauge

# ─── Tracking global ───
request_count = 0
start_time_global = time.time()

# Referensi mean fitur dari data training
REFERENCE_MEANS = {
    "Pclass": -0.3, "Age": -0.1, "Fare": -0.2,
    "SibSp": 0.0, "Parch": 0.0, "FamilySize": -0.1
}

# ─── Definisi Metrics ───
REQUEST_COUNT = Counter(
    'mlflow_request_total',
    'Total jumlah request ke model',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'mlflow_request_latency_seconds',
    'Latency request ke model dalam detik',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

PREDICTION_SCORE = Gauge(
    'mlflow_prediction_score',
    'Skor prediksi model (probabilitas survived)'
)

MODEL_ACCURACY = Gauge(
    'mlflow_model_accuracy',
    'Akurasi model pada test set'
)

ACTIVE_REQUESTS = Gauge(
    'mlflow_active_requests',
    'Jumlah request yang sedang aktif'
)

ERROR_COUNT = Counter(
    'mlflow_error_total',
    'Total jumlah error pada model serving',
    ['error_type']
)

DATA_DRIFT_SCORE = Gauge(
    'mlflow_data_drift_score',
    'Skor data drift (0=no drift, 1=full drift)'
)

FEATURE_MISSING_RATE = Gauge(
    'mlflow_feature_missing_rate',
    'Tingkat missing values pada input features'
)

PREDICTION_CONFIDENCE = Gauge(
    'mlflow_prediction_confidence',
    'Confidence score prediksi model'
)

THROUGHPUT = Gauge(
    'mlflow_throughput_rps',
    'Throughput model dalam requests per second'
)


def send_prediction_request(data):
    """Kirim request ke MLflow model serving."""
    global request_count

    url = "http://127.0.0.1:5001/invocations"
    headers = {"Content-Type": "application/json"}

    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        response = requests.post(url, headers=headers, json=data, timeout=10)
        latency = time.time() - start_time

        REQUEST_LATENCY.observe(latency)
        REQUEST_COUNT.labels(method='POST', endpoint='/invocations', status=str(response.status_code)).inc()

        if response.status_code == 200:
            request_count += 1
            result = response.json()
            predictions = result.get('predictions', [0.5])
            score = predictions[0] if isinstance(predictions, list) else 0.5
            PREDICTION_SCORE.set(score)
            PREDICTION_CONFIDENCE.set(abs(score - 0.5) * 2)
        else:
            ERROR_COUNT.labels(error_type='http_error').inc()

    except Exception as e:
        ERROR_COUNT.labels(error_type='connection_error').inc()
        print(f"[ERROR] {e}")
    finally:
        ACTIVE_REQUESTS.dec()


def update_system_metrics(input_data):
    """Update metrics sistem dari data nyata."""

    df = pd.DataFrame(input_data["dataframe_records"])

    # FEATURE_MISSING_RATE → hitung dari input nyata
    missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    FEATURE_MISSING_RATE.set(missing / total_cells if total_cells > 0 else 0)

    # DATA_DRIFT_SCORE → deviasi input dari referensi training
    drift_values = []
    for col, ref in REFERENCE_MEANS.items():
        if col in df.columns:
            drift_values.append(abs(df[col].mean() - ref))
    drift_score = min(sum(drift_values) / len(drift_values) / 3, 1.0) if drift_values else 0
    DATA_DRIFT_SCORE.set(drift_score)

    # THROUGHPUT → request per second sejak program berjalan
    elapsed = time.time() - start_time_global
    THROUGHPUT.set(request_count / elapsed if elapsed > 0 else 0)

    # MODEL_ACCURACY → nilai tetap dari hasil evaluasi test set
    MODEL_ACCURACY.set(0.8156)


def main():
    start_http_server(8000)
    print("[INFO] Prometheus exporter berjalan di http://127.0.0.1:8000/metrics")

    # Load test data for real streaming simulation
    test_data_path = r"D:\Pijak Dicoding\Project Membangun Sistem Machine Learning\Membangun_model\titanic_preprocessing\test.csv"
    try:
        test_df = pd.read_csv(test_data_path)
        if "Survived" in test_df.columns:
            test_df = test_df.drop("Survived", axis=1)
        print(f"[INFO] Berhasil memuat {len(test_df)} baris observasi nyata untuk disimulasikan.")
    except Exception as e:
        print(f"[ERROR] Gagal memuat data testing: {e}")
        return

    print("[INFO] Mulai mengirim request ke model menggunakan data observasi nyata...")

    idx = 0
    while True:
        # Ambil satu baris data observasi aktual
        row_df = test_df.iloc[[idx]]
        
        # Format ke bentuk JSON yang diterima MLflow
        payload = {
            "dataframe_records": row_df.to_dict(orient="records")
        }
        
        send_prediction_request(payload)
        update_system_metrics(payload)
        
        print(f"[INFO] Metrics updated from real data point {idx} - {time.strftime('%H:%M:%S')}")
        
        idx = (idx + 1) % len(test_df)
        time.sleep(2)

if __name__ == "__main__":
    main()