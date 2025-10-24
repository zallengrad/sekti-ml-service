import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import re
import logging
from . import supabase_service
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Variabel global untuk model
MODEL_PATH = "models/user_performance_model.pkl"
scaler = None
kmeans = None
perf_map = None

# == PERBAIKAN: Nama kolom dibuat konsisten (snake_case) di semua tempat ==
error_cols = [
    'weighted_error_score',
    'total_error_types',
    'error_submission_ratio'
]
weights = {
    'error_semicolon_expected': 1.0,
    'error_identifier_expected': 1.5,
    'error_cannot_find_symbol': 2.0,
    'error_constructor': 2.5,
    'error_runtime_exception': 2.0,
    'error_illegal_start_of_type': 1.5
}
patterns = {
    'error_cannot_find_symbol': r'cannot find symbol',
    'error_semicolon_expected': r'; expected',
    'error_runtime_exception': r'runtimeexception',
    'error_constructor': r'constructor.*cannot be applied',
    'error_identifier_expected': r'<identifier> expected',
    'error_illegal_start_of_type': r'illegal start of type'
}

def load_model():
    """Memuat model dari file .pkl atau menginisialisasi model baru jika tidak ada."""
    global scaler, kmeans, perf_map
    try:
        scaler, kmeans, perf_map, _ = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        logger.warning(f"{MODEL_PATH} not found. Initializing a new model.")
        scaler = StandardScaler()
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        perf_map = {0: 'High', 1: 'Medium', 2: 'Low'}

def parse_and_prepare_data(data):
    """Mengekstrak fitur dari data mentah dan mempersiapkannya untuk model."""
    snapshots = [data.error_snapshot] if data.error_snapshot else []
    total_submissions = data.submission_count if data.submission_count else len(snapshots)
    return aggregate_and_prepare_data(snapshots, total_submissions)

def aggregate_and_prepare_data(error_history, total_submissions):
    """
    Mengakumulasi error snapshot dan menyiapkan data fitur agregat
    sesuai dengan format databaru2.xlsx dan skema database (snake_case).
    """
    error_counts = {key: 0 for key in weights.keys()}
    valid_snapshots = [snap for snap in error_history if isinstance(snap, str) and snap.strip()]

    for snapshot in valid_snapshots:
        lowered_snapshot = snapshot.lower()
        for error_type, pattern in patterns.items():
            if re.search(pattern, lowered_snapshot):
                error_counts[error_type] += 1
    
    total_error_submissions = len(valid_snapshots)

    total_5_jenis_error = (
        error_counts.get('error_cannot_find_symbol', 0) +
        error_counts.get('error_semicolon_expected', 0) +
        error_counts.get('error_runtime_exception', 0) +
        error_counts.get('error_constructor', 0) +
        error_counts.get('error_identifier_expected', 0)
    )

    weighted_score = sum(error_counts[k] * v for k, v in weights.items())
    total_error_types = sum(1 for count in error_counts.values() if count > 0)
    error_submission_ratio = total_error_types / total_error_submissions if total_error_submissions > 0 else 0

    # --- PERUBAHAN UTAMA ADA DI SINI (Kunci diubah ke snake_case) ---
    processed_data = {
        "total_error_submissions": total_error_submissions,
        "jumlah_feedback_dibutuhkan": total_error_submissions,
        "feedback_easy": 0,
        "feedback_medium": 0,
        "feedback_hard": 0,
        "error_cannot_find_symbol": error_counts.get('error_cannot_find_symbol', 0),
        "error_semicolon_expected": error_counts.get('error_semicolon_expected', 0),
        "error_runtime_exception": error_counts.get('error_runtime_exception', 0),
        "error_constructor": error_counts.get('error_constructor', 0),
        "error_identifier_expected": error_counts.get('error_identifier_expected', 0),
        # Pastikan Anda punya kolom ini di skema Prisma
        "error_illegal_start_of_type": error_counts.get('error_illegal_start_of_type', 0), 
        "total_5_jenis_error": total_5_jenis_error,

        'weighted_error_score': weighted_score,
        'total_error_types': total_error_types,
        'error_submission_ratio': error_submission_ratio
    }
    
    return processed_data

def predict_performance(processed_data):
    """Melakukan prediksi cluster dan performa."""
    if not scaler or not kmeans:
        raise RuntimeError("Model is not loaded. Cannot perform prediction.")
        
    df = pd.DataFrame([processed_data])
    X_new = df[error_cols].fillna(0).values
    X_scaled = scaler.transform(X_new)
    
    cluster_label = kmeans.predict(X_scaled)[0]
    performance = perf_map.get(cluster_label, 'Medium')
    
    return performance, int(cluster_label)

def determine_optimal_clusters(X_scaled, preferred_k=3, max_k=5):
    """Menghitung silhouette scores untuk menentukan jumlah cluster yang optimal."""
    scores = []
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))
    
    optimal_k = np.argmax(scores) + 2
    logger.info(f"Silhouette scores (k=2 to {max_k}): {scores}")
    
    threshold = 0.1 
    if scores[optimal_k - 2] > scores[preferred_k - 2] + threshold:
        logger.info(f"Switching to optimal k={optimal_k} due to significantly better score.")
        return optimal_k, scores
    
    logger.info(f"Keeping preferred k={preferred_k} as per domain knowledge.")
    return preferred_k, scores

def retrain_model():
    """Fungsi untuk melatih ulang model KMeans dengan data terbaru."""
    logger.info("Starting model retraining...")
    global scaler, kmeans, perf_map
    
    data = supabase_service.fetch_all_user_metrics()
    if len(data) < 10:
        logger.warning("Not enough data to retrain model. Aborting.")
        return

    df = pd.DataFrame(data)
    
    if 'error_submission_ratio' not in df.columns:
        df['total_error_types'] = df[[key for key in weights.keys()]].gt(0).sum(axis=1)
        df['error_submission_ratio'] = df['total_error_types'] / df['total_submissions'].replace(0, 1)

    X_new = df[error_cols].fillna(0)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_new)
    
    optimal_k, scores = determine_optimal_clusters(X_scaled)
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # == PERBAIKAN: Menggunakan nama kolom snake_case yang konsisten ==
    cluster_order = df.groupby('Cluster')['weighted_error_score'].mean().sort_values().index
    labels = ['HIGH', 'MEDIUM', 'LOW']
    perf_map = {cluster_id: labels[i] for i, cluster_id in enumerate(cluster_order) if i < len(labels)}
    
    joblib.dump((scaler, kmeans, perf_map, error_cols), MODEL_PATH)
    logger.info(f"Model successfully retrained with {optimal_k} clusters and saved to {MODEL_PATH}.")

    updates = []
    metrics_columns = list(weights.keys()) + [
        'weighted_error_score',
        'total_error_types',
        'error_submission_ratio',
        'total_submissions'
    ]

    for index, row in df.iterrows():
        new_performance = perf_map.get(row['Cluster'])
        updates.append({
            'user_id': row['user_id'],
            'cluster': int(row['Cluster']),
            'performance': new_performance,
            # == PERBAIKAN UTAMA: Menambahkan timestamp secara eksplisit ==
            'updated_at': datetime.now(timezone.utc).isoformat()
        })
        previous_performance = row.get('performance')
        if previous_performance != new_performance:
            metrics_snapshot = {}
            for column in metrics_columns:
                if column in df.columns:
                    value = row[column]
                    if isinstance(value, (np.generic,)):
                        value = value.item()
                    metrics_snapshot[column] = value
            supabase_service.record_user_metrics_history(
                user_id=row['user_id'],
                performance=new_performance,
                cluster=int(row['Cluster']),
                metrics=metrics_snapshot
            )
    
    if updates:
        logger.info(f"Updating cluster and performance for {len(updates)} users in the database...")
        supabase_service.update_user_metrics_batch(updates)
    
    supabase_service.update_model_metadata({
        "id": 1,
        "optimal_k": optimal_k,
        "silhouette_scores": scores,
        "last_retrained_at": "now()"
    })
