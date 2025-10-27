# app/services/prediction_service.py
import joblib
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score # Tidak digunakan jika K=3 dipaksa
import logging
from . import supabase_service # Impor supabase service
from datetime import datetime, timezone
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Variabel global untuk model
MODEL_PATH = "models/user_eq_performance_model_k3.pkl" # Nama model spesifik K=3
scaler = None
kmeans = None
perf_map = None # Map dari cluster index (0, 1, 2) ke performa ('HIGH', 'MEDIUM', 'LOW')
cluster_label_map = None # Map dari cluster index (0, 1, 2) ke label (1, 2, 3)

# == Nama fitur BARU untuk clustering ==
feature_cols = ['average_eq_score'] # Fitur utama adalah rata-rata EQ

def load_model():
    """Memuat model EQ K=3 dari file .pkl atau menginisialisasi model baru jika tidak ada."""
    global scaler, kmeans, perf_map, cluster_label_map
    try:
        scaler, kmeans, perf_map, cluster_label_map = joblib.load(MODEL_PATH)
        logger.info(f"EQ Model (K=3) loaded from {MODEL_PATH}")
        if kmeans is not None and kmeans.n_clusters != 3:
             logger.warning(f"Loaded model has {kmeans.n_clusters} clusters, expected 3. Re-initializing.")
             raise FileNotFoundError("Model K mismatch")
        if perf_map is None or cluster_label_map is None:
             logger.warning("Loaded model missing perf_map or cluster_label_map. Re-initializing.")
             raise FileNotFoundError("Missing maps")
    except (FileNotFoundError, ValueError, EOFError, TypeError, AttributeError) as e:
        logger.warning(f"{MODEL_PATH} not found or invalid ({e}). Initializing a new K=3 EQ model.")
        scaler = StandardScaler()
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        perf_map = {0: 'HIGH', 1: 'MEDIUM', 2: 'LOW'}
        cluster_label_map = {0: 1, 1: 2, 2: 3}
    except Exception as e:
        logger.error(f"Unexpected error loading EQ model: {e}. Initializing a new one.", exc_info=True)
        scaler = StandardScaler()
        kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
        perf_map = {0: 'HIGH', 1: 'MEDIUM', 2: 'LOW'}
        cluster_label_map = {0: 1, 1: 2, 2: 3}


def predict_performance(average_eq_score: float) -> Tuple[str, int]:
    """
    Melakukan prediksi cluster dan performa berdasarkan average_eq_score.
    Mengembalikan (performance_label, cluster_label_1_2_3).
    """
    global scaler, kmeans, perf_map, cluster_label_map
    if scaler is None or kmeans is None or perf_map is None or cluster_label_map is None:
         logger.error("EQ Model (K=3) components are not loaded properly. Loading default...")
         load_model()
         if scaler is None or kmeans is None or perf_map is None or cluster_label_map is None:
              raise RuntimeError("EQ Model (K=3) cannot be loaded. Cannot perform prediction.")

    if not hasattr(scaler, 'mean_') or scaler.mean_ is None:
        logger.warning("Scaler has not been fitted. Returning default prediction.")
        return 'MEDIUM', 2

    if not hasattr(kmeans, 'cluster_centers_') or kmeans.cluster_centers_ is None:
        logger.warning("KMeans has not been fitted. Returning default prediction.")
        return 'MEDIUM', 2

    try:
        X_new = np.array([[average_eq_score]])
        X_scaled = scaler.transform(X_new)
        cluster_index = kmeans.predict(X_scaled)[0] # Hasilnya 0, 1, atau 2
        performance = perf_map.get(int(cluster_index), 'MEDIUM')
        cluster_label = cluster_label_map.get(int(cluster_index), 2) # Hasilnya 1, 2, atau 3
        return performance, int(cluster_label)
    except Exception as e:
         logger.error(f"Error during prediction: {e}", exc_info=True)
         return 'MEDIUM', 2


def retrain_model():
    """Fungsi untuk melatih ulang model KMeans (K=3) dengan data EQ terbaru."""
    logger.info("Starting EQ model (K=3) retraining...")
    global scaler, kmeans, perf_map, cluster_label_map

    # 1. Ambil data EQ terbaru
    data = supabase_service.fetch_all_eq_metrics()
    if not data or len(data) < 3:
        logger.warning(f"Not enough data in eq_metrics ({len(data)} records, need >= 3) for K=3. Aborting.")
        return

    df = pd.DataFrame(data)
    df = df.dropna(subset=feature_cols)
    df = df[df['total_sessions_analyzed'] > 0]
    if df.shape[0] < 3:
        logger.warning(f"Not enough valid data ({df.shape[0]} records with sessions > 0, need >= 3) for K=3. Aborting.")
        return

    X_new = df[feature_cols].values

    # 2. Scaling
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_new)
    except ValueError as e:
        logger.error(f"Error during scaling: {e}. Check data variance. Aborting.", exc_info=True)
        return

    # 3. Latih KMeans dengan K=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    try:
        df['ClusterIndex'] = kmeans.fit_predict(X_scaled) # Menghasilkan index 0, 1, 2
    except Exception as e:
        logger.error(f"Error during K-Means fitting (K=3): {e}", exc_info=True)
        return

    # 4. Tentukan mapping cluster index (0, 1, 2) -> performa & label (1, 2, 3)
    try:
        cluster_order = df.groupby('ClusterIndex')['average_eq_score'].mean().sort_values(ascending=True).index
    except Exception as e:
        logger.error(f"Error grouping cluster means: {e}. Aborting.", exc_info=True)
        return

    performance_labels = ['HIGH', 'MEDIUM', 'LOW']
    cluster_labels = [1, 2, 3]

    perf_map = {}
    cluster_label_map = {}
    if len(cluster_order) == 3:
        for i, cluster_idx in enumerate(cluster_order):
            perf_map[int(cluster_idx)] = performance_labels[i]
            cluster_label_map[int(cluster_idx)] = cluster_labels[i]
        logger.info(f"New performance map (Index -> Perf): {perf_map}")
        logger.info(f"New cluster label map (Index -> Label 1-3): {cluster_label_map}")
    else:
        logger.error(f"KMeans did not produce 3 clusters (found {len(cluster_order)}). Aborting retraining.")
        scaler, kmeans, perf_map, cluster_label_map = None, None, None, None
        return

    df['Performance'] = df['ClusterIndex'].map(perf_map)
    df['ClusterLabel'] = df['ClusterIndex'].map(cluster_label_map) # Label 1, 2, 3

    # 5. Simpan model baru
    try:
        joblib.dump((scaler, kmeans, perf_map, cluster_label_map), MODEL_PATH)
        logger.info(f"EQ Model (K=3) successfully retrained and saved to {MODEL_PATH}.")
    except Exception as e:
        logger.error(f"Failed to save retrained EQ model (K=3): {e}", exc_info=True)
        return

    # 6. Siapkan data update untuk tabel eq_metrics
    updates_metrics = []
    current_time_iso = datetime.now(timezone.utc).isoformat()
    user_final_state = df.set_index('user_id')[['ClusterLabel', 'Performance']].to_dict('index')

    for index, row in df.iterrows():
        user_id = row['user_id']
        cluster_label_val = row.get('ClusterLabel')
        performance_val = row.get('Performance')
        if pd.isna(cluster_label_val): cluster_label_val = None
        if pd.isna(performance_val): performance_val = None

        updates_metrics.append({
            'user_id': user_id,
            'cluster': int(cluster_label_val) if cluster_label_val is not None else None,
            'performance': performance_val,
            'last_calculated_at': current_time_iso
        })

    # 7. Update batch ke eq_metrics di Supabase
    if updates_metrics:
        logger.info(f"Updating cluster (1,2,3) and performance for {len(updates_metrics)} users in eq_metrics...")
        supabase_service.update_eq_metrics_batch(updates_metrics)

    # 8. Update eq_metrics_history
    # ---- MULAI BAGIAN UPDATE HISTORY ----
    logger.info("--- Starting eq_metrics_history update ---")
    all_history_records = supabase_service.fetch_all_eq_metrics_history()
    updates_history = []
    if all_history_records:
        logger.info(f"Fetched {len(all_history_records)} history records to potentially update.")
        updated_count = 0
        skipped_count = 0
        for record in all_history_records:
            user_id = record.get('user_id')
            record_id = record.get('id')

            # Debugging: Log user_id dan record_id
            # logger.debug(f"Processing history record: ID={record_id}, User={user_id}")

            if user_id in user_final_state and record_id:
                 final_state = user_final_state[user_id]
                 final_cluster_label = final_state.get('ClusterLabel') # Hasilnya 1, 2, atau 3
                 final_performance = final_state.get('Performance') # Hasilnya 'HIGH', 'MEDIUM', 'LOW'

                 # Konversi tipe data final state untuk perbandingan
                 final_cluster_label_int = int(final_cluster_label) if final_cluster_label is not None else None

                 # Ambil nilai history saat ini
                 current_hist_cluster = record.get('cluster') # Bisa None atau int
                 current_hist_performance = record.get('performance') # Bisa None atau string

                 # Log nilai sebelum perbandingan
                 # logger.debug(f"User {user_id} - History: Cls={current_hist_cluster}, Perf={current_hist_performance} | Final: Cls={final_cluster_label_int}, Perf={final_performance}")

                 needs_update = False
                 if current_hist_cluster != final_cluster_label_int:
                      logger.debug(f"Cluster mismatch for hist_id {record_id} (User {user_id}): {current_hist_cluster} != {final_cluster_label_int}")
                      needs_update = True
                 if current_hist_performance != final_performance:
                      logger.debug(f"Performance mismatch for hist_id {record_id} (User {user_id}): {current_hist_performance} != {final_performance}")
                      needs_update = True

                 if needs_update:
                     updates_history.append({
                         'id': record_id, # Kunci utama untuk update
                         'cluster': final_cluster_label_int,
                         'performance': final_performance
                     })
                     updated_count += 1
                 else:
                     skipped_count += 1
                     # logger.debug(f"No update needed for history record ID={record_id}")

            else:
                 logger.warning(f"Skipping history record update: User {user_id} not in final state or record ID {record_id} missing.")
                 skipped_count += 1

        logger.info(f"Prepared {len(updates_history)} updates for history table ({skipped_count} skipped).")
        # Lakukan batch update pada history berdasarkan ID
        if updates_history:
            logger.info(f"Submitting {len(updates_history)} history updates to Supabase...")
            supabase_service.update_eq_metrics_history_batch(updates_history) # Fungsi ini harus ada di supabase_service.py
        else:
            logger.info("No history records needed updating.")
    else:
        logger.warning("No history records found to update.") # Ubah log level ke warning
    logger.info("--- Finished eq_metrics_history update ---")
    # ---- AKHIR BAGIAN UPDATE HISTORY ----


    # 9. Update metadata model
    current_time_iso = datetime.now(timezone.utc).isoformat()
    try:
        metadata_payload = {
            "optimal_k": 3, # K=3
            "silhouette_scores": [], # Kosongkan
            "last_retrained_at": current_time_iso
        }
        supabase_service.update_model_metadata(metadata_payload)
    except Exception as meta_e:
        logger.error(f"Failed to update model metadata: {meta_e}", exc_info=True)

    logger.info("✅ EQ Model (K=3) retraining finished successfully.")