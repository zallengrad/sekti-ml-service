# app/services/supabase_service.py
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict
import re # Import re for fallback timezone parsing if needed

load_dotenv()
logger = logging.getLogger(__name__)

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.error("Supabase URL and Key must be set in the .env file.")
    raise ValueError("Supabase credentials not found.")

supabase: Client = create_client(supabase_url, supabase_key)

# --- Fungsi Inti ---
def save_raw_error_snapshot(user_id, project_id, error_snapshot, code_snapshot=None):
    """Menyimpan error snapshot mentah ke tabel ai_automated_feedbacks."""
    try:
        payload = {
            "user_id": user_id,
            "project_id": project_id,
            "error_snapshot": error_snapshot,
            "code_snapshot": code_snapshot if code_snapshot else {},
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        supabase.table("ai_automated_feedbacks").insert(payload).execute()
        logger.info(f"Saved raw error snapshot for user {user_id}.")
    except Exception as e:
        logger.error(f"Failed to save raw error snapshot for user {user_id}: {e}", exc_info=True)


# === Fungsi untuk EQ ===

def fetch_all_feedback_for_user(user_id: str, batch_size: int = 1000) -> List[Dict]:
    """Mengambil seluruh riwayat feedback (error snapshot & timestamp) untuk user tertentu."""
    records = []
    start = 0
    try:
        while True:
            response = (
                supabase
                .table("ai_automated_feedbacks")
                .select("user_id, project_id, error_snapshot, created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=False)
                .range(start, start + batch_size - 1)
                .execute()
            )
            batch = response.data or []
            records.extend(batch)
            if not batch or len(batch) < batch_size:
                break
            start += batch_size
        logger.info(f"Fetched {len(records)} feedback events for user {user_id}.")
    except Exception as e:
        logger.error(f"Failed to fetch feedback history for user {user_id}: {e}", exc_info=True)
    return records


def fetch_unique_users_from_feedback() -> List[str]:
    """Mengambil daftar user_id unik dari tabel feedback menggunakan query biasa."""
    logger.info("Fetching unique users from feedback (using improved fallback method)...")
    all_user_ids = set()
    start = 0
    batch_size = 1000 # Ukuran batch bisa disesuaikan

    try:
        while True:
            response = (
                supabase.table("ai_automated_feedbacks")
                .select("user_id")
                .range(start, start + batch_size - 1)
                .execute()
            )
            batch = response.data or []

            if not batch:
                break

            for item in batch:
                all_user_ids.add(item['user_id'])

            if len(batch) < batch_size:
                 break

            start += batch_size
            logger.debug(f"Fetched up to record {start}, found {len(all_user_ids)} unique users so far...")

        user_list = list(all_user_ids)
        logger.info(f"Finished fetching. Found {len(user_list)} unique users.")
        return user_list

    except Exception as e:
        logger.error(f"Failed to fetch unique users using fallback: {e}", exc_info=True)
        return []


def delete_eq_metrics_history(user_id: str):
    """Menghapus semua history EQ untuk user tertentu."""
    try:
        supabase.table("eq_metrics_history").delete().eq("user_id", user_id).execute()
        logger.debug(f"Deleted old EQ history for user {user_id}.")
    except Exception as e:
        logger.error(f"Failed to delete EQ history for user {user_id}: {e}", exc_info=True)


def insert_eq_metrics_history_batch(history_records: List[Dict]):
    """Menyimpan batch record history EQ."""
    if not history_records: return
    try:
        supabase.table("eq_metrics_history").insert(history_records).execute()
        user_id_sample = history_records[0].get('user_id', 'unknown')
        logger.info(f"Inserted {len(history_records)} EQ history records (sample user: {user_id_sample}).")
    except Exception as e:
        logger.error(f"Failed to insert EQ history batch: {e}", exc_info=True)


def upsert_eq_metrics(metrics_data: Dict):
    """Melakukan upsert (insert atau update) pada tabel eq_metrics."""
    try:
        supabase.table("eq_metrics").upsert(metrics_data).execute()
        logger.info(f"Upserted EQ metrics for user {metrics_data.get('user_id', 'N/A')}.")
    except Exception as e:
        user_id = metrics_data.get('user_id', 'N/A')
        logger.error(f"Failed to upsert EQ metrics for user {user_id}: {e}", exc_info=True)


def fetch_all_eq_metrics() -> List[Dict]:
    """Mengambil semua data dari tabel eq_metrics untuk retraining."""
    records = []
    try:
        response = supabase.table("eq_metrics").select("*").execute()
        records = response.data or []
        logger.info(f"Fetched {len(records)} records from eq_metrics.")
    except Exception as e:
        logger.error(f"Failed to fetch data from eq_metrics: {e}", exc_info=True)
    return records


def get_user_average_eq(user_id: str) -> Optional[float]:
    """Mengambil average_eq_score terbaru untuk user."""
    avg_eq = None
    try:
        response = supabase.table("eq_metrics").select("average_eq_score").eq("user_id", user_id).limit(1).maybe_single().execute()
        if response.data:
            avg_eq = response.data.get('average_eq_score')
    except Exception as e:
        logger.error(f"Failed to fetch average EQ for user {user_id}: {e}", exc_info=True)
    return avg_eq


def update_eq_metrics_batch(updates: list):
    """Melakukan batch update pada tabel eq_metrics (untuk cluster/performance)."""
    if not updates: return
    try:
        supabase.table("eq_metrics").upsert(updates).execute()
        logger.info(f"Successfully batch updated {len(updates)} eq_metrics records.")
    except Exception as e:
        logger.error(f"Failed to batch update eq_metrics: {e}", exc_info=True)


def fetch_all_eq_metrics_history() -> List[Dict]:
    """Mengambil semua data dari tabel eq_metrics_history."""
    records = []
    start = 0
    batch_size = 2000 # Sesuaikan ukuran batch
    try:
        while True:
            response = supabase.table("eq_metrics_history").select("*").range(start, start + batch_size - 1).execute()
            batch = response.data or []
            records.extend(batch)
            if not batch or len(batch) < batch_size:
                break
            start += batch_size
            logger.debug(f"Fetched {len(records)} history records so far...")
        logger.info(f"Fetched {len(records)} total records from eq_metrics_history.")
    except Exception as e:
        logger.error(f"Failed to fetch data from eq_metrics_history: {e}", exc_info=True)
    return records


def update_eq_metrics_history_batch(updates: List[Dict]):
    """Melakukan batch update pada tabel eq_metrics_history berdasarkan 'id'."""
    if not updates:
        logger.info("No history records provided for update.")
        return

    try:
        # Upsert berdasarkan primary key 'id' akan melakukan update
        # Pastikan setiap dict di 'updates' memiliki key 'id'
        supabase.table("eq_metrics_history").upsert(updates).execute()
        logger.info(f"Successfully batch updated {len(updates)} eq_metrics_history records.")
    except Exception as e:
        logger.error(f"Failed to batch update eq_metrics_history: {e}", exc_info=True)


def final_prediction_update(user_id: str, cluster_label: int, performance: str, average_eq: float):
    """
    Melakukan update terpadu:
    1. Update kolom cluster/performance di tabel eq_metrics (utama).
    2. Update kolom cluster/performance di SEMUA record eq_metrics_history milik user ini.
    """
    current_time_iso = datetime.now(timezone.utc).isoformat()

    # 1. Update eq_metrics (memastikan average_eq, cluster, dan performance final tersimpan)
    primary_metric_update = [{
        'user_id': user_id,
        'cluster': cluster_label,
        'performance': performance,
        'average_eq_score': average_eq,
        'last_calculated_at': current_time_iso
    }]
    # Gunakan upsert untuk memastikan data utama terupdate
    supabase.table("eq_metrics").upsert(primary_metric_update).execute()
    logger.info(f"Updated primary metrics for user {user_id}.")

    # 2. Update eq_metrics_history (semua record user ini, termasuk yang baru di-insert)
    history_records_to_update = []
    try:
        # Ambil semua ID history record user ini
        response = (
            supabase.table("eq_metrics_history")
            .select("id")
            .eq("user_id", user_id)
            .execute()
        )
        # Siapkan payload update untuk setiap ID history
        for record in response.data or []:
            history_records_to_update.append({
                'id': record['id'],
                'user_id': user_id, # Penting agar upsert tidak melanggar not-null
                'cluster': cluster_label,
                'performance': performance
            })

        if history_records_to_update:
            # Gunakan fungsi update history batch yang sudah Anda buat
            supabase.table("eq_metrics_history").upsert(history_records_to_update).execute()
            logger.info(f"Updated {len(history_records_to_update)} history records for user {user_id} with cluster {cluster_label}.")
        
    except Exception as e:
        logger.error(f"Failed to update history records for user {user_id} during prediction: {e}", exc_info=True)


# Fungsi delete_all_eq_metrics_history TIDAK disarankan karena menghapus semua history
# Jika memang diperlukan (misal sebelum insert ulang semua history), pastikan logikanya benar
# def delete_all_eq_metrics_history():
#     """Menghapus SEMUA data dari eq_metrics_history."""
#     try:
#         logger.warning("Attempting to delete ALL records from eq_metrics_history.")
#         # Tambahkan konfirmasi atau mekanisme pengaman lain jika perlu
#         supabase.table("eq_metrics_history").delete().gt("session_eq_score", -1).execute() # Filter agar tidak kosong
#         logger.info("Deleted ALL records from eq_metrics_history.")
#     except Exception as e:
#         logger.error(f"Failed to delete all eq_metrics_history: {e}", exc_info=True)


# --- Fungsi Metadata Model ---
def update_model_metadata(metadata: dict):
    """Menyimpan metadata hasil retraining model ke tabel model_metadata."""
    try:
        payload = {'id': 1, **metadata}
        # Gunakan nama tabel snake_case sesuai schema.prisma
        supabase.table("model_metadata").upsert(payload).execute()
        logger.info("Successfully updated model metadata in Supabase.")
    except Exception as e:
        logger.error(f"Failed to update model metadata: {e}", exc_info=True)