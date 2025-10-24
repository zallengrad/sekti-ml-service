import os
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone # Pastikan ini di-import
from typing import Optional

# Muat environment variables dari file .env
load_dotenv()
logger = logging.getLogger(__name__)

# Ambil URL dan Key dari environment
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.error("Supabase URL and Key must be set in the .env file.")
    raise ValueError("Supabase credentials not found.")

# Inisialisasi Supabase client
supabase: Client = create_client(supabase_url, supabase_key)

def fetch_all_user_metrics():
    """Mengambil semua data dari tabel user_metrics untuk retraining."""
    try:
        response = supabase.table("user_metrics").select("*").execute()
        logger.info(f"Fetched {len(response.data)} records from user_metrics.")
        return response.data
    except Exception as e:
        logger.error(f"Failed to fetch data from Supabase: {e}")
        return []

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
        logger.error(f"Failed to save raw error snapshot for user {user_id}: {e}")

def fetch_user_error_history(user_id, batch_size: int = 1000):
    """Mengambil seluruh riwayat error_snapshot untuk user tertentu."""
    try:
        records = []
        start = 0
        while True:
            end = start + batch_size - 1
            response = (
                supabase
                .table("ai_automated_feedbacks")
                .select("error_snapshot, created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=False)
                .range(start, end)
                .execute()
            )
            batch = response.data or []
            records.extend(batch)
            if len(batch) < batch_size:
                break
            start += batch_size

        logger.info(f"Fetched {len(records)} error snapshots for user {user_id}.")
        return records
    except Exception as e:
        logger.error(f"Failed to fetch error history for user {user_id}: {e}")
        return []


def record_user_metrics_history(user_id: str, performance, cluster, metrics: Optional[dict] = None):
    """Menyimpan snapshot metrik ke tabel user_metrics_history."""
    try:
        snapshot = {
            "user_id": user_id,
            "performance": performance,
            "cluster": cluster,
            **(dict(metrics) if metrics else {}),
            "recorded_at": datetime.now(timezone.utc).isoformat()
        }
        supabase.table("user_metrics_history").insert(snapshot).execute()
        logger.info(f"Recorded metrics history for user {user_id}.")
    except Exception as e:
        logger.error(f"Failed to record metrics history for user {user_id}: {e}")

def save_feedback_and_metrics(user_id, project_id, error_snapshot, code_snapshot, processed_data, performance, cluster):
    """Menyimpan data feedback dan metrik pengguna ke Supabase."""
    try:
        metric_data = {
            "user_id": user_id,
            "performance": performance,
            "cluster": cluster,
            **processed_data, # <-- Ini akan memasukkan semua kolom baru secara otomatis
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        supabase.table("user_metrics").upsert(metric_data).execute()
        record_user_metrics_history(
            user_id=user_id,
            performance=performance,
            cluster=cluster,
            metrics=processed_data
        )
        
        logger.info(f"Successfully saved metrics for user {user_id} to Supabase.")

    except Exception as e:
        logger.error(f"Failed to save data to Supabase for user {user_id}: {e}")
        # Kita tidak perlu mengembalikan nilai karena endpoint utama akan menangani error

def update_model_metadata(metadata: dict):
    """Menyimpan metadata hasil retraining model."""
    try:
        supabase.table("model_metadata").upsert(metadata).execute()
        logger.info("Successfully updated model metadata in Supabase.")
    except Exception as e:
        logger.error(f"Failed to update model metadata: {e}")


def update_user_metrics_batch(updates: list):
    """Melakukan batch update pada tabel user_metrics."""
    try:
        supabase.table("user_metrics").upsert(updates).execute()
        logger.info(f"Successfully updated {len(updates)} user metrics records.")
    except Exception as e:
        logger.error(f"Failed to batch update user metrics: {e}")
