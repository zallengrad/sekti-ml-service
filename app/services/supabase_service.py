import os
from supabase import create_client, Client
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone # Pastikan ini di-import

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

def save_feedback_and_metrics(user_id, project_id, error_snapshot, code_snapshot, processed_data, performance, cluster):
    """Menyimpan data feedback dan metrik pengguna ke Supabase."""
    try:
        metric_data = {
            "user_id": user_id,
            "performance": performance,
            "cluster": cluster,
            **processed_data,
            # == PERBAIKAN UTAMA ADA DI SINI ==
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        supabase.table("user_metrics").upsert(metric_data).execute()
        
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
