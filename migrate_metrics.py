import os
import logging
from datetime import datetime, timezone
from typing import List

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

from app.services import prediction_service

# Konfigurasi dasar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Inisialisasi Supabase Client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logging.error("Pastikan SUPABASE_URL dan SUPABASE_KEY ada di file .env")
    exit()

supabase: Client = create_client(supabase_url, supabase_key)


def aggregate_history(snapshots: List[str]) -> dict:
    """Helper agar script migrasi memakai logika agregasi yang sama dengan endpoint."""
    total_submissions = len(snapshots)
    return prediction_service.aggregate_and_prepare_data(snapshots, total_submissions)


def fetch_all_feedbacks(batch_size: int = 1000) -> List[dict]:
    """Mengambil seluruh catatan feedback dari Supabase tanpa batas 1000 default."""
    records: List[dict] = []
    start = 0
    while True:
        end = start + batch_size - 1
        response = (
            supabase
            .table("ai_automated_feedbacks")
            .select("user_id, error_snapshot")
            .range(start, end)
            .execute()
        )
        batch = response.data or []
        records.extend(batch)
        if len(batch) < batch_size:
            break
        start += batch_size
    return records


def migrate_data():
    """
    Membaca seluruh riwayat error dari ai_automated_feedbacks, menghitung agregasi,
    menjalankan prediksi performa, dan menyinkronkan hasilnya ke user_metrics.
    """
    logging.info("Memulai proses migrasi data historis...")

    try:
        prediction_service.load_model()

        feedback_records = fetch_all_feedbacks()
        if not feedback_records:
            logging.warning("Tidak ada data di tabel 'ai_automated_feedbacks'. Proses dihentikan.")
            return

        feedbacks_df = pd.DataFrame(feedback_records)
        logging.info(f"Ditemukan {len(feedbacks_df)} total catatan feedback.")

        grouped_by_user = feedbacks_df.groupby('user_id')

        upserts = []
        logging.info(f"Memproses data untuk {len(grouped_by_user)} pengguna unik...")

        for user_id, user_feedbacks in grouped_by_user:
            snapshots = [
                snapshot for snapshot in user_feedbacks['error_snapshot'].tolist()
                if isinstance(snapshot, str) and snapshot.strip()
            ]

            if not snapshots:
                logging.debug(f"Melewati user {user_id} karena tidak ada snapshot valid.")
                continue

            processed_data = aggregate_history(snapshots)
            performance, cluster = prediction_service.predict_performance(processed_data)

            metric_data = {
                "user_id": user_id,
                "performance": performance,
                "cluster": cluster,
                **processed_data,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            upserts.append(metric_data)

        if upserts:
            logging.info(f"Menyimpan {len(upserts)} data metrik ke tabel 'user_metrics'...")
            supabase.table("user_metrics").upsert(upserts).execute()

        logging.info("Migrasi data selesai.")

    except Exception as e:
        logging.error(f"Terjadi kesalahan selama migrasi: {e}")


if __name__ == "__main__":
    migrate_data()
