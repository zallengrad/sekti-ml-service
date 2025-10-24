# migrate_metrics.py

import os
import logging
from datetime import datetime, timezone
from typing import List, Dict

import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# Impor layanan prediksi Anda yang sudah diperbarui
from app.services import prediction_service

# Konfigurasi dasar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Inisialisasi Supabase Client
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logging.error("Pastikan SUPABASE_URL dan SUPABASE_KEY ada di file .env")
    exit()

supabase: Client = create_client(supabase_url, supabase_key)

def fetch_all_feedbacks(batch_size: int = 1000) -> List[Dict]:
    """
    Mengambil seluruh catatan feedback dari Supabase, diurutkan berdasarkan pengguna dan waktu.
    PENTING: Mengambil `created_at` untuk merekonstruksi riwayat.
    """
    records: List[Dict] = []
    start = 0
    while True:
        end = start + batch_size - 1
        response = (
            supabase
            .table("ai_automated_feedbacks")
            .select("user_id, error_snapshot, created_at")
            .order("user_id")
            .order("created_at", desc=False) # Urutkan dari yang terlama ke terbaru
            .range(start, end)
            .execute()
        )
        batch = response.data or []
        records.extend(batch)
        if len(batch) < batch_size:
            break
        start += batch_size
    logging.info(f"Mengambil {len(records)} total feedback dari database.")
    return records

def migrate_and_reconstruct_history():
    """
    Membaca SELURUH riwayat error, membangun kembali data historis untuk 
    user_metrics_history, dan menyinkronkan data TERAKHIR ke user_metrics.
    """
    logging.info("Memulai proses migrasi dan rekonstruksi riwayat...")

    try:
        # 1. Muat model untuk mendapatkan prediksi
        prediction_service.load_model()
        logging.info("Model ML berhasil dimuat.")

        # 2. Ambil semua data mentah
        feedback_records = fetch_all_feedbacks()
        if not feedback_records:
            logging.warning("Tidak ada data di 'ai_automated_feedbacks'. Proses dihentikan.")
            return

        feedbacks_df = pd.DataFrame(feedback_records)
        grouped_by_user = feedbacks_df.groupby('user_id')

        final_user_metrics = []
        all_history_records = []
        
        logging.info(f"Memproses data untuk {len(grouped_by_user)} pengguna unik...")

        # 3. Iterasi per pengguna untuk membangun riwayat
        for user_id, user_feedbacks in grouped_by_user:
            # Pastikan diurutkan berdasarkan waktu
            user_feedbacks = user_feedbacks.sort_values(by='created_at').reset_index()
            
            # List untuk menyimpan snapshot error pengguna secara bertahap
            cumulative_snapshots = []
            
            for index, row in user_feedbacks.iterrows():
                # Tambahkan snapshot saat ini ke riwayat kumulatif
                snapshot = row['error_snapshot']
                if isinstance(snapshot, str) and snapshot.strip():
                    cumulative_snapshots.append(snapshot)

                # Hitung metrik berdasarkan riwayat HINGGA SAAT INI
                total_submissions = len(cumulative_snapshots)
                processed_data = prediction_service.aggregate_and_prepare_data(
                    cumulative_snapshots, 
                    total_submissions
                )
                
                # Lakukan prediksi performa
                performance, cluster = prediction_service.predict_performance(processed_data)

                # Siapkan data untuk tabel riwayat (user_metrics_history)
                history_record = {
                    "user_id": user_id,
                    "performance": performance,
                    "cluster": cluster,
                    **processed_data,
                    "recorded_at": row['created_at'] # <-- MENGGUNAKAN TIMESTAMP ASLI
                }
                all_history_records.append(history_record)

            # Setelah iterasi selesai, `history_record` terakhir adalah metrik terbaru pengguna
            if 'history_record' in locals():
                latest_metric = history_record.copy()
                # Ganti 'recorded_at' dengan 'updated_at' untuk tabel utama
                latest_metric['updated_at'] = latest_metric.pop('recorded_at')
                final_user_metrics.append(latest_metric)

        # 4. Hapus data lama dan simpan data baru yang sudah direkonstruksi
        if all_history_records:
            logging.info("Menghapus data lama dari 'user_metrics_history'...")
            supabase.table("user_metrics_history").delete().neq("user_id", "0").execute()
            
            logging.info(f"Menyimpan {len(all_history_records)} rekaman baru ke 'user_metrics_history'...")
            supabase.table("user_metrics_history").insert(all_history_records).execute()

        if final_user_metrics:
            logging.info(f"Menyimpan/memperbarui {len(final_user_metrics)} data metrik terbaru ke 'user_metrics'...")
            supabase.table("user_metrics").upsert(final_user_metrics).execute()

        logging.info("✅ Migrasi dan rekonstruksi riwayat selesai dengan sukses.")

    except Exception as e:
        logging.error(f"❌ Terjadi kesalahan selama migrasi: {e}", exc_info=True)


if __name__ == "__main__":
    migrate_and_reconstruct_history()