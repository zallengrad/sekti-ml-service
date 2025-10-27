# migrate_metrics.py

import os
import logging
from datetime import datetime, timezone
from typing import List, Dict
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# Impor layanan prediksi Anda yang sudah diperbarui
from app.services import prediction_service
from app.services import supabase_service # Impor juga supabase_service

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logging.error("Pastikan SUPABASE_URL dan SUPABASE_KEY ada di file .env")
    exit()

supabase: Client = create_client(supabase_url, supabase_key)

def migrate_and_reconstruct_eq_history():
    """
    Membaca SELURUH riwayat feedback, membangun kembali riwayat EQ per sesi
    untuk eq_metrics_history, dan menyinkronkan data agregat TERAKHIR ke eq_metrics.
    """
    logging.info("Memulai proses migrasi dan rekonstruksi riwayat EQ...")

    try:
        # 1. Muat model (diperlukan untuk mapping performa akhir)
        #    Jika model belum ada, ini akan menginisialisasi yang baru.
        #    Retraining sebenarnya akan dilakukan *setelah* data historis dihitung.
        prediction_service.load_model()
        logging.info("Model ML (akan dilatih ulang nanti) berhasil dimuat/diinisialisasi.")

        # 2. Ambil semua data mentah feedback
        all_feedback_records = supabase_service.fetch_all_raw_feedbacks()
        if not all_feedback_records:
            logging.warning("Tidak ada data di 'ai_automated_feedbacks'. Proses dihentikan.")
            return

        feedbacks_df = pd.DataFrame(all_feedback_records)
        grouped_by_user = feedbacks_df.groupby('user_id')

        final_user_eq_metrics = [] # Untuk tabel eq_metrics
        all_history_records = []   # Untuk tabel eq_metrics_history

        logging.info(f"Memproses data untuk {len(grouped_by_user)} pengguna unik...")

        # 3. Iterasi per pengguna untuk membangun riwayat EQ
        for user_id, user_feedbacks_df in grouped_by_user:
            user_history = user_feedbacks_df.to_dict('records')
            sessions = prediction_service.group_into_sessions(user_history) # Gunakan fungsi dari service

            cumulative_session_eqs = [] # Lacak skor EQ sesi untuk rata-rata kumulatif

            for i, session in enumerate(sessions):
                session_eq = prediction_service.calculate_session_eq(session) # Gunakan fungsi dari service
                if session_eq is not None:
                    # Dapatkan detail sesi
                    try:
                        session_start = datetime.fromisoformat(session[0]['created_at'].replace('Z', '+00:00'))
                        session_end = datetime.fromisoformat(session[-1]['created_at'].replace('Z', '+00:00'))
                    except (IndexError, KeyError, ValueError) as e:
                         logging.warning(f"Gagal memproses timestamp sesi untuk user {user_id}: {e}. Melewati sesi.")
                         continue

                    session_id = f"{user_id}_{session_start.timestamp()}"
                    cumulative_session_eqs.append(session_eq)
                    cumulative_avg_eq = np.mean(cumulative_session_eqs) if cumulative_session_eqs else 0.0

                    # Siapkan data untuk tabel riwayat (eq_metrics_history)
                    # Cluster & Performance belum diketahui saat ini
                    history_record = {
                        "user_id": user_id,
                        "session_id": session_id,
                        "session_start_time": session_start.isoformat(),
                        "session_end_time": session_end.isoformat(),
                        "session_event_count": len(session),
                        "session_eq_score": session_eq,
                        "cumulative_average_eq_score": cumulative_avg_eq,
                        "cluster": None, # Akan diisi setelah retraining
                        "performance": None, # Akan diisi setelah retraining
                        "recorded_at": session_end.isoformat() # Gunakan waktu akhir sesi
                    }
                    all_history_records.append(history_record)

            # Setelah iterasi sesi selesai, hitung metrik agregat terakhir pengguna
            if cumulative_session_eqs:
                final_avg_eq = np.mean(cumulative_session_eqs)
                final_metric = {
                    "user_id": user_id,
                    "average_eq_score": float(final_avg_eq),
                    "total_sessions": len(cumulative_session_eqs),
                    "cluster": None,      # Akan diisi setelah retraining
                    "performance": None,  # Akan diisi setelah retraining
                    # updated_at akan diisi saat upsert/retraining nanti
                }
                final_user_eq_metrics.append(final_metric)

        # 4. Simpan metrik agregat (tanpa cluster/perf) ke eq_metrics sementara
        if final_user_eq_metrics:
            logging.info(f"Menyimpan {len(final_user_eq_metrics)} metrik EQ agregat awal ke 'eq_metrics'...")
            # Gunakan upsert untuk menangani pengguna yang mungkin sudah ada
            for metric in final_user_eq_metrics:
                metric['updated_at'] = datetime.now(timezone.utc).isoformat()
            supabase_service.upsert_eq_metrics_batch(final_user_eq_metrics)

        # 5. Lakukan Retraining Model berdasarkan metrik agregat yang baru dihitung
        logging.info("Memulai retraining model berdasarkan metrik EQ yang direkonstruksi...")
        prediction_service.retrain_model() # Fungsi retrain akan mengambil data dari DB, melatih, dan mengupdate cluster/perf di eq_metrics dan eq_metrics_history

        logging.info("✅ Migrasi, rekonstruksi riwayat EQ, dan retraining model selesai dengan sukses.")

    except Exception as e:
        logging.error(f"❌ Terjadi kesalahan selama migrasi EQ: {e}", exc_info=True)


if __name__ == "__main__":
    migrate_and_reconstruct_eq_history()