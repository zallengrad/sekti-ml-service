import os
import re
import logging
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime

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

# Logika parsing error dengan kunci snake_case agar cocok dengan kolom database
# --- PERUBAHAN DI SINI ---
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
    'error_runtime_exception': r'RuntimeException',
    'error_constructor': r'constructor.*cannot be applied',
    'error_identifier_expected': r'<identifier> expected',
    'error_illegal_start_of_type': r'illegal start of type'
}
# --- AKHIR PERUBAHAN ---

def migrate_data():
    """
    Fungsi utama untuk membaca data dari ai_automated_feedbacks,
    memprosesnya, dan mengisikannya ke user_metrics.
    """
    logging.info("Memulai proses migrasi data historis...")

    try:
        # ... (kode Anda untuk mengambil dan mengelompokkan data tetap sama) ...
        # 1. Ambil semua data dari tabel feedback
        response = supabase.table("ai_automated_feedbacks").select("user_id, error_snapshot").execute()
        if not response.data:
            logging.warning("Tidak ada data di tabel 'ai_automated_feedbacks'. Proses dihentikan.")
            return

        feedbacks_df = pd.DataFrame(response.data)
        logging.info(f"Ditemukan {len(feedbacks_df)} total catatan feedback.")

        # 2. Kelompokkan data berdasarkan user_id
        grouped_by_user = feedbacks_df.groupby('user_id')
        
        all_metrics_to_upsert = []

        logging.info(f"Memproses data untuk {len(grouped_by_user)} pengguna unik...")

        # 3. Proses setiap pengguna
        for user_id, user_feedbacks in grouped_by_user:
            
            total_error_counts = {key: 0 for key in weights.keys()}
            
            total_submissions = len(user_feedbacks)

            for index, row in user_feedbacks.iterrows():
                error_snapshot = row['error_snapshot']
                if not error_snapshot or not isinstance(error_snapshot, str):
                    continue
                
                for error_type, pattern in patterns.items():
                    if re.search(pattern, error_snapshot.lower()):
                        total_error_counts[error_type] += 1
            
            # 4. Hitung fitur agregat untuk pengguna ini
            weighted_score = sum(total_error_counts[k] * v for k, v in weights.items())
            total_error_types = sum(1 for count in total_error_counts.values() if count > 0)
            error_submission_ratio = total_error_types / total_submissions if total_submissions > 0 else 0
            
            # Siapkan data untuk di-upsert
            metric_data = {
                "user_id": user_id,
                **total_error_counts,
                "weighted_error_score": weighted_score,
                "total_error_types": total_error_types,
                "error_submission_ratio": error_submission_ratio,
                "total_submissions": total_submissions,
                # --- PERBAIKAN DI SINI ---
                "updated_at": datetime.now().isoformat()
                # --- AKHIR PERBAIKAN ---
            }
            all_metrics_to_upsert.append(metric_data)
        
        # 5. Simpan semua data yang sudah diproses ke user_metrics
        if all_metrics_to_upsert:
            logging.info(f"Menyimpan {len(all_metrics_to_upsert)} data metrik ke tabel 'user_metrics'...")
            supabase.table("user_metrics").upsert(all_metrics_to_upsert).execute()
        
        logging.info("🎉 Migrasi data berhasil diselesaikan!")

    except Exception as e:
        logging.error(f"Terjadi kesalahan selama migrasi: {e}")

if __name__ == "__main__":
    migrate_data()