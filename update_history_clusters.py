# update_history_clusters.py

import logging
import os
from dotenv import load_dotenv
from typing import List, Dict
import pandas as pd # Import pandas untuk check NaN

# Impor service Supabase (pastikan path impor benar)
# Sesuaikan 'app.services' jika struktur folder Anda berbeda
try:
    from app.services import supabase_service
except ImportError:
    try:
        import supabase_service
    except ImportError:
        logging.error("Could not import supabase_service. Please check the import path.")
        exit()

# Konfigurasi dasar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Inisialisasi logger

load_dotenv()

# Inisialisasi Supabase Client (opsional)
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logger.error("Pastikan SUPABASE_URL dan SUPABASE_KEY ada di file .env")
    exit()

def run_history_update():
    """
    Mengambil hasil clustering final dari eq_metrics dan menggunakannya
    untuk mengisi kolom cluster & performance di eq_metrics_history.
    """
    logger.info("Starting update process for eq_metrics_history...")

    try:
        # 1. Ambil data final dari eq_metrics
        final_metrics = supabase_service.fetch_all_eq_metrics()
        if not final_metrics:
            logger.warning("No data found in eq_metrics. Ensure initial training has run. Aborting history update.")
            return

        # Buat mapping user_id -> {cluster, performance}
        user_final_state = {}
        for user_metric in final_metrics:
            user_id = user_metric.get('user_id')
            if user_id:
                 cluster_val = user_metric.get('cluster')
                 perf_val = user_metric.get('performance')
                 user_final_state[user_id] = {
                    'cluster': cluster_val,
                    'performance': perf_val
                 }
        logger.info(f"Loaded final state for {len(user_final_state)} users from eq_metrics.")

        # 2. Ambil semua data dari eq_metrics_history
        all_history_records = supabase_service.fetch_all_eq_metrics_history()
        if not all_history_records:
            logger.warning("No records found in eq_metrics_history to update.")
            return

        # 3. Siapkan batch update untuk history
        updates_history: List[Dict] = []
        updated_count = 0
        skipped_count = 0
        update_payload_check = {}

        for record in all_history_records:
            user_id = record.get('user_id')
            record_id = record.get('id')

            if not user_id or not record_id:
                skipped_count += 1
                continue

            if user_id in user_final_state:
                final_state = user_final_state[user_id]
                final_cluster = final_state.get('cluster')
                final_performance = final_state.get('performance')

                if final_performance is None or str(final_performance).strip() == '':
                     final_performance = None

                needs_update = False
                current_cluster = record.get('cluster')
                current_performance = record.get('performance')
                final_cluster_int = int(final_cluster) if final_cluster is not None else None

                if current_cluster != final_cluster_int:
                    needs_update = True
                if current_performance != final_performance:
                    needs_update = True

                if needs_update:
                     log_payload = {
                        'id': record_id,
                        # --- PERBAIKAN DI SINI: Tambahkan user_id ---
                        'user_id': user_id,
                        # --- AKHIR PERBAIKAN ---
                        'cluster': final_cluster_int,
                        'performance': final_performance
                     }
                     if user_id not in update_payload_check:
                         update_payload_check[user_id] = log_payload
                     updates_history.append(log_payload)
                     updated_count += 1
                else:
                    skipped_count += 1
            else:
                if record.get('cluster') is not None or record.get('performance') is not None:
                     updates_history.append({
                         'id': record_id,
                         # --- PERBAIKAN DI SINI: Tambahkan user_id ---
                         'user_id': user_id,
                         # --- AKHIR PERBAIKAN ---
                         'cluster': None,
                         'performance': None
                     })
                     updated_count += 1
                else:
                    skipped_count += 1

        if update_payload_check:
             sample_user = list(update_payload_check.keys())[0]
             logger.info(f"Sample update payload for user {sample_user}: {update_payload_check[sample_user]}")
        else:
             logger.info("No update payloads prepared.")

        # 4. Jalankan batch update
        if updates_history:
            logger.info(f"Attempting to update {updated_count} history records (skipped {skipped_count} unchanged/missing users).")
            supabase_service.update_eq_metrics_history_batch(updates_history)
        else:
            logger.info(f"No history records required updates (skipped {skipped_count}).")

        logging.info("✅ History update process finished.")

    except Exception as e:
        logging.error(f"❌ Error during history update: {e}", exc_info=True)

if __name__ == "__main__":
    logging.warning("Ensure calculate_initial_eq.py and run_initial_training.py have been run successfully before this script.")
    run_history_update()