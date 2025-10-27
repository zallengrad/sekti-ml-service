# calculate_initial_eq.py

import logging
import os
from dotenv import load_dotenv

# Impor service EQ dan Supabase
from app.services import eq_service, supabase_service # Pastikan path impor benar

# Konfigurasi dasar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Inisialisasi Supabase Client (opsional jika sudah di service, tapi bagus untuk standalone script)
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logging.error("Pastikan SUPABASE_URL dan SUPABASE_KEY ada di file .env")
    exit()

# Script ini hanya memanggil fungsi kalkulasi historis dari eq_service
if __name__ == "__main__":
    logging.info("Starting initial calculation of Error Quotient for all users...")
    # Panggil fungsi yang menghitung ulang EQ untuk semua pengguna
    eq_service.calculate_historical_eq_all_users()
    logging.info("Initial EQ calculation process completed.")
    logging.info("Please check the 'eq_metrics' and 'eq_metrics_history' tables in your database.")