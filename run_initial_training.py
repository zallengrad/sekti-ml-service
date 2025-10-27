import logging
from app.services.prediction_service import retrain_model, load_model
from dotenv import load_dotenv
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Inisialisasi Supabase Client (opsional)
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    logging.error("Pastikan SUPABASE_URL dan SUPABASE_KEY ada di file .env")
    exit()

def run_training():
    """
    Memuat model dan menjalankan fungsi retraining EQ secara manual.
    Pastikan data EQ awal sudah dihitung sebelumnya.
    """
    print("Memuat resources model EQ...")
    try:
        load_model() # Inisialisasi variabel global yang dibutuhkan
        print("Memulai proses training awal model EQ...")
        retrain_model()
        print("Proses training awal model EQ selesai.")
        print("Periksa tabel 'eq_metrics' dan 'model_metadata' di database Anda.")
    except Exception as e:
        print(f"Error during initial training: {e}")
        logging.error("Error during initial training", exc_info=True)


if __name__ == "__main__":
    # Penting: Jalankan calculate_initial_eq.py SEBELUM menjalankan ini untuk pertama kali
    print("WARNING: Pastikan Anda sudah menjalankan 'calculate_initial_eq.py' sebelumnya!")
    run_training()