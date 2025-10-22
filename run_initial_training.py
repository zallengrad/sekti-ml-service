import logging
from app.services.prediction_service import retrain_model, load_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_training():
    """
    Memuat model dan menjalankan fungsi retraining secara manual.
    """
    print("Memuat resources model...")
    load_model() # Inisialisasi variabel global yang dibutuhkan
    print("Memulai proses training awal...")
    retrain_model()
    print("Proses training awal selesai. Periksa database Anda.")

if __name__ == "__main__":
    run_training()