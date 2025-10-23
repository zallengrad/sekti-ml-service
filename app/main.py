from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
from typing import Optional # <-- TAMBAHAN 1: Impor Optional
from .services import prediction_service, supabase_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="User Performance Classification API",
    description="An API to classify user performance based on coding errors using KMeans clustering.",
    version="1.0.0"
)

# Model input Pydantic
class UserData(BaseModel):
    user_id: str
    error_snapshot: str
    submission_count: int
    project_id: str
    # --- PERBAIKAN DI SINI ---
    code_snapshot: Optional[dict] = None # Menggunakan Optional[dict] agar kompatibel

# Muat model saat startup
@app.on_event("startup")
def load_model_on_startup():
    prediction_service.load_model()
    logger.info("Machine learning model loaded successfully.")

# Endpoint utama untuk klasifikasi
@app.post("/classify", tags=["Classification"])
async def classify_user(data: UserData):
    """
    Menerima data error pengguna, memprosesnya, dan mengklasisikasikan performa pengguna.
    """
    try:
        # 1. Simpan error snapshot terbaru
        supabase_service.save_raw_error_snapshot(
            user_id=data.user_id,
            project_id=data.project_id,
            error_snapshot=data.error_snapshot,
            code_snapshot=data.code_snapshot
        )

        # 2. Ambil seluruh riwayat error untuk pengguna
        history_records = supabase_service.fetch_user_error_history(data.user_id)
        error_history = [record.get("error_snapshot") for record in history_records if record.get("error_snapshot")]
        total_submissions = len(history_records)

        # 3. Proses data riwayat menjadi agregat
        processed_data = prediction_service.aggregate_and_prepare_data(error_history, total_submissions)

        # 4. Lakukan prediksi performa
        performance, cluster_label = prediction_service.predict_performance(processed_data)
        
        # 5. Simpan hasil ke Supabase
        supabase_service.save_feedback_and_metrics(
            user_id=data.user_id,
            project_id=data.project_id,
            error_snapshot=data.error_snapshot,
            code_snapshot=data.code_snapshot,
            processed_data=processed_data,
            performance=performance,
            cluster=cluster_label
        )
        
        logger.info(f"Successfully classified user {data.user_id} as '{performance}' in cluster {cluster_label}")
        return {"performance": performance, "cluster": cluster_label}

    except Exception as e:
        logger.error(f"Error during classification for user {data.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during classification.")

# Konfigurasi dan mulai scheduler untuk retraining
scheduler = BackgroundScheduler()
scheduler.add_job(
    prediction_service.retrain_model, 
    CronTrigger(hour=0, minute=0, timezone='Asia/Jakarta'), # Setiap hari jam 00:00 WIB
    id="retrain_model_job",
    name="Daily model retraining job",
    replace_existing=True
)
scheduler.start()
logger.info("Scheduler started for daily model retraining at 00:00 WIB.")

@app.on_event("shutdown")
def shutdown_scheduler():
    scheduler.shutdown()
    logger.info("Scheduler has been shut down.")
