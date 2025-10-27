from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import logging
from typing import Optional

# Impor services
from .services import prediction_service, supabase_service, eq_service # Tambahkan eq_service

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inisialisasi aplikasi FastAPI
app = FastAPI(
    title="User Performance EQ Classification API",
    description="An API to classify user performance based on coding errors using Error Quotient (EQ) and KMeans clustering.",
    version="2.0.0" # Naikkan versi
)

# Model input Pydantic (tidak berubah)
class UserData(BaseModel):
    user_id: str
    error_snapshot: str
    submission_count: int # Mungkin tidak relevan lagi, tapi biarkan dulu
    project_id: str
    code_snapshot: Optional[dict] = None

# Muat model saat startup
@app.on_event("startup")
def load_model_on_startup():
    try:
        prediction_service.load_model()
        logger.info("EQ machine learning model loaded successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to load ML model on startup: {e}", exc_info=True)
        # Pertimbangkan untuk menghentikan aplikasi jika model gagal load?

# Endpoint utama untuk klasifikasi
@app.post("/classify", tags=["Classification"])
async def classify_user(data: UserData):
    """
    Menerima data error pengguna, menghitung ulang EQ,
    dan mengklasifikasikan performa pengguna berdasarkan EQ rata-rata terbaru.
    """
    try:
        # 1. Simpan error snapshot terbaru (tetap penting untuk history)
        supabase_service.save_raw_error_snapshot(
            user_id=data.user_id,
            project_id=data.project_id,
            error_snapshot=data.error_snapshot,
            code_snapshot=data.code_snapshot
        )

        # 2. Proses ulang EQ (Ini akan insert new history dan update average_eq di eq_metrics)
        average_eq = eq_service.process_user_eq(data.user_id)

        # 3. Lakukan prediksi performa berdasarkan EQ rata-rata terbaru
        if average_eq is None or average_eq < 0:
             raise ValueError("Could not calculate a valid average EQ.")

        performance, cluster_label = prediction_service.predict_performance(average_eq)

        # 4. FINAL UPDATE: Sinkronkan cluster/performance ke eq_metrics dan SEMUA history records
        # Ini akan memastikan record history yang baru di-insert di langkah 2 terupdate
        supabase_service.final_prediction_update(
             user_id=data.user_id,
             cluster_label=cluster_label,
             performance=performance,
             average_eq=average_eq
        )

        logger.info(f"Successfully classified user {data.user_id} as '{performance}' in cluster {cluster_label} (Avg EQ: {average_eq:.4f})")
        return {"performance": performance, "cluster": cluster_label, "average_eq": average_eq}

    except RuntimeError as model_err:
        logger.error(f"Model Error during classification for user {data.user_id}: {model_err}")
        raise HTTPException(status_code=500, detail="Internal server error: Model not available.")
    except Exception as e:
        logger.error(f"Error during classification for user {data.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during classification.")

# Konfigurasi dan mulai scheduler untuk retraining (tidak berubah)
scheduler = BackgroundScheduler(timezone='Asia/Jakarta') # Pastikan timezone di set
scheduler.add_job(
    prediction_service.retrain_model,
    CronTrigger(hour=0, minute=0), # Setiap hari jam 00:00 WIB
    id="retrain_model_job",
    name="Daily EQ model retraining job",
    replace_existing=True
)
try:
    scheduler.start()
    logger.info("Scheduler started for daily EQ model retraining at 00:00 WIB.")
except Exception as e:
     logger.error(f"Failed to start scheduler: {e}", exc_info=True)


@app.on_event("shutdown")
def shutdown_scheduler():
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler has been shut down.")