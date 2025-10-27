import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

# Impor layanan supabase (asumsi struktur folder Anda)
from . import supabase_service

logger = logging.getLogger(__name__)

def parse_flexible_isoformat(ts_str: Optional[str]) -> Optional[datetime]:
    """Mencoba parsing string ISO format dengan mikrodetik yang bervariasi."""
    if not ts_str:
        return None
    try:
        # Hapus 'Z' jika ada dan tambahkan timezone UTC
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1] + '+00:00'
        # Tambahkan timezone jika tidak ada
        elif '+' not in ts_str and '-' not in ts_str[10:]: # Cek timezone setelah tanggal
             ts_str += '+00:00' # Asumsikan UTC jika tidak ada timezone

        # Cek bagian mikrodetik
        if '.' in ts_str:
            base, micro = ts_str.split('.', 1)
            # Ambil timezone jika ada setelah mikrodetik
            tz_part = ''
            if '+' in micro:
                micro, tz_part = micro.split('+', 1)
                tz_part = '+' + tz_part
            elif '-' in micro:
                 micro, tz_part = micro.split('-', 1)
                 tz_part = '-' + tz_part

            # Pad mikrodetik menjadi 6 digit
            micro = micro.ljust(6, '0')[:6] # Ambil 6 digit pertama setelah padding
            ts_str = f"{base}.{micro}{tz_part}"

        return datetime.fromisoformat(ts_str)
    except ValueError as e:
        logger.error(f"Failed to parse timestamp '{ts_str}': {e}")
        # Coba format tanpa mikrodetik sebagai fallback
        try:
             # Coba parse tanpa bagian mikrodetik
             base_part = ts_str.split('.')[0]
             tz_part = ''
             if '+' in ts_str:
                 tz_part = '+' + ts_str.split('+', 1)[1]
             elif '-' in ts_str[10:]:
                 tz_part = '-' + ts_str.split('-', 1)[-1] # Ambil setelah T dan :
                 # Perlu parsing timezone manual jika formatnya rumit
                 if len(tz_part) > 6 : # e.g., "2023-10-27T10:00:00-07:00" -> tz_part = "-07:00"
                    # Cari posisi timezone
                    match_tz = re.search(r'([-+]\d{2}:\d{2})$', ts_str)
                    if match_tz:
                        tz_part = match_tz.group(1)
                        base_part = ts_str[:match_tz.start()] # Ambil bagian sebelum timezone

             if not tz_part: tz_part = '+00:00' # Default UTC

             # Coba parse base part saja
             dt_naive = datetime.fromisoformat(base_part)
             # Buat aware (perlu import pytz jika timezone kompleks, tapi coba manual dulu)
             offset_str = tz_part.replace(':', '') # e.g., +0700
             offset_hours = int(offset_str[:3])
             offset_minutes = int(offset_str[0] + offset_str[3:]) # +/- minutes
             tz = timezone(timedelta(hours=offset_hours, minutes=offset_minutes))
             return dt_naive.replace(tzinfo=tz)

        except Exception as fallback_e:
             logger.error(f"Fallback parsing also failed for '{ts_str}': {fallback_e}")
             return None
    except Exception as general_e:
         logger.error(f"Unexpected error parsing timestamp '{ts_str}': {general_e}")
         return None

# Pola regex untuk mengekstrak tipe error dan baris dari pesan error Java umum
# Sesuaikan pola ini jika format error_snapshot Anda berbeda
ERROR_LINE_PATTERN = re.compile(r'^(.*?):(\d+): error: (.*)', re.MULTILINE)

# Pola error yang sudah ada dari prediction_service (untuk tipe error)
patterns = {
    'error_cannot_find_symbol': r'cannot find symbol',
    'error_semicolon_expected': r'; expected',
    'error_runtime_exception': r'runtimeexception', # Mungkin tidak relevan untuk EQ sintaks
    'error_constructor': r'constructor.*cannot be applied',
    'error_identifier_expected': r'<identifier> expected',
    'error_illegal_start_of_type': r'illegal start of type',
    # Tambahkan pola lain jika perlu
    'bracket_expected': r'[{(] expected|illegal start of expression', # Gabungkan beberapa pola jika perlu
    'class_or_interface_expected': r'class or interface expected',
    'dot_class_expected': r'\.class expected',
    'not_a_statement': r'not a statement',
    'missing_return': r'missing return statement|missing return value',
    'incompatible_types': r'incompatible types',
    'private_access_violation': r'has private access',
    'method_application_error': r'cannot be applied to|actual and formal argument lists differ',
}

# Parameter EQ (berdasarkan Tabel 4.2 - From Search)
ETYPE_SAME_PENALTY = 11
ETYPE_DIFF_PENALTY = 8
MAX_PENALTY = max(ETYPE_SAME_PENALTY, ETYPE_DIFF_PENALTY) # Untuk normalisasi




def parse_error_details(error_snapshot: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Mencoba mengekstrak tipe error pertama dan nomor baris dari error_snapshot.
    Mengembalikan (error_type, error_line) atau (None, None) jika tidak ada error sintaks yang dikenali.
    """
    if not isinstance(error_snapshot, str) or not error_snapshot.strip():
        return None, None

    # Coba cocokkan dengan pola error Java umum untuk mendapatkan baris
    match = ERROR_LINE_PATTERN.search(error_snapshot)
    error_line = int(match.group(2)) if match else None
    error_message_lower = error_snapshot.lower()

    # Identifikasi tipe error berdasarkan pola yang ada
    error_type = None
    for type_key, pattern in patterns.items():
        if re.search(pattern, error_message_lower):
            error_type = type_key
            break # Ambil tipe error pertama yang cocok

    # Hanya kembalikan jika kita berhasil mengidentifikasi tipe error sintaks
    if error_type:
        # Jika baris tidak ketemu dari pola utama, coba cari nomor baris lain
        if error_line is None:
             line_matches = re.findall(r':(\d+):', error_snapshot)
             if line_matches:
                 try:
                    error_line = int(line_matches[0]) # Ambil nomor baris pertama yang ditemukan
                 except ValueError:
                    error_line = None
        return error_type, error_line
    else:
        # Jika tidak ada pola error yang cocok, anggap bukan error sintaks yang relevan untuk EQ
        return None, None

def identify_sessions(user_events: List[Dict], max_gap_minutes: int = 30) -> List[List[Dict]]:
    """
    Mengelompokkan event error pengguna menjadi sesi berdasarkan jeda waktu.
    """
    if not user_events:
        return []

    # Parse timestamp DULU dengan fungsi fleksibel
    parsed_events = []
    for event in user_events:
        event_time = parse_flexible_isoformat(event.get('created_at'))
        if event_time:
            event['parsed_time'] = event_time # Simpan waktu yang sudah di-parse
            parsed_events.append(event)
        else:
            logger.warning(f"Skipping event due to unparseable timestamp: {event.get('created_at')} for user {event.get('user_id')}")

    if not parsed_events:
        return []

    # Urutkan event berdasarkan waktu yang sudah di-parse
    parsed_events.sort(key=lambda x: x['parsed_time'])

    sessions = []
    current_session = []
    last_event_time = None

    for event in parsed_events:
        event_time = event['parsed_time']

        if last_event_time and (event_time - last_event_time) > timedelta(minutes=max_gap_minutes):
            if current_session:
                sessions.append(current_session)
            current_session = [event]
        else:
            current_session.append(event)

        last_event_time = event_time

    if current_session:
        sessions.append(current_session)

    # Hapus 'parsed_time' jika tidak ingin disimpan di history
    # for session in sessions:
    #     for event in session:
    #         event.pop('parsed_time', None)

    return sessions


def calculate_session_eq(session_events: List[Dict]) -> Tuple[float, Optional[datetime], Optional[datetime]]:
    """
    Menghitung skor EQ untuk satu sesi pemrograman.
    Mengembalikan (eq_score, session_start, session_end).
    """
    # Ambil waktu dari 'parsed_time' yang sudah ada
    if len(session_events) < 2:
        start_time = session_events[0]['parsed_time'] if session_events and 'parsed_time' in session_events[0] else None
        end_time = start_time
        return 0.0, start_time, end_time

    pair_scores = []
    session_start = session_events[0]['parsed_time']
    session_end = session_events[-1]['parsed_time']

    for i in range(1, len(session_events)):
        prev_event = session_events[i-1]
        curr_event = session_events[i]

        prev_error_type, prev_error_line = parse_error_details(prev_event.get('error_snapshot'))
        curr_error_type, curr_error_line = parse_error_details(curr_event.get('error_snapshot'))

        # Asumsi: Data hanya berisi event error sintaks yang terdeteksi
        # Jika salah satu event tidak menghasilkan error sintaks yang dikenali, skor pasangan = 0
        if prev_error_type is None or curr_error_type is None:
            pair_score = 0
        else:
            # Keduanya adalah error sintaks yang dikenali
            if prev_error_type == curr_error_type:
                pair_score = ETYPE_SAME_PENALTY
            else:
                pair_score = ETYPE_DIFF_PENALTY

        # Normalisasi skor pasangan
        normalized_score = pair_score / MAX_PENALTY if MAX_PENALTY > 0 else 0
        pair_scores.append(normalized_score)

    # EQ Sesi adalah rata-rata skor pasangan ternormalisasi
    session_eq_score = np.mean(pair_scores) if pair_scores else 0.0

    return float(session_eq_score), session_start, session_end

def process_user_eq(user_id: str):
    """
    Memproses semua feedback untuk user, menghitung EQ per sesi,
    menyimpan riwayat, dan memperbarui metrik EQ agregat.
    """
    logger.info(f"Processing EQ for user {user_id}...")
    try:
        # 1. Ambil semua feedback untuk user
        user_events = supabase_service.fetch_all_feedback_for_user(user_id)
        if not user_events:
            logger.warning(f"No feedback events found for user {user_id}. Skipping EQ calculation.")
            # Pastikan ada record di EqMetrics agar tidak error saat fetch all di retrain
            supabase_service.upsert_eq_metrics({
                'user_id': user_id,
                'average_eq_score': 0.0,
                'total_sessions_analyzed': 0,
                'last_calculated_at': datetime.now(timezone.utc).isoformat()
            })
            return 0.0 # Kembalikan EQ rata-rata 0

        # 2. Identifikasi Sesi
        sessions = identify_sessions(user_events)
        if not sessions:
             logger.warning(f"Could not identify any sessions for user {user_id}.")
             supabase_service.upsert_eq_metrics({
                'user_id': user_id,
                'average_eq_score': 0.0,
                'total_sessions_analyzed': 0,
                'last_calculated_at': datetime.now(timezone.utc).isoformat()
            })
             return 0.0

        # 3. Hitung EQ per Sesi dan siapkan data history
        all_session_eqs = []
        history_records = []
        for session in sessions:
            if not session: continue
            session_eq, start_time, end_time = calculate_session_eq(session)
            all_session_eqs.append(session_eq)

            history_data = {
                "user_id": user_id,
                "session_eq_score": session_eq,
                "session_start_time": start_time.isoformat() if start_time else None,
                "session_end_time": end_time.isoformat() if end_time else None,
                "session_compilations": len(session),
                "recorded_at": datetime.now(timezone.utc).isoformat() # Waktu pencatatan
            }
            history_records.append(history_data)

        # 4. Hapus history lama dan simpan history baru (atau lakukan upsert jika memungkinkan)
        # Lebih aman menghapus dan insert ulang untuk memastikan konsistensi
        supabase_service.delete_eq_metrics_history(user_id)
        if history_records:
            supabase_service.insert_eq_metrics_history_batch(history_records)

        # 5. Hitung EQ rata-rata
        average_eq = np.mean(all_session_eqs) if all_session_eqs else 0.0

        # 6. Simpan/Update data agregat di EqMetrics
        metrics_data = {
            'user_id': user_id,
            'average_eq_score': float(average_eq),
            'total_sessions_analyzed': len(sessions),
            'last_calculated_at': datetime.now(timezone.utc).isoformat()
            # Kolom cluster & performance akan diupdate oleh retrain_model
        }
        supabase_service.upsert_eq_metrics(metrics_data)

        logger.info(f"Successfully processed EQ for user {user_id}. Average EQ: {average_eq:.4f}")
        return float(average_eq)

    except Exception as e:
        logger.error(f"Error processing EQ for user {user_id}: {e}", exc_info=True)
        return 0.0 # Kembalikan default jika error

def calculate_historical_eq_all_users():
    """Menghitung ulang EQ untuk semua pengguna berdasarkan data feedback."""
    logger.info("Starting historical EQ calculation for all users...")
    try:
        unique_users = supabase_service.fetch_unique_users_from_feedback()
        if not unique_users:
            logger.warning("No users found in feedback table.")
            return

        total_users = len(unique_users)
        logger.info(f"Found {total_users} unique users to process.")

        processed_count = 0
        for user_id in unique_users:
            process_user_eq(user_id)
            processed_count += 1
            if processed_count % 50 == 0:
                logger.info(f"Processed {processed_count}/{total_users} users...")

        logger.info("✅ Historical EQ calculation finished for all users.")

    except Exception as e:
        logger.error(f"❌ Error during historical EQ calculation: {e}", exc_info=True)