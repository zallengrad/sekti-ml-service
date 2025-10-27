import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np

# Impor layanan supabase (asumsi struktur folder Anda)
from . import supabase_service

logger = logging.getLogger(__name__)

# Pola regex untuk mengekstrak tipe error dan baris dari pesan error Java umum
ERROR_LINE_PATTERN = re.compile(r'^(.*?):(\d+): error: (.*)', re.MULTILINE)

# Pola error yang digunakan untuk EQ SCORE
patterns = {
    'error_cannot_find_symbol': r'cannot find symbol',
    'error_semicolon_expected': r'; expected',
    'error_runtime_exception': r'runtimeexception',
    'error_constructor': r'constructor.*cannot be applied',
    'error_identifier_expected': r'<identifier> expected',
    'error_illegal_start_of_type': r'illegal start of type',
    'bracket_expected': r'[{(] expected|illegal start of expression',
    'class_or_interface_expected': r'class or interface expected',
    'dot_class_expected': r'\.class expected',
    'not_a_statement': r'not a statement',
    'missing_return': r'missing return statement|missing return value',
    'incompatible_types': r'incompatible types',
    'private_access_violation': r'has private access',
    'method_application_error': r'cannot be applied to|actual and formal argument lists differ',
}

# Pola error yang akan DIHITUNG dan DISIMPAN ke tabel (Subset dari 'patterns')
COUNTED_ERROR_TYPES = {
    'error_cannot_find_symbol': r'cannot find symbol',
    'error_semicolon_expected': r'; expected',
    'error_runtime_exception': r'runtimeexception',
    'error_constructor': r'constructor.*cannot be applied',
    'error_identifier_expected': r'<identifier> expected',
    'error_illegal_start_of_type': r'illegal start of type'
}


# Parameter EQ (berdasarkan Tabel 4.2 - From Search)
ETYPE_SAME_PENALTY = 11
ETYPE_DIFF_PENALTY = 8
MAX_PENALTY = max(ETYPE_SAME_PENALTY, ETYPE_DIFF_PENALTY)

def parse_flexible_isoformat(ts_str: Optional[str]) -> Optional[datetime]:
    """Mencoba parsing string ISO format dengan mikrodetik yang bervariasi."""
    if not ts_str:
        return None
    try:
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1] + '+00:00'
        elif '+' not in ts_str and '-' not in ts_str[10:]:
             ts_str += '+00:00'

        if '.' in ts_str:
            base, micro_tz = ts_str.split('.', 1)
            tz_part = ''
            tz_match = re.search(r'([-+]\d{2}(:\d{2})?)$', micro_tz)
            if tz_match:
                 tz_part = tz_match.group(0)
                 micro = micro_tz[:tz_match.start()]
            else:
                 micro = micro_tz

            micro = micro.ljust(6, '0')[:6]
            ts_str = f"{base}.{micro}{tz_part}"

        return datetime.fromisoformat(ts_str)
    except ValueError as e:
        logger.error(f"Failed to parse timestamp '{ts_str}': {e}")
        return None
    except Exception as general_e:
         logger.error(f"Unexpected error parsing timestamp '{ts_str}': {general_e}")
         return None


def get_specific_error_counts(error_snapshot: str) -> Dict[str, int]:
    """Menghitung apakah event error termasuk dalam 6 tipe error yang disimpan."""
    counts = {k: 0 for k in COUNTED_ERROR_TYPES.keys()}
    if not isinstance(error_snapshot, str) or not error_snapshot.strip():
        return counts

    lowered_snapshot = error_snapshot.lower()

    for error_type, pattern in COUNTED_ERROR_TYPES.items():
        if re.search(pattern, lowered_snapshot):
            counts[error_type] = 1 # Hanya 1 karena ini adalah per event error
    return counts


def parse_error_details(error_snapshot: str) -> Tuple[Optional[str], Optional[int]]:
    """Mengekstrak tipe error PERTAMA dan nomor baris untuk perhitungan EQ."""
    if not isinstance(error_snapshot, str) or not error_snapshot.strip():
        return None, None

    lowered_snapshot = error_snapshot.lower()
    error_line = None

    # Cari nomor baris error
    match = ERROR_LINE_PATTERN.search(error_snapshot)
    error_line = int(match.group(2)) if match else None

    # Identifikasi tipe error EQ (ambil yang pertama cocok)
    error_type = None
    for type_key, pattern in patterns.items():
        if re.search(pattern, lowered_snapshot):
            error_type = type_key
            break

    return error_type, error_line


def identify_sessions(user_events: List[Dict], max_gap_minutes: int = 30) -> List[List[Dict]]:
    """Mengelompokkan event error pengguna menjadi sesi berdasarkan jeda waktu."""
    if not user_events:
        return []

    parsed_events = []
    for event in user_events:
        event_time = parse_flexible_isoformat(event.get('created_at'))
        if event_time:
            event['parsed_time'] = event_time
            parsed_events.append(event)
        else:
            logger.warning(f"Skipping event due to unparseable timestamp: {event.get('created_at')}")

    if not parsed_events:
        return []

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

    return sessions


def calculate_session_eq(session_events: List[Dict]) -> Tuple[float, Dict[str, int]]:
    """
    Menghitung skor EQ untuk satu sesi dan total 6 error counts.
    Mengembalikan (eq_score, total_error_counts).
    """
    if len(session_events) < 2:
        return 0.0, {k: 0 for k in COUNTED_ERROR_TYPES.keys()}

    pair_scores = []
    session_error_counts = {k: 0 for k in COUNTED_ERROR_TYPES.keys()}

    # Proses event-event yang sudah di-parse dan diurutkan
    parsed_details = [parse_error_details(event.get('error_snapshot')) for event in session_events]

    for i in range(1, len(session_events)):
        prev_event_details = parsed_details[i-1]
        curr_event_details = parsed_details[i]

        prev_error_type, _ = prev_event_details
        curr_error_type, _ = curr_event_details

        # Hitung skor EQ
        if prev_error_type is None or curr_error_type is None:
            pair_score = 0
        else:
            if prev_error_type == curr_error_type:
                pair_score = ETYPE_SAME_PENALTY
            else:
                pair_score = ETYPE_DIFF_PENALTY

        normalized_score = pair_score / MAX_PENALTY if MAX_PENALTY > 0 else 0
        pair_scores.append(normalized_score)

        # Akumulasi error counts (hanya dari event saat ini)
        current_event_counts = get_specific_error_counts(session_events[i].get('error_snapshot'))
        for key, count in current_event_counts.items():
            session_error_counts[key] += count


    # Tambahkan hitungan dari event pertama yang dilewati loop (i=0)
    first_event_counts = get_specific_error_counts(session_events[0].get('error_snapshot'))
    for key, count in first_event_counts.items():
        session_error_counts[key] += count


    session_eq_score = np.mean(pair_scores) if pair_scores else 0.0

    return float(session_eq_score), session_error_counts


def process_user_eq(user_id: str):
    """
    Memproses semua feedback untuk user, menghitung EQ per sesi,
    menyimpan riwayat, dan memperbarui metrik EQ agregat.
    """
    logger.info(f"Processing EQ for user {user_id}...")
    try:
        user_events = supabase_service.fetch_all_feedback_for_user(user_id)
        if not user_events:
            logger.warning(f"No feedback events found for user {user_id}. Skipping EQ calculation.")
            return 0.0

        sessions = identify_sessions(user_events)
        if not sessions:
             logger.warning(f"Could not identify any sessions for user {user_id}.")
             return 0.0

        all_session_eqs = []
        all_error_counts = {k: 0 for k in COUNTED_ERROR_TYPES.keys()}
        history_records = []

        for session in sessions:
            if not session: continue
            
            # Hitung EQ dan 6 Error Counts per sesi
            session_eq, session_error_counts = calculate_session_eq(session)

            all_session_eqs.append(session_eq)
            
            # Akumulasi 6 Error Counts ke total user
            for key, count in session_error_counts.items():
                all_error_counts[key] += count

            # Siapkan data untuk history (eq_metrics_history)
            start_time = session[0]['parsed_time']
            end_time = session[-1]['parsed_time']

            history_record = {
                "user_id": user_id,
                "session_eq_score": session_eq,
                "session_start_time": start_time.isoformat(),
                "session_end_time": end_time.isoformat(),
                "session_compilations": len(session),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                # Tambahkan 6 kolom error counts
                **session_error_counts 
            }
            history_records.append(history_record)

        # 4. Hapus history lama dan simpan history baru
        supabase_service.delete_eq_metrics_history(user_id)
        if history_records:
            supabase_service.insert_eq_metrics_history_batch(history_records)

        # 5. Hitung EQ rata-rata
        average_eq = np.mean(all_session_eqs) if all_session_eqs else 0.0

        # 6. Siapkan data agregat untuk eq_metrics
        metrics_data = {
            'user_id': user_id,
            'average_eq_score': float(average_eq),
            'total_sessions_analyzed': len(sessions),
            'last_calculated_at': datetime.now(timezone.utc).isoformat(),
            # Tambahkan 6 kolom error counts total
            **all_error_counts 
        }
        supabase_service.upsert_eq_metrics(metrics_data)

        logger.info(f"Successfully processed EQ and error counts for user {user_id}. Average EQ: {average_eq:.4f}")
        return float(average_eq)

    except Exception as e:
        logger.error(f"Error processing EQ for user {user_id}: {e}", exc_info=True)
        return 0.0

def calculate_historical_eq_all_users():
    """Menghitung ulang EQ untuk semua pengguna berdasarkan data feedback."""
    logger.info("Starting historical EQ and error counts calculation for all users...")
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

        logger.info("✅ Historical EQ and error counts calculation finished for all users.")

    except Exception as e:
        logger.error(f"❌ Error during historical EQ calculation: {e}", exc_info=True)