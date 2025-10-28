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

# Parameter EQ (berdasarkan Tabel 4.2 - From Search di PDF)
ETYPE_SAME_PENALTY = 11
ETYPE_DIFF_PENALTY = 8
MAX_PENALTY = max(ETYPE_SAME_PENALTY, ETYPE_DIFF_PENALTY) # Skor maksimum per pasangan (karena eline_penalty=0)

def parse_flexible_isoformat(ts_str: Optional[str]) -> Optional[datetime]:
    """Mencoba parsing string ISO format dengan mikrodetik yang bervariasi."""
    if not ts_str:
        return None
    try:
        # Penanganan zona waktu
        if ts_str.endswith('Z'):
            ts_str = ts_str[:-1] + '+00:00'
        # Tambahkan offset default jika tidak ada
        elif '+' not in ts_str and (len(ts_str) <= 19 or '-' not in ts_str[10:]):
             ts_str += '+00:00'

        # Penanganan mikrodetik
        if '.' in ts_str:
            base_part = ts_str.split('.', 1)[0]
            fractional_part_with_tz = ts_str[len(base_part)+1:]

            tz_part = ''
            # Cari zona waktu di akhir
            tz_match = re.search(r'([-+]\d{2}(:?\d{2})?)$', fractional_part_with_tz)
            if tz_match:
                 tz_part = tz_match.group(0)
                 micro = fractional_part_with_tz[:-len(tz_part)]
            else:
                 micro = fractional_part_with_tz # Tidak ada zona waktu eksplisit

            # Pastikan 6 digit mikrodetik
            micro = micro.ljust(6, '0')[:6]
            ts_str = f"{base_part}.{micro}{tz_part}"

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
    """Mengekstrak tipe error PERTAMA dan nomor baris."""
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

    # Catatan: nomor baris (error_line) diekstrak tapi tidak digunakan
    # dalam perhitungan skor EQ sesuai parameter optimal PDF (eline_penalty=0)
    return error_type, error_line


def identify_sessions(user_events: List[Dict], max_gap_minutes: int = 30) -> List[List[Dict]]:
    """Mengelompokkan event error pengguna menjadi sesi berdasarkan jeda waktu."""
    if not user_events:
        return []

    parsed_events = []
    for event in user_events:
        event_time = parse_flexible_isoformat(event.get('created_at'))
        if event_time:
             # Hanya tambahkan event jika timestamp berhasil di-parse
             event['parsed_time'] = event_time
             parsed_events.append(event)
        else:
            logger.warning(f"Skipping event due to unparseable timestamp: {event.get('created_at')}")

    if not parsed_events:
         logger.warning("No valid events found after parsing timestamps.")
         return []

    # Urutkan event berdasarkan waktu
    parsed_events.sort(key=lambda x: x['parsed_time'])

    sessions = []
    current_session = [parsed_events[0]] # Mulai sesi pertama dengan event pertama
    last_event_time = parsed_events[0]['parsed_time']

    for event in parsed_events[1:]: # Mulai dari event kedua
        event_time = event['parsed_time']

        # Jika jeda waktu terlalu besar, mulai sesi baru
        if (event_time - last_event_time) > timedelta(minutes=max_gap_minutes):
            if current_session: # Simpan sesi sebelumnya jika tidak kosong
                sessions.append(current_session)
            current_session = [event] # Sesi baru dimulai dengan event ini
        else:
            # Jika jeda waktu masih dalam batas, tambahkan ke sesi saat ini
            current_session.append(event)

        last_event_time = event_time

    # Jangan lupa simpan sesi terakhir
    if current_session:
        sessions.append(current_session)

    logger.info(f"Identified {len(sessions)} sessions from {len(parsed_events)} valid events.")
    return sessions


def calculate_session_eq(session_events: List[Dict]) -> Tuple[float, Dict[str, int]]:
    """
    Menghitung skor EQ untuk satu sesi dan total 6 error counts.
    Menggunakan parameter optimal dari Jadud (2006) Tabel 4.2.
    Mengembalikan (eq_score, total_error_counts).
    """
    if len(session_events) < 2:
        # Jika hanya 0 atau 1 event, EQ tidak bisa dihitung (tidak ada pasangan)
        # Hitung error counts untuk satu event jika ada
        initial_counts = {k: 0 for k in COUNTED_ERROR_TYPES.keys()}
        if len(session_events) == 1:
            snapshot = session_events[0].get('error_snapshot')
            if snapshot:
                initial_counts = get_specific_error_counts(snapshot)
        return 0.0, initial_counts

    pair_scores = []
    session_error_counts = {k: 0 for k in COUNTED_ERROR_TYPES.keys()}

    # Proses event-event, ambil detail error dan snapshot mentah
    parsed_details = []
    for event in session_events:
        snapshot = event.get('error_snapshot')
        details = parse_error_details(snapshot)
        parsed_details.append((details, snapshot))

    # --- Hitung skor EQ berdasarkan pasangan ---
    for i in range(1, len(parsed_details)):
        prev_event_details, prev_snapshot = parsed_details[i-1] # Ambil detail dan snapshot sebelumnya
        curr_event_details, curr_snapshot = parsed_details[i]   # Ambil detail dan snapshot saat ini

        # --- REVISI: Filter pasangan tanpa perubahan kode ---
        # Periksa apakah ada perubahan kode (snapshot error berbeda)
        # Jika snapshot sama persis, lewati pasangan ini (sesuai footnote PDF Gambar 4.4)
        # Pastikan snapshot tidak None sebelum membandingkan
        if prev_snapshot is not None and prev_snapshot == curr_snapshot:
            # logger.debug(f"Skipping pair {i-1}-{i}: No code change detected (snapshots identical).") # Opsional logging
            continue # Lanjut ke pasangan event berikutnya
        # --- AKHIR REVISI ---

        prev_error_type, _ = prev_event_details if prev_event_details else (None, None)
        curr_error_type, _ = curr_event_details if curr_event_details else (None, None)

        # Hitung skor EQ
        pair_score = 0 # Default skor 0
        # Hanya hitung skor jika KEDUA event adalah error sintaks (punya tipe error)
        if prev_error_type is not None and curr_error_type is not None:
            if prev_error_type == curr_error_type:
                pair_score = ETYPE_SAME_PENALTY # Penalti tinggi jika tipe sama
            else:
                pair_score = ETYPE_DIFF_PENALTY # Penalti lebih rendah jika tipe beda
            # Penalti lokasi (eline_penalty) = 0 sesuai parameter optimal PDF Tabel 4.2

        # Normalisasi skor (Pembagi 11 sudah benar sesuai parameter optimal PDF)
        normalized_score = pair_score / MAX_PENALTY if MAX_PENALTY > 0 else 0
        pair_scores.append(normalized_score)

    # --- Akumulasi error counts untuk SEMUA event dalam sesi ---
    for _, snapshot in parsed_details: # Iterasi melalui semua detail yang sudah diproses
        if snapshot: # Hanya proses jika snapshot ada
            current_event_counts = get_specific_error_counts(snapshot)
            for key, count in current_event_counts.items():
                session_error_counts[key] += count

    # Hitung rata-rata skor ternormalisasi untuk sesi ini
    # Jika tidak ada pasangan yang valid (misal semua snapshot sama), pair_scores kosong
    session_eq_score = np.mean(pair_scores) if pair_scores else 0.0

    return float(session_eq_score), session_error_counts


def process_user_eq(user_id: str):
    """
    Memproses semua feedback untuk user, menghitung EQ per sesi,
    menyimpan riwayat, dan memperbarui metrik EQ agregat.
    Mengembalikan skor rata-rata EQ terakhir atau None jika gagal.
    """
    logger.info(f"Processing EQ for user {user_id}...")
    try:
        user_events = supabase_service.fetch_all_feedback_for_user(user_id)
        if not user_events:
            logger.warning(f"No feedback events found for user {user_id}. Skipping EQ calculation.")
            # Kembalikan None agar /classify tahu tidak ada data EQ
            return None

        sessions = identify_sessions(user_events)
        if not sessions:
             logger.warning(f"Could not identify any valid sessions for user {user_id}.")
             # Kembalikan None jika tidak ada sesi valid
             return None

        all_session_eqs = []
        cumulative_error_counts = {k: 0 for k in COUNTED_ERROR_TYPES.keys()} # Ubah nama variabel
        history_records = []

        logger.info(f"Calculating EQ for {len(sessions)} sessions for user {user_id}...")
        for session_idx, session in enumerate(sessions):
            if not session or not session[0].get('parsed_time'): # Pastikan sesi tidak kosong dan punya waktu
                 logger.warning(f"Skipping empty or invalid session {session_idx+1} for user {user_id}")
                 continue

            # Hitung EQ dan Error Counts per sesi
            try:
                session_eq, session_error_counts = calculate_session_eq(session)
            except Exception as calc_err:
                 logger.error(f"Error calculating EQ for session {session_idx+1} (user {user_id}): {calc_err}", exc_info=True)
                 continue # Lewati sesi ini jika perhitungan gagal

            all_session_eqs.append(session_eq)

            # Akumulasi Error Counts ke total user
            for key, count in session_error_counts.items():
                cumulative_error_counts[key] += count

            # Siapkan data untuk history (eq_metrics_history)
            try:
                start_time = session[0]['parsed_time']
                end_time = session[-1]['parsed_time']
                session_start_iso = start_time.isoformat()
                session_end_iso = end_time.isoformat()
            except (IndexError, KeyError, AttributeError) as time_err:
                 logger.warning(f"Could not determine start/end time for session {session_idx+1} (user {user_id}): {time_err}. Skipping history record.")
                 continue # Jangan buat record history jika waktu tidak valid

            history_record = {
                "user_id": user_id,
                "session_eq_score": session_eq,
                "session_start_time": session_start_iso,
                "session_end_time": session_end_iso,
                "session_compilations": len(session), # Ganti nama kolom jika perlu
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                # Tambahkan 6 kolom error counts sesi
                **session_error_counts
            }
            history_records.append(history_record)

        # Jika tidak ada sesi valid yang bisa diproses setelah loop
        if not all_session_eqs:
             logger.warning(f"No valid session EQ scores could be calculated for user {user_id}.")
             return None

        # --- Penyimpanan ke Database ---
        # Hapus history lama dan simpan history baru (jika ada)
        try:
            supabase_service.delete_eq_metrics_history(user_id)
            if history_records:
                supabase_service.insert_eq_metrics_history_batch(history_records)
            else:
                 logger.info(f"No new history records to insert for user {user_id}.")
        except Exception as db_hist_err:
             logger.error(f"Database error handling history for user {user_id}: {db_hist_err}", exc_info=True)
             # Pertimbangkan apakah mau lanjut atau stop jika DB error

        # Hitung EQ rata-rata keseluruhan
        average_eq = np.mean(all_session_eqs) # all_session_eqs dijamin tidak kosong di sini

        # Siapkan data agregat untuk eq_metrics
        metrics_data = {
            'user_id': user_id,
            'average_eq_score': float(average_eq),
            'total_sessions_analyzed': len(all_session_eqs), # Jumlah sesi yang berhasil dihitung EQnya
            'last_calculated_at': datetime.now(timezone.utc).isoformat(),
            # Tambahkan 6 kolom error counts total KESELURUHAN
            **cumulative_error_counts
        }

        # Simpan/update metrik agregat
        try:
             supabase_service.upsert_eq_metrics(metrics_data)
        except Exception as db_metrics_err:
             logger.error(f"Database error upserting metrics for user {user_id}: {db_metrics_err}", exc_info=True)
             # Mungkin kembalikan None jika penyimpanan gagal?
             return None

        logger.info(f"Successfully processed EQ and error counts for user {user_id}. Average EQ: {average_eq:.4f}")
        return float(average_eq) # Kembalikan skor rata-rata

    except Exception as e:
        logger.error(f"General error processing EQ for user {user_id}: {e}", exc_info=True)
        return None # Kembalikan None jika ada error tak terduga


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
        success_count = 0
        fail_count = 0
        for user_id in unique_users:
            logger.info(f"Processing historical data for user {user_id} ({processed_count+1}/{total_users})...")
            result = process_user_eq(user_id) # Fungsi ini akan mengurus penyimpanan
            if result is not None:
                 success_count += 1
            else:
                 fail_count +=1
                 logger.warning(f"Failed to process historical EQ for user {user_id}.")

            processed_count += 1
            # Log progress sesekali
            if processed_count % 50 == 0 or processed_count == total_users:
                logger.info(f"Progress: Processed {processed_count}/{total_users} users (Success: {success_count}, Fail: {fail_count}).")

        logger.info(f"✅ Historical EQ and error counts calculation finished. Processed: {processed_count}, Success: {success_count}, Fail: {fail_count}.")

    except Exception as e:
        logger.error(f"❌ Unhandled error during historical EQ calculation: {e}", exc_info=True)