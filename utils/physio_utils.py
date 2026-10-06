from pathlib import Path
import re

import numpy as np
import pandas as pd
import neurokit2 as nk
import logging
import pyxdf
import constants as c

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def find_xdf_files(root_dir):
    log.info(f"Searching for .xdf files in: {root_dir}")
    files = list(Path(root_dir).rglob("*.xdf"))
    log.info(f"Found {len(files)} .xdf files.")
    return files


def extract_physiological_data(file_path):
    """Load XDF and extract available data streams."""
    log.info(f"Extracting data from: {file_path.name}")
    data, _ = pyxdf.load_xdf(file_path)
    return {
        s["info"]["name"][0]: {
            'timestamps': s['time_stamps'],
            'series': np.array(s['time_series'])
        }
        for s in data if s['time_stamps'].any()
    }


# --- 2. Preprocessing and Segmentation ---


def _create_segment_data(
        segment_id,
        segment_type,
        start_event,
        end_event,
        start_time,
        end_time,
        shimmer_ts,
        raw_eda,
        eda_df,
        raw_ppg,
        ppg_df
):
    """Helper: slice data for one segment."""
    s_idx = np.searchsorted(shimmer_ts, start_time, side='left')
    e_idx = np.searchsorted(shimmer_ts, end_time, side='right')
    padding_sec = 4.0
    effective_end = end_time + padding_sec if segment_type == 'video' else end_time
    e_idx = np.searchsorted(shimmer_ts, effective_end, side='right')

    if s_idx >= e_idx:
        return None
    return {
        'segment_id': segment_id, 'segment_type': segment_type,
        'start_event': start_event, 'end_event': end_event,
        'start_time': start_time, 'end_time': effective_end,
        'EDA_timestamps': shimmer_ts[s_idx:e_idx],
        'EDA_series': raw_eda[s_idx:e_idx],
        'EDA_Processed_Segment': eda_df.iloc[s_idx:e_idx],
        'PPG_series': raw_ppg[s_idx:e_idx],
        'PPG_Processed_Segment': ppg_df.iloc[s_idx:e_idx]
    }


def preprocess_and_segment(data_dict, sampling_rate):
    """
    Process EDA/PPG and cut segments between VideoDisplay start/end markers.
    An 'early_exit' marks a restarted session: earlier video segments are discarded.
    """
    shimmer_name = 'Shimmer_GSRCOM7'
    video_name = 'VideoDisplay'

    # --- Load raw signals ---
    shimmer_ts = data_dict[shimmer_name]['timestamps']
    video_ts = data_dict[video_name]['timestamps']
    raw_eda = data_dict[shimmer_name]['series'][:, 0]
    raw_ppg = data_dict[shimmer_name]['series'][:, 1]

    # Mean > 50 means the device recorded resistance (kOhm): convert to conductance (µS)
    eda_mean = np.nanmean(raw_eda)
    if eda_mean > 50:
        log.info(f"EDA in kOhm (mean={eda_mean:.1f}), converting to µS")
        raw_eda = 1000.0 / (raw_eda + 1e-6)

    # Resample to an even grid: some devices did not record at the nominal rate
    grid = np.arange(shimmer_ts[0], shimmer_ts[-1], 1 / sampling_rate)
    raw_eda = np.interp(grid, shimmer_ts, raw_eda)
    raw_ppg = np.interp(grid, shimmer_ts, raw_ppg)
    shimmer_ts = grid

    # --- Preprocess full signals once ---
    eda_df, _ = nk.eda_process(raw_eda, sampling_rate=sampling_rate)
    ppg_df, _ = nk.ppg_process(raw_ppg, sampling_rate=sampling_rate)

    # --- Sanity check for EDA tonic values ---
    tonic_min = eda_df['EDA_Tonic'].min()
    tonic_mean = eda_df['EDA_Tonic'].mean()
    log.info(f"EDA_Tonic mean={tonic_mean:.4f}, min={tonic_min:.4f}")

    # --- Extract start/end/exit events from video stream ---
    def event_type(state):
        s = state.lower()
        return 'exit' if s == 'early_exit' else 'start' if 'start' in s else 'end' if 'end' in s else None

    video_states = data_dict[video_name]['series'].flatten()
    events = [
        {'idx': i, 'type': t, 'state': s, 'time': video_ts[i]}
        for i, s in enumerate(video_states)
        if (t := event_type(s))
    ]

    # --- Find first start (for calibration segment) ---
    first_start = next((e for e in events if e['type'] == 'start'), None)
    if not first_start:
        return data_dict, []

    segments = []
    video_seg_id = 0

    # Calibration segment: before first video start
    calibration_seg = _create_segment_data(
        'call', 'calibration', 'stream_start', first_start['state'],
        shimmer_ts[0], first_start['time'],
        shimmer_ts, raw_eda, eda_df, raw_ppg, ppg_df
    )
    if calibration_seg:
        segments.append(calibration_seg)

    # --- Video segments: between start/end events ---
    open_start = None
    for event in events:
        if event['idx'] < first_start['idx']:
            continue
        if event['type'] == 'exit':  # false start: session restarted from the first stimulus
            log.info("early_exit: discarding earlier video segments")
            segments = [s for s in segments if s['segment_type'] != 'video']
            video_seg_id, open_start = 0, None
        elif event['type'] == 'start':
            open_start = event
        elif event['type'] == 'end' and open_start:

            video_seg = _create_segment_data(
                video_seg_id, 'video',
                open_start['state'], event['state'],
                open_start['time'], event['time'],
                shimmer_ts, raw_eda, eda_df, raw_ppg, ppg_df
            )
            if video_seg:
                segments.append(video_seg)
            video_seg_id += 1
            open_start = None

    return data_dict, segments


# --- 3. Sampling-rate report ---


def report_sampling_rates(files, target_stream='Shimmer_GSRCOM7'):
    """Generate a report comparing nominal and effective sampling rates"""
    log.info(f"--- Sampling Rate Report ({target_stream}) ---")
    results = []

    for f in files:
        data, _ = pyxdf.load_xdf(f)
        for s in data:
            name = s["info"]["name"][0]
            if target_stream.lower() in name.lower():
                ts = np.array(s["time_stamps"])

                nominal_val = s["info"]["nominal_srate"]
                nominal = float(nominal_val[0]) if isinstance(nominal_val, (list, np.ndarray)) else float(nominal_val)

                effective = (len(ts) - 1) / (ts[-1] - ts[0])
                log.info(f"{f.name}: nominal={nominal:.2f} Hz, effective={effective:.3f} Hz")
                results.append({'file': f.name, 'nominal': nominal, 'effective': effective})
                break

    pd.DataFrame(results).to_csv("sampling_rate_report.csv", index=False)


# --- 4. Metrics Extraction ---

def sliding_window_features(signal, window_size=2000, step_size=1000):
    """
    Compute sliding-window mean, SD, and slope for dynamic physiological response.
    - window_size: samples (1500 = 15s at 100 Hz)
    - step_size: sampling step between windows
    """
    feats = []
    for start in range(0, len(signal) - window_size, step_size):
        window = signal[start:start + window_size]
        x = np.arange(len(window))
        slope = np.polyfit(x, window, 1)[0]
        feats.append({
            "win_mean": np.mean(window),
            "win_sd": np.std(window),
            "win_slope": slope
        })
    return pd.DataFrame(feats)


def get_max_peak_recovery(eda_df):
    """
    Finds the index and amplitude of the largest SCR peak.
    Returns: max_amp, half_recov_time, max_peak_idx
    """
    if "SCR_Peaks" not in eda_df.columns:
        return np.nan, np.nan, np.nan

    # Find indices where a peak occurs
    peak_idx = np.where(eda_df["SCR_Peaks"] == 1)[0]

    if len(peak_idx) == 0:
        return np.nan, np.nan, np.nan

    # Get amplitudes for these specific peaks
    # Use .values to ensure alignment
    amps = eda_df["SCR_Amplitude"].iloc[peak_idx].values

    # Find the index of the maximum amplitude in the amps array
    max_i_local = np.argmax(amps)

    max_amp = amps[max_i_local]
    max_peak_idx = peak_idx[max_i_local]

    # Try to get recovery time, handle NaNs gracefully
    try:
        half_recov_time = eda_df["SCR_RecoveryTime"].iloc[max_peak_idx]
    except:
        half_recov_time = np.nan

    return max_amp, half_recov_time, max_peak_idx

def analyze_segment(eda_df, ppg_df, sampling_rate):
    """
    Robust extraction of EDA and PPG metrics for 30s segments.
    Handles short data, missing peaks, and artifacts gracefully.
    """

    # --- 1. Initialize Default Metrics (All NaNs) ---

    metrics = {
        # EDA Phasic (event-related)
        'SCR_Peaks_N': 0,
        'SCR_Peaks_Amplitude_Mean': np.nan,
        'SCR_Peaks_Amplitude_SD': np.nan,
        'SCR_Peaks_Amplitude_Max': np.nan,
        'SCR_Mean': np.nan,
        'SCR_SD': np.nan,
        'SCR_AUC': np.nan, # Area Under Curve
        'SCR_Recovery_Time_Half': np.nan,
        'SCR_Recovery_Slope': np.nan,

        # EDA Tonic (baseline)
        'SCL_Mean': np.nan,
        'SCL_SD': np.nan,
        'SCL_Max': np.nan,
        'SCL_Min': np.nan,
        'SCL_Slope': np.nan,
        'SCL_window_mean': np.nan,
        'SCL_window_sd': np.nan,
        'SCL_window_slope_mean': np.nan,
        'SCL_window_slope_max': np.nan,
        'SCL_window_slope_min': np.nan,

        # PPG / HRV
        'PPG_Rate_Mean': np.nan,
        'PPG_Rate_SD': np.nan,
        'HR_Min': np.nan,
        'HR_Max': np.nan,
        'HRV_MeanNN': np.nan,
        'HRV_RMSSD': np.nan,
        'HRV_SDNN': np.nan,
        'HRV_pNN20': np.nan,
        'HRV_pNN50': np.nan,
        'HRV_SD1': np.nan,
        'HRV_LF': np.nan,
        'HRV_HF': np.nan,
        'HRV_LFHF': np.nan
    }

    # --- 2. EDA PROCESSING ---

    # Smooth Tonic
    if 'EDA_Tonic' in eda_df.columns:
        eda_df['EDA_Tonic'] = eda_df['EDA_Tonic'].rolling(window=5, center=True, min_periods=1).median()

    # SCR Peaks
    scr_amp = eda_df.loc[eda_df['SCR_Amplitude'] > 0, 'SCR_Amplitude'].dropna()
    scr_amp = scr_amp.clip(lower=0, upper=50)  # Remove massive artifacts (???)

    # Update Basic EDA
    metrics.update({
        'SCR_Peaks_N': int(len(scr_amp)),
        'SCR_Peaks_Amplitude_Mean': scr_amp.mean() if not scr_amp.empty else 0,
        'SCR_Peaks_Amplitude_SD': scr_amp.std() if len(scr_amp) > 1 else 0,
        'SCR_Mean': eda_df['EDA_Phasic'].mean(),
        'SCR_SD': eda_df['EDA_Phasic'].std(),
        'SCL_Mean': eda_df['EDA_Tonic'].mean(),
        'SCL_SD': eda_df['EDA_Tonic'].std(),
        'SCL_Max': eda_df['EDA_Tonic'].max(),
        'SCL_Min': eda_df['EDA_Tonic'].min(),
    })


    # EDA AUC (Phasic)
    phasic = eda_df['EDA_Phasic'].fillna(0).values
    metrics['SCR_AUC'] = np.trapezoid(np.abs(phasic), dx=1 / sampling_rate)

    # EDA Recovery Time
    try:
        max_amp, rec_time, peak_i = get_max_peak_recovery(eda_df)

        metrics['SCR_Peaks_Amplitude_Max'] = max_amp
        metrics['SCR_Recovery_Time_Half'] = rec_time

        if not np.isnan(peak_i):
            peak_i = int(peak_i)

            win_size = 2 * sampling_rate
            start = peak_i
            end = min(len(eda_df), peak_i + win_size)

            # Extract Phasic data
            y = eda_df['EDA_Phasic'].iloc[start:end].values

            if len(y) >= 5:
                x = np.arange(len(y))
                # Fit line: y = mx + c. We want m (index 0)
                metrics['SCR_Recovery_Slope'] = np.polyfit(x, y, 1)[0]
            else:
                metrics['SCR_Recovery_Slope'] = np.nan
        else:
            metrics['SCR_Recovery_Slope'] = np.nan

    except Exception as e:
        log.warning(f"EDA Recovery Calc Failed: {e}")
        metrics['SCR_Recovery_Slope'] = np.nan

    # EDA Slopes (Phasic & Tonic)
    if len(eda_df) > 10:
        # 10 samples minimum for slope
        x = np.arange(len(eda_df))
        metrics['SCL_Slope'] = np.polyfit(x, eda_df['EDA_Tonic'].fillna(0), 1)[0]

    # EDA Sliding Window
    WINDOW_SIZE = 1500
    STEP_SIZE = 1500
    if len(eda_df) >= WINDOW_SIZE:
        win = sliding_window_features(eda_df['EDA_Tonic'].values, window_size=WINDOW_SIZE, step_size=STEP_SIZE)
        if not win.empty:
            metrics['SCL_window_mean'] = win['win_mean'].mean()
            metrics['SCL_window_sd'] = win['win_sd'].mean()
            metrics['SCL_window_slope_mean'] = win['win_slope'].mean()
            metrics['SCL_window_slope_max'] = win['win_slope'].max()
            metrics['SCL_window_slope_min'] = win['win_slope'].min()

    # --- 3. PPG / HRV PROCESSING ---

    peak_indices = np.where(ppg_df['PPG_Peaks'] == 1)[0]

    # Chose 5 to ensure we catch low-HR participants
    if len(peak_indices) >= 5:

        # Correct missed and extra beats (Kubios) before HR and HRV
        _, peak_indices = nk.signal_fixpeaks(
            peak_indices, sampling_rate=sampling_rate, iterative=True, method='Kubios'
        )

        # A. Instantaneous HR (For Min/Max/SD)
        peak_times = peak_indices / sampling_rate
        nn_ms = np.diff(peak_times) * 1000.0

        # Filter Artifacts (Physiologically impossible HRs)
        inst_hr = 60000.0 / nn_ms
        inst_hr_clean = inst_hr[(inst_hr >= 30) & (inst_hr <= 200)]

        if inst_hr_clean.size > 0:
            metrics['HR_Min'] = np.min(inst_hr_clean)
            metrics['HR_Max'] = np.max(inst_hr_clean)
            metrics['PPG_Rate_SD'] = np.std(inst_hr_clean)

        def get_val(df, col):
            return df[col].iloc[0] if col in df.columns and not df[col].isna().all() else np.nan

        # B. Time Domain (NeuroKit)
        try:
            hrv_time = nk.hrv_time(peak_indices, sampling_rate=sampling_rate, show=False)

            # Extract Neurokit Values
            mean_nn = get_val(hrv_time, 'HRV_MeanNN')
            metrics['HRV_MeanNN'] = mean_nn
            metrics['HRV_RMSSD'] = get_val(hrv_time, 'HRV_RMSSD')
            metrics['HRV_SDNN'] = get_val(hrv_time, 'HRV_SDNN')
            metrics['HRV_pNN20'] = get_val(hrv_time, 'HRV_pNN20')
            metrics['HRV_pNN50'] = get_val(hrv_time, 'HRV_pNN50')

            # Calculate Mean HR from Mean NN (More robust than instantaneous mean)
            if mean_nn > 0:
                metrics['PPG_Rate_Mean'] = 60000 / mean_nn

        except Exception as e:
            log.warning(f"Time-domain HRV failed: {e}")

        # C. Nonlinear (SD1)
        try:
            hrv_non = nk.hrv_nonlinear(peak_indices, sampling_rate=sampling_rate, show=False)
            metrics['HRV_SD1'] = get_val(hrv_non, 'HRV_SD1')
        except Exception as e:
            log.warning(f"Nonlinear HRV failed: {e}")

        # D. Frequency Domain
        try:
            # 'welch' is safer for short signals than interpolation
            hrv_freq = nk.hrv_frequency(peak_indices, sampling_rate=sampling_rate, show=False, psd_method='welch')
            metrics['HRV_LF'] = get_val(hrv_freq, 'HRV_LF')
            metrics['HRV_HF'] = get_val(hrv_freq, 'HRV_HF')
            metrics['HRV_LFHF'] = get_val(hrv_freq, 'HRV_LFHF')
        except Exception as e:
            log.warning(f"Frequency-domain HRV failed: {e}")

    return metrics


def map_physio_segments_to_videos(physio_df, experiment_setup, trial_label, video_counts):
    # TODO: Generalize for other trial types
    """
    Maps physiological segments to DJI videos based on strict positional matching.
    - Assumes input physio_df contains *only* video segments (no 'call').
    - Maps segment_id X directly to the stimulus in column X of experiment_setup.
    - Includes print statements for verification.
    """
    mapped = []

    # --- Data Prep ---
    physio_df[c.PARTICIPANT_ID] = physio_df[c.PARTICIPANT_ID].astype("Int64")
    physio_df['segment_id'] = pd.to_numeric(physio_df['segment_id'])
    experiment_setup.index = experiment_setup.index.astype("Int64")

    for pid in experiment_setup.index:
        log.info(f"--- Processing Participant {pid} ---")

        # --- 1. Get Video Segments & Index ---
        part_segments = physio_df[physio_df[c.PARTICIPANT_ID] == pid].copy()
        if part_segments.empty:
            log.warning(f"Participant {pid}: no physiological data found. Skipping.\n")
            continue

        # Set index directly on video segments
        video_segments = part_segments.set_index('segment_id', drop=False)
        n_available_segments = len(video_segments)
        available_ids = sorted(video_segments.index.tolist())
        log.info(f"Participant {pid}: Found {n_available_segments} video segments (IDs: {available_ids})")

        # --- 2. Find DJI Video Positions ---
        setup_row = experiment_setup.loc[pid]
        dji_videos_with_positions = [
            (col_idx, stimulus_name) for col_idx, stimulus_name in enumerate(setup_row)
            if isinstance(stimulus_name, str) and stimulus_name.upper().startswith("DJI")
        ]

        if not dji_videos_with_positions:
            log.warning(f"Participant {pid}: no DJI videos found in setup row. Skipping.\n")
            continue

        n_dji_videos = len(dji_videos_with_positions)
        dji_indices = [pos[0] for pos in dji_videos_with_positions]
        log.info(f"Participant {pid}: Found {n_dji_videos} DJI videos at positions (column indices): {dji_indices}")

        mapped_count = 0
        mapped_segment_ids = []

        # --- 3. Loop through DJI positions and Map ---
        for column_idx, vid_name in dji_videos_with_positions:
            target_segment_id = column_idx # Direct mapping

            try:
                # --- Direct Lookup ---
                seg_data = video_segments.loc[target_segment_id].to_dict()
                seg_data[c.PARTICIPANT_ID] = pid
                seg_data[c.VIDEO_ID_COL] = vid_name
                mapped.append(seg_data)
                mapped_count += 1
                mapped_segment_ids.append(target_segment_id)

            except KeyError:
                # This indicates missing segment data for this position
                log.warning(
                    f"Participant {pid}: [Mapping FAIL] Cannot find segment_id {target_segment_id} "
                    f"(needed for video '{vid_name}' at column {column_idx})."
                )

        # --- Verification Print ---
        log.info(f"Participant {pid}: Successfully mapped {mapped_count} / {n_dji_videos} expected DJI videos.")
        log.info(f"Participant {pid}: Mapped segment IDs -> {sorted(mapped_segment_ids)}\n")

    # --- 4. Final DataFrame Creation ---
    if not mapped:
        log.error("No physiological segments were successfully mapped across all participants.")
        return pd.DataFrame()

    mapped_df = pd.DataFrame(mapped)

    # Extract video number from the name
    mapped_df[c.VIDEO_ID_COL] = (
        mapped_df[c.VIDEO_ID_COL]
        .astype(str)
        .str.extract(r"video_(\d+)", expand=False)
        .astype("Int64")
    )

    log.info(f"Mapping complete. Created DataFrame with {len(mapped_df)} rows.")
    return mapped_df


def _block_of(code):
    return "DJI" if code.startswith("DJI") else code.split("_")[0]


def get_physio_trial_df(df_physio, experiment_setup, trial_label, metrics=None):
    """
    One physiology value per sequence, keyed by presentation order:
    physio segment_id k == the k-th column of the participant's setup row
    (same ordering the ratings use). Keeps the columns belonging to `trial_label`.

    Returns: participant_id, sequence_code (e.g. '3_2'), segment_id (presentation
    slot in the session), + metric columns.
    """
    rows = []
    have = set(df_physio[c.PARTICIPANT_ID].unique())
    for pid in experiment_setup.index:
        if pid not in have:
            continue
        setup_row = experiment_setup.loc[pid]
        psub = df_physio[df_physio[c.PARTICIPANT_ID] == pid].set_index("segment_id")
        for col_idx, fname in enumerate(setup_row):
            if pd.isna(fname):
                continue
            code = re.sub(r"\.mp4$|\.MP4$", "", fname)
            if _block_of(code) != trial_label:
                continue
            if col_idx not in psub.index:
                continue
            rows.append({c.PARTICIPANT_ID: pid, "sequence_code": code, "segment_id": col_idx,
                         **psub.loc[col_idx][metrics].to_dict()})
    return pd.DataFrame(rows)


def check_segment_alignment(physio, setup, tol=0.25):
    """Flag segments whose duration deviates > `tol` from the median of their slot's block type."""
    rows = []
    for pid, setup_row in setup.iterrows():
        dur = physio[physio[c.PARTICIPANT_ID] == pid].set_index("segment_id")["duration"]
        for slot, fname in enumerate(setup_row):
            if pd.notna(fname) and slot in dur.index:
                rows.append({c.PARTICIPANT_ID: pid, "slot": slot, "block": _block_of(fname), "duration": dur[slot]})
    d = pd.DataFrame(rows)
    d["expected"] = d.groupby("block")["duration"].transform("median")
    flagged = d[(d["duration"] - d["expected"]).abs() / d["expected"] > tol].copy()
    flagged["n_segments"] = flagged[c.PARTICIPANT_ID].map(physio.groupby(c.PARTICIPANT_ID).size())
    return flagged.round(1)


def validate_physio(physio, setup):
    """Raise if a metric is empty for all segments; log missing shares and misaligned segments."""
    metrics = physio.columns.difference([c.PARTICIPANT_ID, "segment_id", "duration"])
    missing = physio[metrics].isna().mean()
    if (missing == 1).any():
        raise RuntimeError(f"Empty metrics (check package versions): {missing[missing == 1].index.tolist()}")
    log.info(f"Missing values (%): {missing[missing > 0].mul(100).round(1).to_dict()}")

    misaligned = check_segment_alignment(physio, setup)
    if not misaligned.empty:
        log.warning(f"Misaligned segments:\n{misaligned.to_string(index=False)}")


