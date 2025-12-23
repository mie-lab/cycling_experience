from pathlib import Path
import numpy as np
import pandas as pd
import neurokit2 as nk
import logging
import matplotlib.pyplot as plt
import pyxdf
import constants as c
import seaborn as sns

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
    Preprocesses EDA and PPG from the Shimmer stream and segments data
    based on VideoDisplay events (start/end).
    """
    shimmer_name = 'Shimmer_GSRCOM7'
    video_name = 'VideoDisplay'

    # --- Load raw signals ---
    shimmer_ts = data_dict[shimmer_name]['timestamps']
    video_ts = data_dict[video_name]['timestamps']
    raw_eda = data_dict[shimmer_name]['series'][:, 0]
    raw_ppg = data_dict[shimmer_name]['series'][:, 1]

    # ------------------------------------------------------------------
    # CRITICAL FIX: Unit Conversion (kOhms -> µS)
    # ------------------------------------------------------------------
    # Diagnosis: If mean is > 50, it is Resistance.

    eda_mean = np.nanmean(raw_eda)

    if eda_mean > 50:
        log.info(f"DETECTED RESISTANCE (Mean={eda_mean:.1f}). Converting kOhms -> µS...")
        # Formula: Conductance = 1000 / Resistance
        # tiny epsilon (1e-6) to prevent DivisionByZero errors
        raw_eda = 1000.0 / (raw_eda + 1e-6)
    else:
        log.info(f"Detected Conductance (Mean={eda_mean:.1f} µS). No conversion needed.")
    # ------------------------------------------------------------------

    # --- Preprocess full signals once ---
    eda_df, _ = nk.eda_process(raw_eda, sampling_rate=sampling_rate)
    ppg_df, _ = nk.ppg_process(raw_ppg, sampling_rate=sampling_rate)

    # --- Sanity check for EDA tonic values ---
    tonic_min = eda_df['EDA_Tonic'].min()
    tonic_mean = eda_df['EDA_Tonic'].mean()
    log.info(f"EDA_Tonic mean={tonic_mean:.4f}, min={tonic_min:.4f}")

    # --- Extract start/end events from video stream ---
    video_states = data_dict[video_name]['series'].flatten()
    events = [
        {'idx': i, 'type': 'start' if 'start' in s.lower() else 'end',
         'state': s, 'time': video_ts[i]}
        for i, s in enumerate(video_states)
        if 'start' in s.lower() or 'end' in s.lower()
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
        if event['type'] == 'start':
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


# --- 3. Plotting ---


def plot_segment_signals(eda_df, ppg_df, file_path, seg_id, output_dir):
    """Plot EDA & PPG (raw vs clean) for one segment."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # EDA
    axes[0].plot(eda_df['EDA_Raw'], label='EDA Raw', alpha=0.7)
    axes[0].plot(eda_df['EDA_Clean'], label='EDA Clean', linewidth=1.5)
    axes[0].set_ylabel('EDA (µS)')
    axes[0].legend()
    axes[0].grid(True, linestyle=':')

    # PPG
    axes[1].plot(ppg_df['PPG_Raw'], label='PPG Raw', alpha=0.7)
    axes[1].plot(ppg_df['PPG_Clean'], label='PPG Clean', linewidth=1.5)
    axes[1].set_ylabel('PPG (a.u.)')
    axes[1].set_xlabel('Samples')
    axes[1].legend()
    axes[1].grid(True, linestyle=':')

    fig.suptitle(f'Segment {seg_id} - {file_path.stem}', fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    save_path = output_dir / f"{file_path.stem}_segment_{seg_id}_processed.png"
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    log.info(f"Saved processed segment plot: {save_path}")


def plot_overall_signals(data, file_path, output_dir):
    """
    Plot raw EDA & PPG signals with vertical markers for video start and end events.
    Uses fixed stream names: 'Shimmer_GSRCOM7' and 'VideoDisplay'.
    """
    shimmer_name = 'Shimmer_GSRCOM7'
    video_name = 'VideoDisplay'

    # --- Extract signals ---
    ts = data[shimmer_name]['timestamps']
    raw_eda = data[shimmer_name]['series'][:, 0]
    raw_ppg = data[shimmer_name]['series'][:, 1]

    # --- Create figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
    fig.suptitle(f'Overall Raw Signals - {file_path.stem}', fontsize=16)

    # Plot EDA
    ax1.plot(ts, raw_eda, color='blue', label='Raw EDA')
    ax1.set_ylabel("EDA")
    ax1.grid(True, linestyle=':')

    # Plot PPG
    ax2.plot(ts, raw_ppg, color='green', label='Raw PPG')
    ax2.set_ylabel("PPG")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, linestyle=':')

    # --- Add video event markers ---
    vts = data[video_name]['timestamps']
    vstates = data[video_name]['series'].flatten()

    start_label_added = False
    end_label_added = False

    for i, state in enumerate(vstates):
        s_lower = state.lower()
        if 'start' in s_lower:
            label = 'Video Start' if not start_label_added else None
            ax1.axvline(vts[i], color='g', linestyle='--', alpha=0.8, label=label)
            ax2.axvline(vts[i], color='g', linestyle='--', alpha=0.8)
            start_label_added = True
        elif 'end' in s_lower:
            label = 'Video End' if not end_label_added else None
            ax1.axvline(vts[i], color='r', linestyle=':', alpha=0.8, label=label)
            ax2.axvline(vts[i], color='r', linestyle=':', alpha=0.8)
            end_label_added = True

    # --- Add legends ---
    ax1.legend(loc='upper right', fontsize='medium')
    ax2.legend(loc='upper right', fontsize='medium')

    # --- Save plot ---
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    save_path = output_dir / f"{file_path.stem}_overall_raw_plot.png"
    fig.savefig(save_path, dpi=120)
    plt.close()

    log.info(f"Saved overall raw signal plot: {save_path}")


def plot_eda_normalization(df, output_dir):
    """
    Plot mean and normalized EDA tonic values per participant across segments.
    Each plot shows the baseline and relative deviations (Δ tonic).
    """
    participants = df['participant_id'].unique()
    for pid in participants:
        sub_df = df[df['participant_id'] == pid].sort_values('segment_id')

        plt.figure(figsize=(10, 6))
        plt.plot(sub_df['segment_id'], sub_df['EDA_Tonic_Mean'], marker='o', label='EDA Tonic Mean')
        plt.plot(sub_df['segment_id'], sub_df['EDA_Tonic_Normalized'], marker='s', label='EDA Tonic Δ (Normalized)')
        plt.axhline(0, color='gray', linestyle='--', label='Calibration Baseline')

        plt.title(f'EDA Tonic Across Segments - {pid}')
        plt.xlabel('Segment ID')
        plt.ylabel('EDA (μS)')
        plt.legend()
        plt.grid(True, linestyle=':')
        plt.tight_layout()

        save_path = output_dir / f"{pid}_eda_normalization_plot.png"
        plt.savefig(save_path, dpi=120)
        plt.close()
        log.info(f"Saved EDA normalization plot for {pid}: {save_path}")


# --- 5. Sampling Rate Check ---


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

    except Exception as e:
        log.warning(f"Failed to calculate recovery metrics: {e}")

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

        # B. Time Domain (NeuroKit)
        try:
            hrv_time = nk.hrv_time(peak_indices, sampling_rate=sampling_rate, show=False)

            def get_val(df, col):
                return df[col].iloc[0] if col in df.columns and not df[col].isna().all() else np.nan

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
        except:
            pass

        # D. Frequency Domain
        try:
            # 'welch' is safer for short signals than interpolation
            hrv_freq = nk.hrv_frequency(peak_indices, sampling_rate=sampling_rate, show=False, psd_method='welch')
            metrics['HRV_LF'] = get_val(hrv_freq, 'HRV_LF')
            metrics['HRV_HF'] = get_val(hrv_freq, 'HRV_HF')
            metrics['HRV_LFHF'] = get_val(hrv_freq, 'HRV_LFHF')
        except:
            pass

    return metrics


def map_physiological_segments_to_videos(physio_df, experiment_setup, trial_label, video_counts):
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


def save_segment_data_csv(seg, file_path, output_dir):
    """
    Saves processed EDA and PPG DataFrames for one segment to CSV files.
    Also saves raw signals and timestamps for completeness.
    """
    seg_id = seg['segment_id']
    base_name = f"{file_path.stem}_segment_{seg_id}"

    # 1. Save Processed EDA DataFrame
    eda_save_path = output_dir / f"{base_name}_eda_processed.csv"
    # EDA_Processed_Segment is a DataFrame containing 'EDA_Raw', 'EDA_Clean', 'EDA_Tonic', etc.
    seg['EDA_Processed_Segment'].to_csv(eda_save_path, index=False)
    log.info(f"Saved processed EDA data: {eda_save_path}")

    # 2. Save Processed PPG DataFrame
    ppg_save_path = output_dir / f"{base_name}_ppg_processed.csv"
    # PPG_Processed_Segment is a DataFrame containing 'PPG_Raw', 'PPG_Clean', 'PPG_Peaks', etc.
    seg['PPG_Processed_Segment'].to_csv(ppg_save_path, index=False)
    log.info(f"Saved processed PPG data: {ppg_save_path}")

    # 3. Save Raw Signals and Timestamps (Combined into one CSV for simplicity)
    raw_df = pd.DataFrame({
        'Time_Stamps': seg['EDA_timestamps'],
        'EDA_Raw_Series': seg['EDA_series'],
        'PPG_Raw_Series': seg['PPG_series'],
    })
    raw_save_path = output_dir / f"{base_name}_raw_signals.csv"
    raw_df.to_csv(raw_save_path, index=False)
    log.info(f"Saved raw segmented signals: {raw_save_path}")


def analyze_physio_affective_relationships(
        df1_with_features: pd.DataFrame,
        participant_col: str = "participant_id",
        valence_col: str = "valence",
        arousal_col: str = "arousal",
        min_periods: int = 30,
        print_output: bool = True
) -> pd.DataFrame:
    """
    Analyze relationships between physiological features and affective ratings.
    Includes ALL metrics calculated in the updated generation pipeline.
    """

    log.info("Analyzing relationships between physiological features and affective ratings...")

    # --- Define ALL physiological feature columns ---
    physio_cols = [
        # --- 1. EDA Phasic (Event-Related) ---
        'SCR_Peaks_N',  # Count of arousal events
        'SCR_Peaks_Amplitude_Mean',  # Intensity of events
        'SCR_Peaks_Amplitude_SD',  # Variability of event intensity
        'SCR_Peaks_Amplitude_Max',  # Max event intensity
        'SCR_Mean',  # General phasic activity
        'SCR_SD',  # Variability of phasic signal
        'SCR_AUC',  # Area under the curve
        'SCR_Recovery_Time_Half', # Time to recover 50% of max peak amp
        'SCR_Recovery_Slope',  # Speed of recovery after max peak

        # --- 2. EDA Tonic (Background Levels) ---
        'SCL_Mean', # Average Tonic Level
        'SCL_Delta',  # Reactivity (Video - Baseline)
        'SCL_SD',  # Variability of tonic level
        'SCL_Max', # Maximum Tonic Level
        'SCL_Min', # Minimum Tonic Level
        'SCL_Slope',  # Overall trend during video (renamed from SCL_Slope)

        # --- 3. EDA Dynamics (Specific Moments) ---
        'SCL_window_mean',
        'SCL_window_sd',
        'SCL_window_slope_mean',
        'SCL_window_slope_max',
        'SCL_window_slope_min',

        # --- 4. PPG Heart Rate ---
        'PPG_Rate_Mean',  # Average Heart Rate
        'HR_Delta',  # HR Reactivity
        'PPG_Rate_SD',  # HR Variability (standard deviation)
        'HR_Min',  # Minimum Heart Rate
        'HR_Max',  # Maximum Heart Rate

        # --- 5. HRV Time Domain ---
        'HRV_RMSSD',  # Vagal Tone (Robust)
        'HRV_RMSSD_Delta',  # Vagal Tone Reactivity
        'HRV_SDNN',  # Overall Variability
        'HRV_SDNN_Delta',  # Overall Variability Reactivity
        'HRV_MeanNN',  # Mean inter-beat interval
        'HRV_pNN20',  # Parasympathetic metric
        'HRV_pNN50',  # Parasympathetic metric (strict)
        'HRV_SD1',  # Non-linear (correlated with RMSSD)

        # --- 6. HRV Frequency Domain ---
        'HRV_LF',  # Low Frequency (Sympathetic/Mixed)
        'HRV_HF',  # High Frequency (Vagal)
        'HRV_HF_Delta',  # HF Reactivity
        'HRV_LFHF'  # Sympathovagal Balance
    ]

    # --- SAFETY FILTER ---
    # Only keep columns that actually exist in your dataframe.
    # This prevents crashes if you decided not to save one specific metric in main().
    existing_cols = [col for col in physio_cols if col in df1_with_features.columns]

    missing_cols = set(physio_cols) - set(existing_cols)
    if missing_cols:
        log.warning(f"The following metrics were requested but not found in CSV: {missing_cols}")

    if not existing_cols:
        raise ValueError("No physiological columns found in the provided DataFrame.")

    # --- Z-score normalization within participant ---
    log.info("Applying within-participant z-score normalization...")

    df1_with_features_z = df1_with_features.copy()
    for col in existing_cols:
        # Transform per participant
        df1_with_features_z[col] = df1_with_features_z.groupby(participant_col)[col].transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0
        )

    # --- Correlation analysis ---
    corr_df = df1_with_features_z[[valence_col, arousal_col] + existing_cols].copy()
    corr_df = corr_df.dropna(subset=[valence_col, arousal_col], how='all')

    # Calculate correlations
    corr_matrix = corr_df.corr(method='spearman', min_periods=min_periods)

    # Sort by Arousal correlation
    physio_corrs = corr_matrix.loc[existing_cols, [valence_col, arousal_col]].sort_values(by=arousal_col,
                                                                                          ascending=False)

    # --- Output ---
    if print_output:
        print("\n" + "=" * 80)
        print("=== CORRELATIONS (Spearman, z-scored) ===")
        print(physio_corrs.round(3))

    return physio_corrs

def plot_physio_affect_correlations(
    corr_results: pd.DataFrame,
    output_dir,
    strong_threshold: float = 0.1,
    palette_main: str = "coolwarm",
    palette_strong: str = "RdBu",
    save_figs: bool = True,
    show_figs: bool = False
):
    """
    Plot correlations between physiological metrics and affective ratings (valence & arousal).
    """

    log.info("Plotting physiological–affective correlations...")

    # --- Prepare Data for Plotting ---
    plot_df = corr_results.reset_index().melt(
        id_vars='index',
        var_name='Affective_Dimension',
        value_name='Correlation'
    )
    plot_df.rename(columns={'index': 'Metric'}, inplace=True)

    # =============================================================================
    # A. Full Bar Plot
    # =============================================================================
    plt.figure(figsize=(10, 7))
    sns.barplot(
        data=plot_df,
        x='Correlation',
        y='Metric',
        hue='Affective_Dimension',
        orient='h',
        palette=palette_main,
        alpha=0.9
    )
    plt.axvline(0, color='gray', linewidth=1)
    plt.title("Correlations Between Physiological Metrics and Affective Ratings (Z-scored)",
              fontsize=14, weight='bold')
    plt.xlabel("Spearman Correlation (ρ)")
    plt.ylabel("Physiological Metric")
    plt.legend(title="", loc='lower right', frameon=False)
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    plt.tight_layout()

    plot_path = output_dir / "physio_affect_correlations_plot.png"
    if save_figs:
        plt.savefig(plot_path, dpi=300)
        log.info(f"Saved bar plot: {plot_path}")
    if show_figs:
        plt.show()
    plt.close()

    # =============================================================================
    # B. Heatmap
    # =============================================================================
    plt.figure(figsize=(6, len(corr_results) * 0.4))
    sns.heatmap(
        corr_results[['valence', 'arousal']],
        annot=True, fmt=".2f",
        cmap=palette_main, center=0,
        cbar_kws={'label': "Spearman ρ"},
        linewidths=0.5, linecolor='gray'
    )
    plt.title("Correlation Heatmap: Physiology vs. Valence & Arousal (Z-scored)",
              fontsize=13, weight='bold')
    plt.xlabel("Affective Dimension")
    plt.ylabel("Physiological Metric")
    plt.tight_layout()

    heatmap_path = output_dir / "physio_affect_correlation_heatmap.png"
    if save_figs:
        plt.savefig(heatmap_path, dpi=300)
        log.info(f"Saved heatmap plot: {heatmap_path}")
    if show_figs:
        plt.show()
    plt.close()

    # =============================================================================
    # C. Strong Correlations Only
    # =============================================================================
    strong_corrs = plot_df[plot_df['Correlation'].abs() >= strong_threshold]
    if not strong_corrs.empty:
        plt.figure(figsize=(8, 5))
        sns.barplot(
            data=strong_corrs.sort_values('Correlation', ascending=False),
            x='Correlation', y='Metric',
            hue='Affective_Dimension',
            palette=palette_strong, orient='h'
        )
        plt.axvline(0, color='gray', linewidth=1)
        plt.title(f"Strongest Physiology–Affect Correlations (|ρ| ≥ {strong_threshold}, Z-scored)",
                  fontsize=13, weight='bold')
        plt.xlabel("Spearman Correlation (ρ)")
        plt.ylabel("Physiological Metric")
        plt.legend(title="", loc='lower right', frameon=False)
        plt.grid(axis='x', linestyle=':', alpha=0.6)
        plt.tight_layout()

        strong_plot_path = output_dir / "strong_physio_affect_correlations.png"
        if save_figs:
            plt.savefig(strong_plot_path, dpi=300)
            log.info(f"Saved strong correlation plot: {strong_plot_path}")
        if show_figs:
            plt.show()
        plt.close()
    else:
        log.info(f"No correlations exceeded |ρ| ≥ {strong_threshold}. Skipping strong correlation plot.")

    log.info("Physiology–affect correlation plotting complete.")


def summarize_physio_data_coverage(
    final_df: pd.DataFrame,
    output_dir,
    physio_id_col: str = "segment_id",
    output_filename: str = "physio_data_coverage_summary.csv",
    save_csv: bool = True,
    print_summary: bool = True
) -> pd.DataFrame:
    """
    Summarize coverage of physiological data in a dataset.

    Calculates how many total observations have corresponding physiological
    data (based on a non-null ID column such as 'segment_id'), logs the results,
    and optionally saves a CSV summary.
    """

    if physio_id_col not in final_df.columns:
        raise ValueError(f"Column '{physio_id_col}' not found in the provided DataFrame.")

    # --- Compute coverage metrics ---
    total_obs = len(final_df)
    missing_physio = final_df[physio_id_col].isna().sum()
    available_physio = total_obs - missing_physio
    missing_pct = (missing_physio / total_obs) * 100 if total_obs > 0 else 0
    available_pct = (available_physio / total_obs) * 100 if total_obs > 0 else 0

    # --- Logging ---
    log.info(f"Total observations: {total_obs}")
    log.info(f"With physiological data: {available_physio} ({available_pct:.1f}%)")
    log.info(f"Missing physiological data: {missing_physio} ({missing_pct:.1f}%)")

    # --- Create summary DataFrame ---
    physio_summary = pd.DataFrame({
        'Total_Observations': [total_obs],
        'With_Physiological_Data': [available_physio],
        'Missing_Physiological_Data': [missing_physio],
        'Coverage_Percentage': [available_pct],
        'Loss_Percentage': [missing_pct]
    })

    # --- Save summary if requested ---
    if save_csv:
        output_path = output_dir / output_filename
        physio_summary.to_csv(output_path, index=False)
        log.info(f"Saved physiological data coverage summary to: {output_path}")

    # --- Print summary if requested ---
    if print_summary:
        print("\n=== Physiological Data Coverage Summary ===")
        print(physio_summary.round(2).to_string(index=False))

    return physio_summary