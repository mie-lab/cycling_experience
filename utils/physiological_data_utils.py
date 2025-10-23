from pathlib import Path
import numpy as np
import pandas as pd
import neurokit2 as nk
import logging
import matplotlib.pyplot as plt
import pyxdf

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
    if s_idx >= e_idx:
        return None
    return {
        'segment_id': segment_id, 'segment_type': segment_type,
        'start_event': start_event, 'end_event': end_event,
        'start_time': start_time, 'end_time': end_time,
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
    raw_eda = data_dict[shimmer_name]['series'][:, 0] / 100
    raw_ppg = data_dict[shimmer_name]['series'][:, 1]

    # --- Preprocess full signals once ---
    eda_df, _ = nk.eda_process(raw_eda, sampling_rate=sampling_rate)
    ppg_df, _ = nk.ppg_process(raw_ppg, sampling_rate=sampling_rate)

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

    segments, seg_id = [], 0

    # Calibration segment: before first video start
    calibration_seg = _create_segment_data(
        seg_id, 'calibration', 'stream_start', first_start['state'],
        shimmer_ts[0], first_start['time'],
        shimmer_ts, raw_eda, eda_df, raw_ppg, ppg_df
    )
    if calibration_seg:
        segments.append(calibration_seg)
        seg_id += 1

    # --- Video segments: between start/end events ---
    open_start = None
    for event in events:
        if event['idx'] < first_start['idx']:
            continue
        if event['type'] == 'start':
            open_start = event
        elif event['type'] == 'end' and open_start:
            if 'video' in open_start['state'].lower():
                video_seg = _create_segment_data(
                    seg_id, 'video',
                    open_start['state'], event['state'],
                    open_start['time'], event['time'],
                    shimmer_ts, raw_eda, eda_df, raw_ppg, ppg_df
                )
                if video_seg:
                    segments.append(video_seg)
                seg_id += 1
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


def analyze_segment(eda_df, ppg_df, sampling_rate):
    """Extract physiologically valid short-term metrics (30–90s)."""
    # --- EDA metrics ---
    metrics = {
        'EDA_SCR_Count': eda_df['SCR_Onsets'].sum(),
        'EDA_Phasic_Mean': eda_df['EDA_Phasic'].mean(),
        'EDA_Tonic_Mean': eda_df['EDA_Tonic'].mean()
    }

    scr_amp = eda_df.loc[eda_df['SCR_Amplitude'] > 0, 'SCR_Amplitude']
    metrics['EDA_SCR_Count'] = len(scr_amp)
    metrics['EDA_SCR_Amplitude_Mean'] = scr_amp.mean() if len(scr_amp) > 0 else np.nan

    # --- HRV metrics (short-term) ---
    peak_indices = np.where(ppg_df['PPG_Peaks'] == 1)[0]

    if len(peak_indices) >= 20:
        hrv = nk.hrv(peak_indices, sampling_rate=sampling_rate, show=False)

        def get_scalar(hrv_df, key):
            val = hrv_df.get(key, pd.Series([np.nan]))
            if isinstance(val, pd.Series):
                return val.iloc[0]
            return val

        mean_nn_val = get_scalar(hrv, 'HRV_MeanNN')
        hr_mean = 60000 / mean_nn_val if pd.notna(mean_nn_val) and mean_nn_val > 0 else np.nan

        if 40 <= hr_mean <= 180:
            metrics.update({
                'Heart_Rate_Mean': hr_mean,
                'HRV_RMSSD': get_scalar(hrv, 'HRV_RMSSD'),
                'HRV_SD1': get_scalar(hrv, 'HRV_SD1'),
            })
        else:
            metrics.update({'Heart_Rate_Mean': np.nan, 'HRV_RMSSD': np.nan, 'HRV_SD1': np.nan})
    else:
        metrics.update({'Heart_Rate_Mean': np.nan, 'HRV_RMSSD': np.nan, 'HRV_SD1': np.nan})

    return metrics


def map_physiological_segments_to_videos(physio_df, experiment_setup, trial_label, video_counts):
    """
    Map physiological segments to the correct video filenames (e.g., DJI videos)
    based on the experiment setup, following the same order logic as get_trial_dict().
    Handles missing/broken physiological segments gracefully.
    """
    mapped = []
    n_total = video_counts[trial_label]
    physio_df[c.PARTICIPANT_ID] = physio_df[c.PARTICIPANT_ID].astype('Int64')
    experiment_setup.index = experiment_setup.index.astype('Int64')

    for pid in physio_df[c.PARTICIPANT_ID].unique():
        if pid not in experiment_setup.index:
            continue

        setup_row = experiment_setup.loc[pid]
        trial_order = list(dict.fromkeys(fn.split('_')[0] for fn in setup_row if pd.notna(fn)))
        if trial_label not in trial_order:
            continue

        # Find column range for this trial
        idx = trial_order.index(trial_label)
        col_start = int(sum(video_counts[t] for t in trial_order[:idx]) / 2)
        col_end = col_start + n_total

        # Extract only valid DJI videos
        trial_files = setup_row.iloc[col_start:col_end].dropna().tolist()
        dji_files = [f for f in trial_files if isinstance(f, str) and f.upper().startswith("DJI")]

        # Physiological segments for this participant
        pid_physio = physio_df[physio_df[c.PARTICIPANT_ID] == pid].sort_values('segment_id')
        n_physio = len(pid_physio)
        n_videos = len(dji_files)

        if n_physio == 0:
            logging.warning(f"Participant {pid}: no physiological data found.")
            continue

        if n_physio != n_videos:
            logging.warning(
                f"Participant {pid}: Missing — {n_physio} physio segments from {n_videos} videos. "
                f"Mapping only first {min(n_physio, n_videos)} entries."
            )

        for (i, row), vid in zip(pid_physio.iterrows(), dji_files):
            mapped.append({**row.to_dict(), c.VIDEO_ID_COL: vid})

    mapped_df = pd.DataFrame(mapped)

    if not mapped_df.empty:
        mapped_df[c.VIDEO_ID_COL] = (
            mapped_df[c.VIDEO_ID_COL]
            .astype(str)
            .str.extract(r'video_(\d+)', expand=False)
            .astype(int)
        )

    return mapped_df