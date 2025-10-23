import re
import configparser
import logging
import matplotlib
import warnings

from utils.physiological_data_utils import *

matplotlib.use('Agg')

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- 6. Main ---


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("config.ini")

    root = Path(config['paths']['physiological_data_dir'])
    out = Path(config['paths']['output_dir'])
    physiological_data_out = Path(config['filenames']['physiological_results_file'])
    out.mkdir(exist_ok=True, parents=True)

    files = find_xdf_files(root)
    report_sampling_rates(files)

    SAMPLING_RATE = 100
    all_metrics = []

    for f in files:
        log.info(f"--- Processing {f.name} ---")
        data = extract_physiological_data(f)

        # Plot complete recording once
        # plot_overall_signals(data, f, out)

        # Preprocess and segment data
        data, segments = preprocess_and_segment(data, SAMPLING_RATE)

        # --- Find calibration baseline ---
        base_tonic, base_scr_amp = np.nan, np.nan
        calibration_seg = next((s for s in segments if s['segment_type'] == 'calibration'), None)

        if calibration_seg:
            cal_metrics = analyze_segment(
                calibration_seg['EDA_Processed_Segment'],
                calibration_seg['PPG_Processed_Segment'],
                SAMPLING_RATE
            )

            base_tonic = cal_metrics.get('EDA_Tonic_Mean', np.nan)
            base_scr_amp = cal_metrics.get('EDA_SCR_Amplitude_Mean', np.nan)
            log.info(f" -> Calibration baseline: tonic={base_tonic:.3f}, SCR_amp={base_scr_amp:.3f}")
        else:
            log.warning("No calibration segment found; skipping baseline normalization.")

        # --- Analyze video segments ---
        for seg in [s for s in segments if s['segment_type'] == 'video']:
            duration_s = seg['end_time'] - seg['start_time']
            log.info(f" -> Segment {seg['segment_id']} ({duration_s:.1f}s)")

            seg_metrics = analyze_segment(
                seg['EDA_Processed_Segment'],
                seg['PPG_Processed_Segment'],
                SAMPLING_RATE
            )

            # --- Combine metrics ---
            m = re.search(r'P(\d+)', f.stem)
            participant_id = int(m.group(1)) if m else np.nan

            seg_metrics.update({
                'participant_id': participant_id,
                'segment_id': seg['segment_id'],
                'EDA_Tonic_Normalized': (
                    seg_metrics['EDA_Tonic_Mean'] - base_tonic
                    if pd.notna(base_tonic) else np.nan
                ),
                'EDA_SCR_Amplitude_Normalized': (
                    seg_metrics['EDA_SCR_Amplitude_Mean'] - base_scr_amp
                    if pd.notna(base_scr_amp) else np.nan
                ),
            })
            # plot_segment_signals(seg['EDA_Processed_Segment'], seg['PPG_Processed_Segment'], f, seg['segment_id'], out)
            all_metrics.append(seg_metrics)

    # --- Save results ---
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        cols = [
            'participant_id', 'segment_id', 'duration_s',
            'Heart_Rate_Mean', 'HRV_RMSSD', 'HRV_SD1',
            'EDA_SCR_Count', 'EDA_SCR_Amplitude_Mean', 'EDA_Tonic_Mean',
            'EDA_Tonic_Normalized', 'EDA_SCR_Amplitude_Normalized'
        ]
        df = df[[c for c in cols if c in df.columns]]
        save_path = physiological_data_out
        df.to_csv(save_path, index=False)
        log.info(f"Metrics saved to: {save_path}")
        # plot_eda_normalization(df, out)
    else:
        log.warning("No metrics extracted. Nothing to save.")


if __name__ == "__main__":
    main()
