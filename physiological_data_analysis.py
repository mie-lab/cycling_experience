import re
import configparser
import matplotlib
import warnings
from utils.physiological_data_utils import *

matplotlib.use('Agg')

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("config.ini")

    root = Path(config['paths']['physiological_data_dir'])
    physiological_data_out = Path(config['filenames']['physiological_results_file'])

    files = find_xdf_files(root)
    report_sampling_rates(files)

    SAMPLING_RATE = 100
    all_metrics = []

    for f in files:
        log.info(f"--- Processing {f.name} ---")

        # Parsing Participant ID
        m = re.search(r'P(\d+)', f.stem)
        participant_id = int(m.group(1)) if m else np.nan

        try:
            data = extract_physiological_data(f)
            data, segments = preprocess_and_segment(data, SAMPLING_RATE)
        except Exception as e:
            log.error(f"Failed to process {f.name}: {e}")
            continue

        # --------------------------------------------------------
        # 1. Establish Baselines (Calibration)
        # --------------------------------------------------------
        # We store baseline values to compute "Deltas" (Reactivity)
        baseline_values = {
            'SCL': np.nan,
            'RMSSD': np.nan,
            'HR': np.nan,
            'SDNN': np.nan,
            'HF': np.nan
        }

        cal_seg = next((s for s in segments if s['segment_type'] == 'calibration'), None)

        if cal_seg:
            c_results = analyze_segment(
                cal_seg['EDA_Processed_Segment'],
                cal_seg['PPG_Processed_Segment'],
                SAMPLING_RATE
            )
            baseline_values['SCL'] = c_results.get('SCL_Mean', np.nan)
            baseline_values['RMSSD'] = c_results.get('HRV_RMSSD', np.nan)
            baseline_values['HR'] = c_results.get('PPG_Rate_Mean', np.nan)
            baseline_values['SDNN'] = c_results.get('HRV_SDNN', np.nan)
            baseline_values['HF'] = c_results.get('HRV_HF', np.nan)

            log.info(f" -> Baseline: SCL={baseline_values['SCL']:.2f}, HR={baseline_values['HR']:.1f}")
        else:
            log.warning("No calibration segment found; Deltas will be NaN.")

        # --------------------------------------------------------
        # 2. Process Video Segments
        # --------------------------------------------------------
        video_segments = [s for s in segments if s['segment_type'] == 'video']

        for seg in video_segments:
            # A. Get All Raw Metrics (from your updated function)
            m = analyze_segment(
                seg['EDA_Processed_Segment'],
                seg['PPG_Processed_Segment'],
                SAMPLING_RATE
            )

            # B. Calculate Reactivity (Deltas)
            # Formula: Segment_Value - Baseline_Value

            # EDA Delta
            m['SCL_Delta'] = m['SCL_Mean'] - baseline_values['SCL']

            # Cardiac Deltas
            m['HR_Delta'] = m['PPG_Rate_Mean'] - baseline_values['HR']
            m['HRV_RMSSD_Delta'] = m['HRV_RMSSD'] - baseline_values['RMSSD']
            m['HRV_SDNN_Delta'] = m['HRV_SDNN'] - baseline_values['SDNN']
            m['HRV_HF_Delta'] = m['HRV_HF'] - baseline_values['HF']

            # C. Add Metadata
            m.update({
                'participant_id': participant_id,
                'segment_id': seg['segment_id'],
                'duration': seg['end_time'] - seg['start_time'],
            })

            all_metrics.append(m)

    # --- Save Results ---
    if all_metrics:
        df = pd.DataFrame(all_metrics)

        # Definition of ALL columns to save (Order matters for readability)
        cols = [
            'participant_id', 'segment_id', 'duration',

            # --- 1. EDA Phasic (Event-Related) ---
            'SCR_Peaks_N',
            'SCR_Peaks_Amplitude_Mean',
            'SCR_Peaks_Amplitude_SD',
            'SCR_Peaks_Amplitude_Max',
            'SCR_Mean',
            'SCR_SD',
            'SCR_AUC',
            'SCR_Recovery_Slope',

            # --- 2. EDA Tonic (Background Levels) ---
            'SCL_Mean',
            'SCL_Delta',  # Calculated in main
            'SCL_SD',
            'SCL_Max',
            'SCL_Min',
            'SCL_Slope',

            # --- 3. EDA Sliding Window ---
            'SCL_window_mean',
            'SCL_window_sd',
            'SCL_window_slope_mean',
            'SCL_window_slope_max',
            'SCL_window_slope_min',

            # --- 4. PPG Heart Rate ---
            'PPG_Rate_Mean',
            'HR_Delta',  # Calculated in main
            'PPG_Rate_SD',
            'HR_Min',
            'HR_Max',

            # --- 5. HRV Time Domain ---
            'HRV_RMSSD',
            'HRV_RMSSD_Delta',  # Calculated in main
            'HRV_SDNN',
            'HRV_SDNN_Delta',  # Calculated in main
            'HRV_MeanNN',
            'HRV_pNN20',
            'HRV_pNN50',
            'HRV_SD1',

            # --- 6. HRV Frequency Domain ---
            'HRV_LF',
            'HRV_HF',
            'HRV_HF_Delta',  # Calculated in main
            'HRV_LFHF',
        ]

        # Filter to ensure we don't crash if a column is missing
        # (Calculates the intersection of desired cols and existing cols)
        final_cols = [c for c in cols if c in df.columns]

        # Check if any new metrics were missed
        missing = set(df.columns) - set(final_cols)
        if missing:
            log.info(f"Note: The following extra columns were generated but not explicitly ordered: {missing}")
            # Optionally append them to the end:
            # final_cols.extend(list(missing))

        df = df[final_cols]
        df.to_csv(physiological_data_out, index=False)
        log.info(f"All metrics saved to: {physiological_data_out}")
    else:
        log.warning(" No metrics extracted.")

if __name__ == "__main__":
    # Ensure the save_segment_data_csv function is defined outside of main()
    # and that all imports (Path, numpy, pandas) are present at the top of your script.
    main()