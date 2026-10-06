import re
import configparser
import matplotlib
import warnings
from utils.physio_utils import *

matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

SAMPLING_RATE = 100

# Reactivity = segment value - calibration (pre-stimulus) value
DELTAS = {'SCL_Delta': 'SCL_Mean', 'HR_Delta': 'PPG_Rate_Mean', 'HRV_RMSSD_Delta': 'HRV_RMSSD',
          'HRV_SDNN_Delta': 'HRV_SDNN', 'HRV_HF_Delta': 'HRV_HF'}

COLUMNS = [
    'participant_id', 'segment_id', 'duration',
    'SCR_Peaks_N', 'SCR_Peaks_Amplitude_Mean', 'SCR_Peaks_Amplitude_SD', 'SCR_Peaks_Amplitude_Max',
    'SCR_Mean', 'SCR_SD', 'SCR_AUC', 'SCR_Recovery_Slope',
    'SCL_Mean', 'SCL_Delta', 'SCL_SD', 'SCL_Max', 'SCL_Min', 'SCL_Slope',
    'SCL_window_mean', 'SCL_window_sd', 'SCL_window_slope_mean', 'SCL_window_slope_max', 'SCL_window_slope_min',
    'PPG_Rate_Mean', 'HR_Delta', 'PPG_Rate_SD', 'HR_Min', 'HR_Max',
    'HRV_RMSSD', 'HRV_RMSSD_Delta', 'HRV_SDNN', 'HRV_SDNN_Delta', 'HRV_MeanNN', 'HRV_pNN20', 'HRV_pNN50',
    'HRV_SD1', 'HRV_LF', 'HRV_HF', 'HRV_HF_Delta', 'HRV_LFHF',
]


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("config.ini")

    out_file = Path(config['filenames']['physiological_results_file'])
    files = find_xdf_files(Path(config['paths']['physiological_data_dir']))
    report_sampling_rates(files)

    all_metrics = []
    for f in files:
        log.info(f"--- Processing {f.name} ---")
        m = re.search(r'P(\d+)', f.stem)
        participant_id = int(m.group(1)) if m else np.nan

        try:
            data, segments = preprocess_and_segment(extract_physiological_data(f), SAMPLING_RATE)
        except Exception as e:
            log.error(f"Failed to process {f.name}: {e}")
            continue

        cal = next((s for s in segments if s['segment_type'] == 'calibration'), None)
        if cal:
            baseline = analyze_segment(cal['EDA_Processed_Segment'], cal['PPG_Processed_Segment'], SAMPLING_RATE)
            log.info(f"Baseline: SCL={baseline['SCL_Mean']:.2f}, HR={baseline['PPG_Rate_Mean']:.1f}")
        else:
            baseline = {}
            log.warning("No calibration segment; deltas will be NaN")

        for seg in (s for s in segments if s['segment_type'] == 'video'):
            m = analyze_segment(seg['EDA_Processed_Segment'], seg['PPG_Processed_Segment'], SAMPLING_RATE)
            m.update({delta: m[col] - baseline.get(col, np.nan) for delta, col in DELTAS.items()})
            m.update({'participant_id': participant_id, 'segment_id': seg['segment_id'],
                      'duration': seg['end_time'] - seg['start_time']})
            all_metrics.append(m)

    if not all_metrics:
        log.warning("No metrics extracted")
        return

    df = pd.DataFrame(all_metrics)
    log.info(f"Not saved: {sorted(set(df.columns) - set(COLUMNS))}")
    df = df[[col for col in COLUMNS if col in df.columns]]

    setup = pd.read_csv(config['filenames']['lab_experiment_setup_file'], header=None).set_index(0)
    validate_physio(df, setup)
    df.to_csv(out_file, index=False)
    log.info(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
