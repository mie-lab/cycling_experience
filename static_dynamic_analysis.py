import pandas as pd
import configparser
import utils.helper_functions
import utils.lmm_utils
import utils.plotting_utils
import utils.processing_utils
from pathlib import Path
import logging
import constants as c

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    """
    Main function to run the entire analysis pipeline from start to finish.
    """
    # ==============================================================================
    # PHASE 0: SETUP & CONFIGURATION
    # ==============================================================================
    log.info("Loading configuration...")
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("config.ini")

    lab_results_file = Path(config["filenames"]["lab_study_results_file"])
    lab_setup_file = Path(config['filenames']['lab_experiment_setup_file'])
    ground_truth_file = Path(config["filenames"]["video_info_ground_truth"])

    # Define and create the output directory.
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # PHASE 1: LOAD DATA
    # ==============================================================================

    log.info('Phase 2.2: Loading and Processing Lab Study Data')
    video_ground_truth_features = pd.read_csv(ground_truth_file)
    experiment_setup = pd.read_csv(lab_setup_file, header=None).set_index(0, drop=True)
    lab_results_df = pd.read_excel(lab_results_file).set_index(c.PARTICIPANT_ID, drop=True)

    # Clean and preprocess the lab study data
    lab_results_df = utils.processing_utils.filter_aggregate_results(
        lab_results_df,
        consent=False,
        duration=False,
        location=False,
        gender=True,
        cycling_environment=True,
        cycling_frequency=True,
        cycling_confidence=True
    )

    demographics_df = lab_results_df[c.DEMOGRAPHIC_COLUMNS].copy()
    video_rating_df = (
        lab_results_df
        .drop(columns=c.DEMOGRAPHIC_COLUMNS + [c.START, c.END])
        .replace(r'^\s*(\d+).*$', r'\1', regex=True)
        .apply(pd.to_numeric, errors='coerce')
    )

    # ==============================================================================
    # PHASE 2: PROCESS LAB STUDY SINGLE VIDEO RATINGS
    # ==============================================================================

    # Here column names do not make sense, thus should be ignored.
    # It is just the position of a rating that is matched to a name using the experiment setup file
    video_rating_dict = utils.helper_functions.get_trial_dict(
        video_rating_df,
        experiment_setup,
        c.TRIAL_1,
        c.VIDEO_COUNTS
    )
    df1 = utils.helper_functions.trial_dict_to_df(video_rating_dict)
    df1 = utils.processing_utils.add_valence_arousal(df1, 'rating')

    df1_with_features = df1.merge(
        video_ground_truth_features,
        on=c.VIDEO_ID_COL,
        how='left'
    )

    final_df = df1_with_features.merge(
        demographics_df,
        left_on=c.PARTICIPANT_ID,
        right_index=True,
        how='left'
    )
    print()

    # ==============================================================================
    # PHASE 3: STATIC-DYNAMIC ANALYSIS
    # ==============================================================================


if __name__ == "__main__":
    main()