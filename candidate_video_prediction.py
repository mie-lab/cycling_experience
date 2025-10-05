import pandas as pd
import configparser
from sklearn.preprocessing import MinMaxScaler
import sys
import constants as c
import utils.helper_functions
import utils.clustering_utils
import utils.plotting_utils
import utils.processing_utils
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
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

    # ==============================================================================
    # PHASE 1: LOAD DATA
    # ==============================================================================
    survey_results_file_name = Path(config["filenames"]["survey_results_file_name"])
    sequence_file_name = Path(config["filenames"]["online_sequence_file_name"])
    predicted_valences_file_name = Path(config['filenames']['video_predictions'])
    ground_truth_file = Path(config["paths"]["output_dir"]) / "ground_truth_features.csv"
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # PHASE 2: PROCESS SURVEY DATA
    # ==============================================================================
    log.info("Phase 2: Processing Survey Data")
    survey_df = pd.read_excel(survey_results_file_name).set_index(c.PARTICIPANT_ID)
    seq_df = pd.read_csv(sequence_file_name, parse_dates=['seq_start', 'seq_end'])

    # Transform survey responses into a long format
    survey_results_df = utils.processing_utils.transform_to_long_df(survey_df, seq_df, id_col=c.PARTICIPANT_ID)

    # Clean and preprocess the survey response data
    survey_results_df = utils.processing_utils.filter_aggregate_results(survey_results_df,
                                                                        age=True,
                                                                        gender=True,
                                                                        cycling_environment=True,
                                                                        cycling_frequency=True,
                                                                        cycling_confidence=True)
    survey_results_df = utils.processing_utils.add_valence_arousal(survey_results_df)
    survey_results_df = utils.processing_utils.assign_affective_states(survey_results_df)

    # Aggregate to video-level valence and arousal scores
    video_level_scores = utils.processing_utils.aggregate_video_level_scores(survey_results_df)

    # ==============================================================================
    # PHASE 3: CLUSTER VIDEOS BY SURVEY LABELS
    # ==============================================================================
    log.info("Phase 3: Clustering on Survey Labels")
    # Cluster videos based on survey labels (e.g., PF, NF)
    label_cluster_df = utils.clustering_utils.perform_clustering(
        video_level_scores,
        c.LABEL_COLS,
        'Cluster_labels',
        c.DEPENDENT_VARIABLES
    )
    # Visualize how the label-based clusters separate in the valence-arousal space
    utils.plotting_utils.plot_cluster_separation(
        label_cluster_df,
        'Cluster_labels',
        c.VALENCE,
        c.AROUSAL,
        id_col=c.VIDEO_ID_COL,
        save_path=output_dir / "Label Clusters scatterplots.png"
    )

    # ==============================================================================
    # PHASE 4: PROCESS AND ENRICH GEOSPATIAL DATA
    # ==============================================================================
    log.info("Phase 4: Merging Data and Final Clustering")
    video_geom = pd.read_csv(ground_truth_file)
    scaler = MinMaxScaler()

    # The selected columns are based on the top negative and positive labels from the survey analysis
    video_geom[c.DATA_COLS] = scaler.fit_transform(video_geom[c.DATA_COLS])

    data_cluster_df = video_level_scores.merge(video_geom, on=c.VIDEO_ID_COL, how='left')

    # Analyze and visualize the correlation between features and target variables
    log.info("Generating full correlation heatmap.")
    correlation_cols = c.DATA_COLS + [c.VALENCE, c.AROUSAL]
    utils.plotting_utils.plot_correlation_heatmap(
        data_cluster_df,
        correlation_cols,
        save_path=output_dir / "Correlation_Heatmap.png"
    )

    data_cluster_df = utils.clustering_utils.perform_clustering(
        data_cluster_df,
        c.DATA_COLS,
        'Cluster_data',
        c.DEPENDENT_VARIABLES
    )

    utils.plotting_utils.plot_cluster_separation(
        data_cluster_df,
        'Cluster_data',
        c.VALENCE,
        c.AROUSAL,
        id_col=c.VIDEO_ID_COL,
        save_path=output_dir / "Data Clusters scatterplots.png"
    )

    # ==============================================================================
    # PHASE 5: TRAIN PREDICTIVE MODEL
    # ==============================================================================
    log.info("Phase 5: Running Predictive Model")
    X = data_cluster_df[c.DATA_COLS].copy()

    # Find the optimal number of neighbors (k) for the predictive model
    rmses_df, best_rmse, best_k = utils.clustering_utils.compute_k_rmse(X, data_cluster_df, c.TARGET_COL)

    # Isolate the candidate videos (those not used in the survey)
    ids_to_exclude = data_cluster_df[c.VIDEO_ID_COL].unique()
    candidate_video_df = video_geom.loc[~video_geom[c.VIDEO_ID_COL].isin(ids_to_exclude)]

    # Predict valence for the candidate videos using the best k
    predicted_valences = utils.clustering_utils.predict_candidates(
        X,
        data_cluster_df,
        candidate_video_df,
        c.DATA_COLS,
        c.TARGET_COL,
        best_k
    )

    # ==============================================================================
    # PHASE 6: SAVE AND VISUALIZE RESULTS
    # ==============================================================================
    log.info("Phase 6: Saving and Visualizing Final Predictions")
    utils.plotting_utils.plot_candidate_predictions(
        predicted_valences,
        best_rmse,
        save_path=output_dir / "Candidate videos valence predictions.png")

    # Save the numerical predictions to a CSV file
    log.info(f"Saving predictions to {predicted_valences_file_name}")
    predicted_valences.to_csv(predicted_valences_file_name, index=False)

    log.info("Analysis pipeline finished successfully!")


if __name__ == "__main__":
    main()
