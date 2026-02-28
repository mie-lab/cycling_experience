import constants as c
import utils.helper_functions
import utils.plotting_utils
import utils.processing_utils
import utils.lmm_utils
import configparser
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    """
    Main function to run the entire analysis pipeline from start to finish.
    """
    # ==============================================================================
    # PHASE 0: SETUP & CONFIGURATION
    # ==============================================================================
    log.info("Loading configuration…")
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("config.ini")

    # --- Load file paths and define directories ---
    survey_results_file = Path(config["filenames"]["survey_results_file"])
    sequence_file = Path(config["filenames"]["online_sequence_file"])
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # PHASE 1: LOAD & PREPROCESS DATA
    # ==============================================================================
    log.info("Loading and preprocessing data...")

    survey_df = pd.read_excel(survey_results_file).set_index(c.PARTICIPANT_ID)
    seq_df = pd.read_csv(sequence_file, parse_dates=['seq_start', 'seq_end'])

    # Clean and preprocess the survey response data
    survey_results_df = utils.processing_utils.transform_to_long_df(survey_df, seq_df, id_col=c.PARTICIPANT_ID)
    survey_results_df = utils.processing_utils.filter_results(survey_results_df)
    survey_results_df = utils.processing_utils.add_valence_arousal(survey_results_df)

    # =============================================================================
    # PHASE 2: DEMOGRAPHIC SUMMARY AND AGGREGATION
    # =============================================================================
    log.info("Generating demographic summaries and aggregating categories...")

    # Aggregate demographic categories to ensure sufficient sample size in each group
    survey_results_df = utils.processing_utils.aggregate_by_characteristics(
        survey_results_df,
        age=True,
        gender=True,
        cycling_frequency=True,
        cycling_confidence=True,
        cycling_purpose=True,
        cycling_environment=True,
        familiarity=True,
        is_swiss=True
    )

    # Generate demographic summary table with aggregated categories
    utils.plotting_utils.generate_demographic_table(
       survey_results_df,
       demo_cols=c.DEMOGRAPHIC_COLUMNS,
       output_path=Path(output_dir / 'demographic_summary_aggregated.csv')
    )

    reliability_df = utils.processing_utils.compute_video_reliability(survey_results_df)
    log.info(f"\n--- Video Reliability (ICC) ---\n{reliability_df}\n")

    # =============================================================================
    # PHASE 3: EXPLORATORY ANALYSIS & VISUALIZATION (RQ1)
    # =============================================================================
    log.info("Generating exploratory visualizations for RQ1...")

    # Plot affect grid for all videos combined
    utils.plotting_utils.plot_affect_grid_subplots(
       survey_results_df,
       ncols=5, nrows=6,
       normalize="count",
       robust_vmax_percentile=99,
       save_path=output_dir / "affect_grid_all_videos.png",
    )

    # Plot separate heatmaps for each video
    utils.plotting_utils.plot_affect_grid(
       survey_results_df,
       video_id=None,
       save_path=output_dir / "affect_grid_heatmaps")

    # Plot affect grid usage with marginals (overall)
    utils.plotting_utils.plot_affect_grid_usage_with_marginals(
       survey_results_df,
       normalize="count",
       save_path=output_dir / "affect_grid_usage.png",
    )

    # Plot affective quadrant distribution for each video
    utils.plotting_utils.create_quadrant_distribution_plot(
       survey_results_df,
       video_id_column=c.VIDEO_ID_COL,
       output_path=output_dir / "affective_quadrant_distribution_per_video.png"
    )

    # ==============================================================================
    # PHASE 4: VIDEO-LEVEL AFFECT METRICS & PF/NF DISAGREEMENT ANALYSIS (RQ2)
    # ==============================================================================
    log.info("Calculating video-level affect metrics and analyzing PF/NF disagreement for RQ2...")

    # Calculate video-level affect metrics and save to CSV
    video_level_scores = utils.processing_utils.calculate_video_level_scores(
        survey_results_df,
        output_path=Path(output_dir / "video_level_affect_metrics.csv")
    )

    # Plot video-level affect means with vectors for OE groups
    utils.plotting_utils.plot_video_affect_means_with_vectors(
       video_level_scores,
       save_path=output_dir / "video_affect_vectors_oe.png",
       oe_col="oe_mode",
    )

    # Add PF/NF label counts to video-level scores and save to CSV
    video_level_scores = utils.processing_utils.add_factor_counts_to_scores(
        video_level_scores,
        survey_results_df,
        c.LABEL_COLS,
        video_col=c.VIDEO_ID_COL,
        pf_col=c.PF,
        nf_col=c.NF
    )

    video_level_scores, _ = utils.processing_utils.pf_nf_disagreement_analysis(
        video_level_scores,
        label_cols=c.LABEL_COLS,
        out_csv_path=output_dir / "video_level_affect_metrics_pf_nf.csv"
    )

    # ==============================================================================
    # PHASE 4.5: MARGINAL AFFECTIVE DRIVER ANALYSIS (LOCAL DISPLACEMENT)
    # ==============================================================================
    log.info("Phase 4.5: Calculating marginal affective pull of environmental drivers...")

    # 1. Identify drivers through local displacement calculation
    affective_drivers = utils.processing_utils.get_marginal_affective_drivers(
        survey_results_df,
        c.LABEL_COLS
    )

    # 2. Plot the affective force field
    utils.plotting_utils.plot_conditional_displacement_vectors(
        affective_drivers,
        output_dir / "environmental_affective_drivers_force_field.png",
        limit=0.35
    )

    # =============================================================================
    # PHASE 5: SUBGROUP ANALYSIS & VISUALIZATION (RQ3)
    # =============================================================================
    log.info("Calculating subgroup video-level scores and generating visualizations for RQ3...")

    # Calculate video-level scores by demographic subgroups and save to separate CSVs
    for col in c.DEMOGRAPHIC_COLUMNS:
        utils.processing_utils.calculate_video_level_scores_by_subgroup(
            survey_results_df,
            subgroup_col=col,
            min_participants=20,
            output_path=output_dir / f"video_metrics_by_{col}.csv"
        )

    files = {
        "Frequency": output_dir / "video_metrics_by_Cycling_frequency.csv",
        "Confidence": output_dir / "video_metrics_by_Cycling_confidence.csv",
        "Purpose": output_dir / "video_metrics_by_Cycling_purpose.csv",
        "Environment": output_dir / "video_metrics_by_Cycling_environment.csv",
        "Gender": output_dir / "video_metrics_by_Gender.csv",
        "Age": output_dir / "video_metrics_by_Age.csv",
        "Familiarity": output_dir / "video_metrics_by_is_swiss.csv",
    }

    subgroup_cols = {
        "Frequency": "Cycling_frequency",
        "Confidence": "Cycling_confidence",
        "Purpose": "Cycling_purpose",
        "Environment": "Cycling_environment",
        "Gender": "Gender",
        "Age": "Age",
        "Familiarity": "is_swiss",
    }

    metrics = [
        "dispersion_mean_distance",
        "anisotropy_index",
        "polarization_index",
        "affect_state_entropy",
        "pf_nf_label_entropy"
    ]

    metric_labels = {
        "dispersion_mean_distance": "Dispersion",
        "anisotropy_index": "Anisotropy",
        "polarization_index": "Polarization",
        "affect_state_entropy": "Quadrant entropy",
        "pf_nf_label_entropy": "Cue entropy"
    }

    # Plot subgroup metrics with bootstrap confidence intervals and save the figure
    utils.plotting_utils.plot_subgroup_metrics_bootstrap(
        files=files,
        subgroup_cols=subgroup_cols,
        metrics=metrics,
        metric_labels=metric_labels,
        overall_path=output_dir / "video_level_affect_metrics_pf_nf.csv",
        n_boot=1000,
        seed=42,
        min_videos=5,
        figsize=(14, 7.5),
        save_path=output_dir / "subgroups_overlay_metrics.png",
    )

    # ==============================================================================
    # DIAGNOSTIC ANALYSIS: CORRELATIONS BETWEEN CLIP-LEVEL METRICS
    # ==============================================================================

    metric_cols = [
        "valence_mean",
        "valence_sd",
        "arousal_mean",
        "arousal_sd",
        "oe_mean",
        "oe_mode",
        "dispersion_mean_distance",
        "anisotropy_index",
        "polarization_index",
        "affect_state_entropy",
        "pf_nf_label_entropy"
    ]

    utils.plotting_utils.plot_metric_correlations(
        df=video_level_scores,
        cols_to_plot=metric_cols,
        output_dir=output_dir / "diagnostic_analysis.png"
    )

    # Plot valence and arousal distributions by observer experience (OE)
    utils.plotting_utils.plot_valence_arousal_by_oe(
        survey_results_df,
        oe_col=c.OE,
        oe_order=c.OE_ORDER,
        valence_col=c.VALENCE,
        arousal_col=c.AROUSAL,
        kind="box",
        save_path=output_dir / "valence_arousal_by_oe.png",
    )

    # Plot disagreement geometry vs. PF/NF label entropy and save the figure
    utils.plotting_utils.plot_disagreement_geometry_vs_cues(
        video_level_scores,
        save_path=output_dir / "metrics_vs_cues_valence.png",
        video_col=c.VIDEO_ID_COL,
        experience_col='valence_mean',
        cue_col="pf_nf_label_entropy"
    )

    # ==============================================================================
    log.info('Analysis complete.')


if __name__ == "__main__":
    main()
