import constants as c
import utils.plot_utils
import utils.process_utils
import configparser
import logging
from pathlib import Path
import pandas as pd
import os

os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 4)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    """Online survey: affect ratings, disagreement between raters, demographic subgroups."""
    # ==============================================================================
    # PHASE 0: SETUP & CONFIGURATION
    # ==============================================================================
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("config.ini")

    survey_results_file = Path(config["filenames"]["survey_results_file"])
    sequence_file = Path(config["filenames"]["online_sequence_file"])
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # PHASE 1: LOAD & PREPROCESS DATA
    # ==============================================================================
    log.info("Phase 1: Load data")

    survey_df = pd.read_excel(survey_results_file).set_index(c.PARTICIPANT_ID)
    seq_df = pd.read_csv(sequence_file, parse_dates=['seq_start', 'seq_end'])

    survey_results_df = utils.process_utils.transform_to_long_df(
        survey_df, seq_df, id_col=c.PARTICIPANT_ID
    )
    survey_results_df = utils.process_utils.filter_results(survey_results_df)
    survey_results_df = utils.process_utils.add_valence_arousal(survey_results_df)

    # =============================================================================
    # PHASE 2: DEMOGRAPHICS
    # =============================================================================
    log.info("Phase 2: Demographics")

    # Merge small categories so every group is large enough
    survey_results_df = utils.process_utils.agg_by_chars(
        survey_results_df, age=True, gender=True, cycling_frequency=True,
        cycling_confidence=True, cycling_purpose=True, cycling_environment=True,
        familiarity=True, is_swiss=True
    )
    utils.plot_utils.generate_demographic_table(
        survey_results_df,
        demo_cols=c.DEMOGRAPHIC_COLUMNS,
        output_path=output_dir / 'demographic_summary_aggregated.csv'
    )

    reliability_df = utils.process_utils.compute_video_reliability(survey_results_df)
    log.info(f"Video reliability (ICC):\n{reliability_df.round(3)}")

    # =============================================================================
    # PHASE 3: AFFECT GRID RATINGS (RQ1)
    # =============================================================================
    log.info("Phase 3: Affect grid plots")

    utils.plot_utils.plot_affect_grid_subplots(
        survey_results_df,
        ncols=5, nrows=6,
        normalize="count",
        robust_vmax_percentile=99,
        save_path=output_dir / "affect_grid_all_videos.png",
    )
    utils.plot_utils.plot_affect_grid(
        survey_results_df,
        video_id=None,
        save_path=output_dir / "affect_grid_heatmaps"
    )
    utils.plot_utils.plot_affect_grid_usage_with_marginals(
        survey_results_df,
        normalize="count",
        save_path=output_dir / "affect_grid_usage.png",
    )
    utils.plot_utils.create_quadrant_distribution_plot(
        survey_results_df,
        video_id_column=c.VIDEO_ID_COL,
        output_path=output_dir / "affective_quadrant_distribution_per_video.png"
    )

    # ==============================================================================
    # PHASE 4: VIDEO-LEVEL METRICS & PF/NF DISAGREEMENT (RQ2)
    # ==============================================================================
    log.info("Phase 4: Video-level metrics")

    video_level_scores = utils.process_utils.calc_video_level_scores(
        survey_results_df,
        output_path=output_dir / "video_level_affect_metrics.csv"
    )
    utils.plot_utils.plot_video_affect_means_with_vectors(
        video_level_scores,
        save_path=output_dir / "video_affect_vectors_oe.png",
        oe_col="oe_mode",
    )

    # Positive (PF) and negative (NF) factor counts per video
    video_level_scores = utils.process_utils.add_factor_counts_to_scores(
        video_level_scores,
        survey_results_df,
        c.LABEL_COLS,
        video_col=c.VIDEO_ID_COL,
        pf_col=c.PF,
        nf_col=c.NF
    )
    video_level_scores, _ = utils.process_utils.pf_nf_disagreement_analysis(
        video_level_scores,
        label_cols=c.LABEL_COLS,
        out_csv_path=output_dir / "video_level_affect_metrics_pf_nf.csv"
    )

    # Marginal pull of each environmental factor on valence/arousal
    affective_drivers = utils.process_utils.get_marginal_affective_drivers(
        survey_results_df, c.LABEL_COLS
    )
    utils.plot_utils.plot_conditional_displacement_vectors(
        affective_drivers,
        output_dir / "environmental_affective_drivers_force_field.png",
        limit=0.35
    )

    # =============================================================================
    # PHASE 5: SUBGROUP ANALYSIS (RQ3)
    # =============================================================================
    log.info("Phase 5: Subgroups")

    for col in c.DEMOGRAPHIC_COLUMNS:
        utils.process_utils.calculate_video_level_scores_by_subgroup(
            survey_results_df,
            subgroup_col=col,
            min_participants=20,
            output_path=output_dir / f"video_metrics_by_{col}.csv"
        )

    subgroup_cols = {
        "Frequency": "Cycling_frequency",
        "Confidence": "Cycling_confidence",
        "Purpose": "Cycling_purpose",
        "Environment": "Cycling_environment",
        "Gender": "Gender",
        "Age": "Age",
        "Familiarity": "is_swiss",
    }
    metric_labels = {
        "dispersion_mean_distance": "Dispersion",
        "anisotropy_index": "Anisotropy",
        "polarization_index": "Polarization",
        "affect_state_entropy": "Quadrant entropy",
        "pf_nf_label_entropy": "Cue entropy"
    }
    utils.plot_utils.plot_subgroup_metrics_bootstrap(
        files={k: output_dir / f"video_metrics_by_{v}.csv" for k, v in subgroup_cols.items()},
        subgroup_cols=subgroup_cols,
        metrics=list(metric_labels),
        metric_labels=metric_labels,
        overall_path=output_dir / "video_level_affect_metrics_pf_nf.csv",
        n_boot=1000,
        seed=42,
        min_videos=5,
        figsize=(14, 7.5),
        save_path=output_dir / "subgroups_overlay_metrics.png",
    )

    # ==============================================================================
    # DIAGNOSTICS: CORRELATIONS BETWEEN VIDEO-LEVEL METRICS
    # ==============================================================================
    rating_cols = ["valence_mean", "valence_sd", "arousal_mean", "arousal_sd",
                   "oe_mean", "oe_mode"]
    utils.plot_utils.plot_metric_correlations(
        df=video_level_scores,
        cols_to_plot=rating_cols + list(metric_labels),
        output_dir=output_dir / "diagnostic_analysis.png"
    )
    utils.plot_utils.plot_valence_arousal_by_oe(
        survey_results_df,
        oe_col=c.OE,
        oe_order=c.OE_ORDER,
        valence_col=c.VALENCE,
        arousal_col=c.AROUSAL,
        kind="box",
        save_path=output_dir / "valence_arousal_by_oe.png",
    )
    utils.plot_utils.plot_disagreement_geometry_vs_cues(
        video_level_scores,
        save_path=output_dir / "metrics_vs_cues_valence.png",
        video_col=c.VIDEO_ID_COL,
        experience_col='valence_mean',
        cue_col="pf_nf_label_entropy"
    )

    log.info('Done.')


if __name__ == "__main__":
    main()
