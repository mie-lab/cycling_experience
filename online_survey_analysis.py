import constants as c
import utils.helper_functions
import utils.plotting_utils
import utils.processing_utils
import utils.lmm_utils
import pandas as pd
from scipy.stats import spearmanr
import configparser
import logging
from itertools import product
from pathlib import Path

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
    utils.plotting_utils.plot_affect_grid(survey_results_df, save_path=output_dir)

    # ==============================================================================
    # PHASE 2: DESCRIBE FINAL SAMPLE
    # ==============================================================================
    log.info("\n--- Final Sample Demographics ---")

    demographic_summary_table = utils.plotting_utils.generate_demographic_summary(
        df=survey_results_df,
        columns=c.DEMOGRAPHIC_COLUMNS,
        orders=c.CATEGORY_ORDERS,
        save_path=output_dir / 'demographic_overview.png'
    )
    log.info(demographic_summary_table)

    # ==============================================================================
    # PHASE 3: RUN STATISTICAL ANALYSES
    # ==============================================================================
    log.info("\n--- Running Statistical Analyses ---")

    # Center categorical predictors for easier interpretation of model intercepts
    survey_results_df = utils.processing_utils.add_midpoint_centered_column(survey_results_df, 'OE', c.OE_ORDER)
    survey_results_df = utils.processing_utils.add_midpoint_centered_column(survey_results_df, 'F', c.FAM_ORDER)

    # --- Correlation Tests ---
    spearman_corr, spearman_p = spearmanr(survey_results_df[c.VALENCE], survey_results_df['OE_centered'])
    log.info(f"Spearman corr (Valence vs OE): rho = {spearman_corr: .3f}, p = {spearman_p: .3g}")

    spearman_corr, spearman_p = spearmanr(survey_results_df[c.AROUSAL], survey_results_df['OE_centered'])
    log.info(f"Spearman corr (Arousal vs OE): rho = {spearman_corr: .3f}, p = {spearman_p: .3g}")

    # ==============================================================================
    # PHASE 4: LMMs ANALYSES - TESTING VALENCE AND AROUSAL
    # ==============================================================================
    log.info("\n--- LMM Analysis: Predicting Overall Experience (OE) ---")

    # --- Step 1: Fit a sequence of nested models ---

    log.info("\n--- Fitting Baseline Model (Intercept-Only) ---")
    model_baseline = utils.lmm_utils.run_lmm(
        survey_results_df,
        formula="OE_centered ~ 1",
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    log.info("\n--- Fitting Model with Valence ---")
    model_valence = utils.lmm_utils.run_lmm(
        survey_results_df,
        formula="OE_centered ~ valence",
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    log.info("\n--- Fitting Model with Arousal ---")
    model_arousal = utils.lmm_utils.run_lmm(
        survey_results_df,
        formula="OE_centered ~ arousal",
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    log.info("\n--- Fitting Additive Model (Valence + Arousal) ---")
    model_additive = utils.lmm_utils.run_lmm(
        survey_results_df,
        formula="OE_centered ~ valence + arousal",
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    log.info("\n--- Fitting Interactive Model (Valence * Arousal) ---")
    model_interactive = utils.lmm_utils.run_lmm(
        survey_results_df,
        formula="OE_centered ~ valence * arousal",
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    # --- Step 2: Perform Likelihood Ratio Tests to compare the models ---
    log.info("\n--- Performing Sequential Model Comparisons ---")

    log.info("\n--- Comparison 1: Is Valence a significant predictor? ---")
    utils.lmm_utils.lr_test_mixed(model_baseline, model_valence)

    log.info("\n--- Comparison 2: Is Arousal a significant predictor? ---")
    utils.lmm_utils.lr_test_mixed(model_baseline, model_arousal)

    log.info("\n--- Comparison 3: Does adding Arousal improve a model that already has Valence? ---")
    utils.lmm_utils.lr_test_mixed(model_valence, model_additive)

    log.info("\n--- Comparison 4: Does the interaction of Valence and Arousal significantly improve the model? ---")
    utils.lmm_utils.lr_test_mixed(model_additive, model_interactive)

    # ==============================================================================

    # --- LMM Analysis: Testing the effect of Familiarity (F_centered) on different outcomes ---

    formulas = ["valence ~ F_centered",
                'arousal ~ F_centered',
                'OE_centered ~ F_centered'
                ]

    for formula in formulas:
        dependent_var = formula.split(' ')[0]
        log.info(f"\n--- Analyzing Effect of Familiarity on: {dependent_var.upper()} ---")

        # --- Step 1: Fit the Baseline Model for this outcome ---
        log.info(f"--- Fitting Baseline Model: {dependent_var} ~ 1 ---")
        model_baseline = utils.lmm_utils.run_lmm(
            survey_results_df,
            formula=f"{dependent_var} ~ 1",
            groups_col=c.PARTICIPANT_ID,
            convergence_method='cg'
        )

        # --- Step 2: Fit the Full Model with the predictor ---
        log.info(f"--- Fitting Full Model: {formula} ---")
        model_full = utils.lmm_utils.run_lmm(
            survey_results_df,
            formula=formula,
            groups_col=c.PARTICIPANT_ID,
            convergence_method='cg'
        )

        # --- Step 3: Perform Likelihood Ratio Test ---
        log.info(f"--- Comparison: Is F_centered a significant predictor for {dependent_var}? ---")
        utils.lmm_utils.lr_test_mixed(model_baseline, model_full)
        log.info("-" * 70)

    # ==============================================================================
    formulas = ['is_Tension ~ F_centered',
                'is_Activation ~ F_centered',
                'is_Contentment ~ F_centered',
                'is_Deactivation ~ F_centered'
                ]
    for formula in formulas:
        log.info(f"\n--- Running Bayesian Logistic LMM: {formula} ---")
        utils.lmm_utils.run_bayes_logistic_lmm(
            df=survey_results_df,
            formula=formula,
            groups_col=c.PARTICIPANT_ID
        )

    log.info("\n--- Testing for Demographic Groups---")
    for dv, group_col in product(c.DEPENDENT_VARIABLES, c.DEMOGRAPHIC_COLUMNS):
        # remove groups with < 5 participants
        analysis_df = utils.processing_utils.filter_by_group_size(survey_results_df, group_col, c.PARTICIPANT_ID, 5)
        analysis_df[group_col] = analysis_df[group_col].astype('category')

        log.info(f"\n--- Analyzing Differences in '{dv.title()}' by '{group_col.replace('_', ' ').title()}' ---")
        log.info(f"Intercept (Reference Group): {analysis_df[group_col].cat.categories[0]}")

        utils.lmm_utils.run_lmm(
            analysis_df,
            formula=f"{dv} ~ C({group_col})",
            groups_col=c.PARTICIPANT_ID,
            convergence_method='cg'
        )

    # ==============================================================================
    # PHASE 4: GENERATE VISUALIZATIONS
    # ==============================================================================
    log.info("\n--- Generating and Saving Visualizations ---")
    # Generate and save all standard plots for the analysis
    utils.plotting_utils.plot_oe_distribution_by_fam(
        survey_results_df,
        c.FAM_ORDER,
        c.OE_ORDER,
        save_path=output_dir / 'oe_dist_by_familiarity.png'
    )

    utils.plotting_utils.plot_overall_experience(
        survey_results_df,
        c.OE,
        c.OE_ORDER,
        save_path=output_dir / 'overall_experience_by_video.png'
    )

    log.info("\nAnalysis pipeline finished successfully.")


if __name__ == "__main__":
    main()
