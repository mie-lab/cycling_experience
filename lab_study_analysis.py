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

    online_results_file = Path(config["filenames"]["survey_results_file"])
    online_sequence_file = Path(config["filenames"]["online_sequence_file"])

    video_predictions_file = Path(config["filenames"]["video_predictions_file"])

    lab_results_file = Path(config["filenames"]["lab_study_results_file"])
    lab_sequence_file = Path(config['filenames']['lab_video_sequence_file'])
    lab_setup_file = Path(config['filenames']['lab_experiment_setup_file'])

    # Define and create the output directory.
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # PHASE 1: LOAD DATA
    # ==============================================================================
    log.info('Phase 2.1: Loading and Processing Online Survey Data')
    survey_df = pd.read_excel(online_results_file).set_index(c.PARTICIPANT_ID)
    online_seq_df = pd.read_csv(online_sequence_file, parse_dates=['seq_start', 'seq_end'])

    # Transform survey responses into a long format
    survey_results_df = utils.processing_utils.transform_to_long_df(survey_df, online_seq_df, id_col=c.PARTICIPANT_ID)

    # Clean and preprocess the survey response data
    survey_results_df = utils.processing_utils.filter_aggregate_results(
        survey_results_df,
        age=True,
        gender=True,
        by_country=True,
        cycling_environment=True,
        cycling_frequency=True,
        cycling_confidence=True
    )
    survey_results_df = utils.processing_utils.add_valence_arousal(survey_results_df)
    survey_results_df = utils.processing_utils.assign_affective_states(survey_results_df)

    # Aggregate individual ratings to get a single summary score per video.
    # TODO: doublecheck how the function differ from aggregate_video_level_scores
    online_video_level_scores = utils.processing_utils.calculate_video_level_scores(
        survey_results_df, rating_col=c.AG
    ).rename(columns={c.VALENCE: 'valence_online', c.AROUSAL: 'arousal_online'})

    log.info('Phase 2.2: Loading and Processing Lab Study Data')
    lab_results_df = pd.read_excel(lab_results_file).set_index(c.PARTICIPANT_ID, drop=True)
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

    demographics_df = lab_results_df[c.DEMOGRAPHIC_COLUMNS]

    lab_seq_df = pd.read_csv(lab_sequence_file)
    experiment_setup = pd.read_csv(lab_setup_file, header=None).set_index(0, drop=True)

    lab_results_df = (
        lab_results_df
        .drop(columns=c.DEMOGRAPHIC_COLUMNS + [c.START, c.END])
        .replace(r'^\s*(\d+).*$', r'\1', regex=True)
        .apply(pd.to_numeric, errors='coerce')
    )

    log.info('Phase 2.3: Loading and Processing Video Prediction Data')
    video_score_predictions = pd.read_csv(video_predictions_file)

    # ==============================================================================
    # PHASE 2: BLOCK 1 ANALYSIS (VIDEO VALIDATION)
    # ==============================================================================

    log.info("\n--- Block 1 Analysis ---")
    df_baseline = utils.helper_functions.get_trial_dict(
        lab_results_df,
        experiment_setup,
        c.TRIAL_1,
        c.VIDEO_COUNTS
    )
    df1 = utils.helper_functions.trial_dict_to_df(df_baseline)
    df1 = utils.processing_utils.add_valence_arousal(df1, 'rating')

    # Aggregate individual ratings to get a single summary score per video.
    lab_video_scores = utils.processing_utils.calculate_video_level_scores(df1, rating_col='rating')

    # Merge lab scores with online scores and model predictions for comparison.
    video_level_scores = pd.merge(lab_video_scores, online_video_level_scores, on=c.VIDEO_ID_COL, how='left')
    video_level_scores = pd.merge(video_level_scores, video_score_predictions, on=c.VIDEO_ID_COL, how='left')

    # Plot comparisons between lab scores, online scores, and model predictions.
    utils.helper_functions.get_video_level_metrics(video_level_scores, c.VALENCE, 'valence_prediction',
                                                   'valence_online')

    print(video_level_scores[['valence', 'valence_online', 'valence_prediction']])
    utils.helper_functions.check_variance_homogeneity(df1, c.PARTICIPANT_ID, c.VALENCE)
    utils.helper_functions.check_variance_homogeneity(df1, c.VIDEO_ID_COL, c.VALENCE)

    utils.plotting_utils.plot_bland_altman(
        df=video_level_scores,
        measurement1='valence',
        measurement2='valence_online',
        label_col=c.VIDEO_ID_COL,
        save_path=output_dir / 'bland_altman_valence_video.png'
    )

    utils.plotting_utils.plot_bland_altman(
        df=video_level_scores,
        measurement1='arousal',
        measurement2='arousal_online',
        label_col=c.VIDEO_ID_COL,
        save_path=output_dir / 'bland_altman_arousal_video.png'
    )

    # ==============================================================================
    # PHASE 3: BLOCK 2 ANALYSIS (EQUAL SCENARIO)
    # ==============================================================================

    log.info("\n--- Analyzing Block 2 (Equal Scenario) ---")
    df_equal = utils.helper_functions.load_and_process_trial_data(
        lab_results_df,
        experiment_setup,
        lab_seq_df,
        c.TRIAL_2_PARAMS,
        video_level_scores
    )
    df_equal = pd.merge(df_equal, demographics_df, on=c.PARTICIPANT_ID, how='left')

    file_name = output_dir / 'trial_2_overall_ratings.png'
    utils.plotting_utils.plot_violin_panels(
        df_equal,
        sequence_order=c.TRIAL_2_PLOT_ORDER,
        save_path=file_name
    )

    file_name = output_dir / 'trial_2_rankings.png'
    utils.plotting_utils.plot_ranking_by_nb_pos(df_equal, 'NB_position', c.TRIAL_2_PARAMS, 'ranking', file_name)

    # ==============================================================================
    # PHASE 4: BLOCK 3 ANALYSIS (POSITIVE SCENARIO)
    # ==============================================================================
    log.info("\n--- Analyzing Block 3 (Positive Scenario) ---")

    # --- 1. Load, Process, and Merge Data ---
    df_positive = utils.helper_functions.load_and_process_trial_data(
        lab_results_df,
        experiment_setup,
        lab_seq_df,
        c.TRIAL_3_PARAMS,
        video_level_scores,
        'NB'
    )
    df_positive = pd.merge(df_positive, demographics_df, on=c.PARTICIPANT_ID, how='left')

    # --- 2. Generate and Save Visualizations ---
    ratings_plot_path = output_dir / 'trial_2_overall_ratings.png'
    utils.plotting_utils.plot_violin_panels(
        df=df_equal,
        sequence_order=c.TRIAL_2_PLOT_ORDER,
        save_path=ratings_plot_path
    )

    rankings_plot_path = output_dir / 'trial_2_rankings.png'
    utils.plotting_utils.plot_ranking_by_nb_pos(
        df=df_equal,
        position_col='NB_position',
        params=c.TRIAL_2_PARAMS,
        ranking_col='ranking',
        save_path=rankings_plot_path
    )

    # ==============================================================================
    # PHASE 5: BLOCK 4 ANALYSIS (NEGATIVE SCENARIO)
    # ==============================================================================
    log.info("\n--- Analyzing Block 4 (Negative Scenario) ---")

    # --- 1. Load, Process, and Merge Data ---
    df_negative = utils.helper_functions.load_and_process_trial_data(
        lab_results_df,
        experiment_setup,
        lab_seq_df,
        c.TRIAL_4_PARAMS,
        video_level_scores,
        'B'
    )
    df_negative = pd.merge(df_negative, demographics_df, on=c.PARTICIPANT_ID, how='left')

    # --- 2. Generate and Save Visualizations ---
    ratings_plot_path = output_dir / 'trial_4_overall_ratings.png'
    utils.plotting_utils.plot_violin_panels(
        df_negative,
        sequence_order=c.TRIAL_4_PLOT_ORDER,
        save_path=ratings_plot_path
    )

    rankings_plot_path = output_dir / 'trial_4_rankings.png'
    utils.plotting_utils.plot_ranking_by_nb_pos(
        df=df_negative,
        position_col='B_position',
        params=c.TRIAL_4_PARAMS,
        ranking_col='ranking',
        save_path=rankings_plot_path
    )

    # ==============================================================================
    # PHASE 6: STATISTICAL MODELING (LINEAR MIXED-EFFECTS MODELS)
    # ==============================================================================
    log.info("\n--- Phase 4: Running Linear Mixed-Effects Model Analyses ---")

    # ------------------------------------------------------------------------------
    # Part 1: Baseline Models - Analyzing Each Scenario in Isolation
    # ------------------------------------------------------------------------------

    log.info("--- Analyzing [Equal Scenario]: Does spoiler position matter in 50/50 case? ---")
    df_equal['NB_count'] = df_equal['sequence_list'].apply(lambda seq: seq.count('NB'))
    formula = "valence ~ C(NB_position)"

    utils.lmm_utils.run_lmm(
        df=df_equal[df_equal['NB_count'] < 2],  # filter for sequences with only one spoiler to isolate the effect.
        formula=formula,
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    log.info("--- Analyzing [Negative Scenario]: Does spoiler position impact a mostly negative experience? ---")
    formula = "valence ~ C(B_position)"
    neg_scenario_model_simple = utils.lmm_utils.run_lmm(
        df=df_negative,
        formula=formula,
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    # Formula with demographic interaction
    formula = "valence ~ C(B_position) * C(Gender)"  # change this to test other demographics
    neg_scenario_model_complex = utils.lmm_utils.run_lmm(
        df=df_negative,
        formula=formula,
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    # compare simple and more complex mixed-effects models
    utils.lmm_utils.lr_test_mixed(neg_scenario_model_simple, neg_scenario_model_complex, mixture=False)

    log.info("--- Analyzing [Positive Scenario]: Does spoiler position impact a mostly positive experience? ---")
    formula = "valence ~ C(NB_position)"
    pos_scenario_model_simple = utils.lmm_utils.run_lmm(
        df=df_positive,
        formula=formula,
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    # Formula with demographic interaction
    formula = "valence ~ C(NB_position) + C(Gender)"  # change this to test other demographics
    pos_scenario_model_complex = utils.lmm_utils.run_lmm(
        df=df_positive,
        formula=formula,
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    # compare simple and more complex mixed-effects models
    utils.lmm_utils.lr_test_mixed(pos_scenario_model_simple, pos_scenario_model_complex, mixture=False)

    # ------------------------------------------------------------------------------
    # Part 2: Combined Analysis - Building More Complex Models
    # ------------------------------------------------------------------------------
    log.info("\n--- Preparing combined dataset for advanced modeling ---")
    df_combined = utils.processing_utils.prepare_combined_scenario_df(df_positive, df_negative)

    # --- Model 1: Main Effects Model ---
    log.info("--- Model 1: Testing Main Effects of Position and Scenario ---")
    log.info("RQ: Do spoiler position and scenario each have an independent effect on valence?")
    formula_1 = "valence ~ C(spoiler_position) + C(scenario)"
    model_1_additive = utils.lmm_utils.run_lmm(
        df=df_combined,
        formula=formula_1,
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    # --- Model 2: Interaction Model ---
    log.info("--- Model 2: Do spoiler position and scenario interact? ---")
    log.info("RQ: Does the effect of a spoiler's position on overall valence DEPEND on scenario (positive/negative)?")
    formula_2 = "valence ~ C(spoiler_position) * C(scenario)"
    model_2_interactive = utils.lmm_utils.run_lmm(
        df=df_combined,
        formula=formula_2,
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    # --- Model 3: Interaction Model with Random Slopes ---
    log.info("--- Model 3: Adding individual sensitivity (Random Slopes) ---")
    log.info("RQ: Do participants vary in their sensitivity to spoiler position?")
    formula_3 = "valence ~ C(spoiler_position) * C(scenario)"
    re_formula_3 = "~ spoiler_position"
    model_3_slopes = utils.lmm_utils.run_lmm(
        df=df_combined,
        formula=formula_3,
        re_formula=re_formula_3,
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    # --- Model 4: Three-Way Interaction Model (Moderation) ---
    log.info("--- Model 4: Does gender moderate the interaction effect? ---")
    log.info("RQ: Is the interaction between spoiler position and scenario different for different environments?")
    formula_4 = "valence ~ C(spoiler_position) * C(scenario) * C(Cycling_environment)"
    model_4_moderation = utils.lmm_utils.run_lmm(
        df=df_combined,
        formula=formula_4,
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    # ------------------------------------------------------------------------------
    # PART 3: PERFORM SEQUENTIAL LIKELIHOOD RATIO TESTS
    # ------------------------------------------------------------------------------
    log.info("\n--- Running sequential model comparisons ---")

    # --- Test 1: Justifying the Interaction ---
    log.info("\n--- Comparison: Model 1 (Main Effects) vs. Model 2 (Interaction) ---")
    utils.lmm_utils.lr_test_mixed(model_1_additive, model_2_interactive)

    # --- Test 2: Justifying Random Slopes ---
    log.info("\n--- Comparison: Model 2 (Interaction) vs. Model 3 (Random Slopes) ---")
    utils.lmm_utils.lr_test_mixed(model_2_interactive, model_3_slopes)

    # --- Test 3: Justifying Moderation by Gender ---
    log.info("\n--- Comparison: Model 2 (Interaction) vs. Model 4 (Moderation) ---")
    utils.lmm_utils.lr_test_mixed(model_2_interactive, model_4_moderation)

    # ------------------------------------------------------------------------------
    # Part 4: Testing Alternative Explanations
    # ------------------------------------------------------------------------------
    log.info("\n--- Model 5: Is it about quantity, not position? ---")
    log.info("RQ: Can valence be explained simply by the number of spoilers in the sequence?")

    formula = "valence ~ C(NB_count)"
    quantity_model = utils.lmm_utils.run_lmm(
        df=df_combined,
        formula=formula,
        groups_col=c.PARTICIPANT_ID,
        convergence_method='cg'
    )

    log.info("\n--- Model 6: Testing the Peak-Valence Heuristic ---")
    log.info("RQ: Is the valence rating driven by the most emotionally intense moment (positive or negative peak)?")

    formula = "valence ~ pos_peak_valence + neg_peak_valence + end_valence"
    re_formula = '~ C(scenario)'
    heuristic_model = utils.lmm_utils.run_lmm(
        df=df_combined,
        formula=formula,
        groups_col=c.PARTICIPANT_ID,
        re_formula=re_formula,
        convergence_method='lbfgs'
    )

    # ==============================================================================
    # PART 2: PERFORM AKAIKE and BIKAIKE COMPARISONS
    # ==============================================================================

    log.info("\n--- Creating a Baseline (Intercept-Only) Model ---")
    baseline_formula = "valence ~ 1"
    baseline_model = utils.lmm_utils.run_lmm(
        df=df_combined,
        formula=baseline_formula,
        groups_col=c.PARTICIPANT_ID
    )

    # --- Compare the best models from each theory using AIC/BIC ---
    best_positional_model = model_2_interactive

    log.info("\n--- Final Model Comparison (Lower AIC/BIC is Better) ---")
    log.info(f"Intercept only Model AIC: {baseline_model.aic:.2f}, BIC: {baseline_model.bic:.2f}")
    log.info(f"Positional Model AIC: {best_positional_model.aic:.2f}, BIC: {best_positional_model.bic:.2f}")
    log.info(f"Quantity Model AIC:   {quantity_model.aic:.2f}, BIC: {quantity_model.bic:.2f}")
    log.info(f"Heuristic Model AIC:  {heuristic_model.aic:.2f}, BIC: {heuristic_model.bic:.2f}")



if __name__ == "__main__":
    main()
