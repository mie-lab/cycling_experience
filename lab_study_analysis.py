import configparser
import utils.helper_functions
import utils.lmm_utils
import utils.plotting_utils
import utils.processing_utils
import logging
import constants as c
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import multipletests

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)





def run_ranking_analysis(df: pd.DataFrame, participant_col: str, condition_col: str, ranking_col: str):
    """
    Runs a Friedman Chi-Square test and calculates Kendall's W for ranking data.

    :param df: DataFrame containing the ranking data
    :param participant_col: Column name for participant IDs
    :param condition_col: Column name for the conditions being ranked (e.g., sequence_type)
    :param ranking_col: Column name for the assigned rank
    :return: Tuple of (Friedman statistic, p-value, Kendall's W)
    """
    # Pivot the data so rows are participants and columns are the ranked conditions
    pivot_df = df.pivot(index=participant_col, columns=condition_col, values=ranking_col).dropna()

    if pivot_df.empty:
        log.warning(
            "Pivoted ranking dataframe is empty. Check for missing values or duplicate participant/condition pairs.")
        return np.nan, np.nan, np.nan

    # Extract columns into a list of arrays for the Friedman test
    conditions = pivot_df.columns
    data_arrays = [pivot_df[cond].values for cond in conditions]

    # 1. Friedman Test
    stat, p_val = friedmanchisquare(*data_arrays)

    # 2. Kendall's W
    n_participants = pivot_df.shape[0]
    k_conditions = pivot_df.shape[1]

    # Formula: W = Chi^2 / (N * (k - 1))
    w = stat / (n_participants * (k_conditions - 1))

    log.info(f"--- Ranking Analysis ---")
    log.info(f"Conditions compared: {list(conditions)}")
    log.info(f"N Participants: {n_participants}")
    log.info(f"Friedman Chi-Square: {stat:.3f}, p-value: {p_val:.4f}")
    log.info(f"Kendall's W (Agreement): {w:.3f}")

    return stat, p_val, w

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

    log.info("Phase 1.1: Online survey (validation reference)")

    survey_df = pd.read_excel(online_results_file).set_index(c.PARTICIPANT_ID)
    online_seq_df = pd.read_csv(online_sequence_file, parse_dates=['seq_start', 'seq_end'])

    survey_results_df = utils.processing_utils.transform_to_long_df(survey_df, online_seq_df, id_col=c.PARTICIPANT_ID)
    survey_results_df = utils.processing_utils.filter_results(survey_results_df)
    survey_results_df = utils.processing_utils.add_valence_arousal(survey_results_df)
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

    online_video_level_scores = utils.processing_utils.calculate_video_level_scores(survey_results_df, lab_bool=True)
    online_video_level_scores = online_video_level_scores.rename(columns={c.VALENCE: 'valence_online', c.AROUSAL: 'arousal_online'})

    log.info("Phase 1.2: Lab study (ratings + demographics)")

    lab_results_df = pd.read_excel(lab_results_file).set_index(c.PARTICIPANT_ID, drop=True)
    lab_results_df = utils.processing_utils.aggregate_by_characteristics(
        lab_results_df,
        cycling_frequency=True,
        cycling_confidence=True,
        cycling_purpose=True)

    demographics_df = lab_results_df[[col for col in c.DEMOGRAPHIC_COLUMNS if col != c.IS_SWISS]]

    lab_results_df = (
        lab_results_df
        .replace(r'\s*\((best|worst) experience\)', '', regex=True)
        .drop(columns=[col for col in c.DEMOGRAPHIC_COLUMNS if col != c.IS_SWISS] + [c.START, c.END])
        .apply(pd.to_numeric, errors='coerce')
    )

    log.info("Phase 1.3: Sequence, setup, and prediction files")

    lab_seq_df = pd.read_csv(lab_sequence_file)
    experiment_setup = pd.read_csv(lab_setup_file, header=None).set_index(0, drop=True)
    video_score_predictions = pd.read_csv(video_predictions_file)

    # ==============================================================================
    # STAGE 1: STIMULUS VALIDATION (Task 1 single-clip ratings)
    # ==============================================================================
    log.info("\n--- Stage 1: Stimulus validation ---")

    # Task 1 ratings -> long -> valence/arousal -> clip-level means
    df_baseline = utils.helper_functions.get_trial_dict(
        lab_results_df, experiment_setup, c.TRIAL_1, c.VIDEO_COUNTS
    )
    df1 = utils.helper_functions.trial_dict_to_df(df_baseline)
    df1 = utils.processing_utils.add_valence_arousal(df1, ag_col='rating')
    lab_video_scores = utils.processing_utils.calculate_video_level_scores(df1, lab_bool=True)

    # Lab-sample valence–arousal coupling: clip-level Pearson (matches the manuscript's
    # linear inverse coupling; computed before the merge while columns are plain).

    va_r = lab_video_scores[['valence', 'arousal']].corr(method='pearson').iloc[0, 1]
    log.info(f"Lab-sample valence–arousal coupling (clip-level Pearson): r = {va_r:.3f}")

    # Merge lab <- online <- k-NN prediction, keyed on clip id
    video_level_scores = (
        lab_video_scores
        .merge(online_video_level_scores, on=c.VIDEO_ID_COL, how='left')
        .merge(video_score_predictions, on=c.VIDEO_ID_COL, how='left')
    )
    video_level_scores.to_csv(output_dir / 'stage1_clip_validation.csv', index=False)

    # Agreement metrics (valence has a prediction; arousal does not)
    val_metrics = utils.helper_functions.get_video_level_metrics(
        video_level_scores, c.VALENCE, 'valence_prediction', 'valence_online'
    )
    aro_metrics = utils.helper_functions.get_video_level_metrics(
        video_level_scores, c.AROUSAL, None, 'arousal_online'
    )
    pd.DataFrame({'valence': val_metrics, 'arousal': aro_metrics}).to_csv(
        output_dir / 'stage1_agreement_metrics.csv'
    )

    # Bland–Altman absolute-agreement plots (lab vs online), both axes
    ba_val = utils.plotting_utils.plot_bland_altman(
        df=video_level_scores, measurement1='valence', measurement2='valence_online',
        label_col=c.VIDEO_ID_COL, save_path=output_dir / 'stage1_bland_altman_valence.png'
    )
    ba_aro = utils.plotting_utils.plot_bland_altman(
        df=video_level_scores, measurement1='arousal', measurement2='arousal_online',
        label_col=c.VIDEO_ID_COL, save_path=output_dir / 'stage1_bland_altman_arousal.png'
    )
    pd.DataFrame({'valence': ba_val, 'arousal': ba_aro}).to_csv(
        output_dir / 'stage1_bland_altman_stats.csv'
    )
    log.info(f"Bland–Altman valence: bias={ba_val['bias']:.3f}, "
             f"LoA=[{ba_val['lower_loa']:.3f}, {ba_val['upper_loa']:.3f}]")
    log.info(f"Bland–Altman arousal: bias={ba_aro['bias']:.3f}, "
             f"LoA=[{ba_aro['lower_loa']:.3f}, {ba_aro['upper_loa']:.3f}]")

    utils.plotting_utils.plot_clip_affect_space(
        video_level_scores,
        save_path=output_dir / 'stage1_clip_affect_space.png'
    )

    # PHYSIOLOGICAL DATA validation still comes here.
    # TODO

    # ==============================================================================
    # STAGE 2 (TASK 2): TWO-SEGMENT ORDER BLOCK - ranking + valence/arousal trends
    # ==============================================================================
    log.info("\n--- Task 2: two-segment order block ---")

    df_two = utils.helper_functions.load_and_process_trial_data(
        lab_results_df, experiment_setup, lab_seq_df, c.TRIAL_2_PARAMS, video_level_scores
    )
    df_two = pd.merge(df_two, demographics_df, on=c.PARTICIPANT_ID, how='left')

    # Four-level pair factor (shared with the later LMM and the plots)
    df_two['pair'] = df_two['sequence_list'].apply(lambda s: ' \u2192 '.join(map(str, s)))
    df_two['sequence_type'] = df_two['pair']

    # --- Descriptive trend (valence + arousal); arousal shows the sequence effect clearly
    utils.plotting_utils.plot_sequence_trend_panels(
        df_two, sequence_order=c.TRIAL_2_PLOT_ORDER, estimator="mean",
        save_path=output_dir / 'task2_trends_CI.png'
    )

    # --- Recalled ranking: distribution plot + order tests (Wilcoxon)
    utils.plotting_utils.plot_ranking_distribution(
        df_two, pair_col='pair', ranking_col='ranking',
        sequence_order=c.TRIAL_2_PLOT_ORDER,
        save_path=output_dir / 'task2_ranking_distribution.png'
    )

    # Order tests (NB→B vs B→NB) on ranking, valence, arousal
    w_rank = utils.helper_functions.wilcoxon_pair(
        df_two, c.PARTICIPANT_ID, 'pair', 'ranking', 'NB \u2192 B', 'B \u2192 NB'
    )
    w_val = utils.helper_functions.wilcoxon_pair(
        df_two, c.PARTICIPANT_ID, 'pair', 'valence', 'NB \u2192 B', 'B \u2192 NB'
    )
    w_aro = utils.helper_functions.wilcoxon_pair(
        df_two, c.PARTICIPANT_ID, 'pair', 'arousal', 'NB \u2192 B', 'B \u2192 NB'
    )
    for label, w in [("ranking", w_rank), ("valence", w_val), ("arousal", w_aro)]:
        log.info(f"Order ({label}): {w}")
    pd.DataFrame([w_val, w_aro, w_rank]).to_csv(output_dir / 'task2_order_wilcoxon.csv', index=False)

    # ==============================================================================
    # STAGE 2 (TASK 3): POSITIVE BLOCK — NB spoiler in a bikeable context
    # ==============================================================================
    log.info("\n--- Task 3: Positive block (NB spoiler) ---")

    df_positive = utils.helper_functions.load_and_process_trial_data(
        lab_results_df, experiment_setup, lab_seq_df,
        c.TRIAL_3_PARAMS, video_level_scores, 'NB'
    )
    df_positive = pd.merge(df_positive, demographics_df, on=c.PARTICIPANT_ID, how='left')
    df_positive['sequence_type'] = df_positive['sequence_list'].apply(lambda s: ' \u2192 '.join(map(str, s)))
    df_positive['spoiler_position'] = df_positive['NB_position']  # 0 = baseline, 1-3 = spoiler position

    utils.plotting_utils.plot_sequence_trend_panels(
        df_positive, sequence_order=c.TRIAL_3_PLOT_ORDER, estimator="mean",
        save_path=output_dir / 'task3_positive_trends_CI.png'
    )
    utils.plotting_utils.plot_ranking_distribution(
        df_positive, pair_col='sequence_type', ranking_col='ranking',
        sequence_order=c.TRIAL_3_PLOT_ORDER,
        save_path=output_dir / 'task3_positive_ranking_distribution.png'
    )

    # Position effect on recalled ranking (spoiler positions only, baseline excluded)
    fk_pos = utils.helper_functions.friedman_kendall(
        df_positive[df_positive['spoiler_position'] != 0],
        subject_col=c.PARTICIPANT_ID, condition_col='spoiler_position', value_col='ranking'
    )
    log.info(f"Positive ranking, position effect — Friedman chi2={fk_pos['chi2']:.2f}, "
             f"df={fk_pos['df']}, p={fk_pos['p']:.4g}, Kendall W={fk_pos['kendall_w']:.3f} "
             f"(n={fk_pos['n_subjects']}, k={fk_pos['n_conditions']})")
    pd.DataFrame([fk_pos]).to_csv(output_dir / 'task3_positive_ranking_friedman.csv', index=False)

    # ==============================================================================
    # STAGE 2 (TASK 3): NEGATIVE BLOCK — B spoiler in a non-bikeable context
    # ==============================================================================
    log.info("\n--- Task 3: Negative block (B spoiler) ---")

    df_negative = utils.helper_functions.load_and_process_trial_data(
        lab_results_df, experiment_setup, lab_seq_df,
        c.TRIAL_4_PARAMS, video_level_scores, 'B'
    )
    df_negative = pd.merge(df_negative, demographics_df, on=c.PARTICIPANT_ID, how='left')
    df_negative['sequence_type'] = df_negative['sequence_list'].apply(lambda s: ' \u2192 '.join(map(str, s)))
    df_negative['spoiler_position'] = df_negative['B_position']  # 0 = baseline, 1-3 = spoiler position

    utils.plotting_utils.plot_sequence_trend_panels(
        df_negative, sequence_order=c.TRIAL_4_PLOT_ORDER, estimator="mean",
        save_path=output_dir / 'task3_negative_trends_CI.png'
    )
    utils.plotting_utils.plot_ranking_distribution(
        df_negative, pair_col='sequence_type', ranking_col='ranking',
        sequence_order=c.TRIAL_4_PLOT_ORDER,
        save_path=output_dir / 'task3_negative_ranking_distribution.png'
    )

    fk_neg = utils.helper_functions.friedman_kendall(
        df_negative[df_negative['spoiler_position'] != 0],
        subject_col=c.PARTICIPANT_ID, condition_col='spoiler_position', value_col='ranking'
    )
    log.info(f"Negative ranking, position effect — Friedman chi2={fk_neg['chi2']:.2f}, "
             f"df={fk_neg['df']}, p={fk_neg['p']:.4g}, Kendall W={fk_neg['kendall_w']:.3f} "
             f"(n={fk_neg['n_subjects']}, k={fk_neg['n_conditions']})")
    pd.DataFrame([fk_neg]).to_csv(output_dir / 'task3_negative_ranking_friedman.csv', index=False)

    utils.plotting_utils.plot_ranking_distribution_combined(
        blocks=[
            (df_two, "Two-segment routes", c.TRIAL_2_PLOT_ORDER),
            (df_positive, "Three-segment: Positive", c.TRIAL_3_PLOT_ORDER),
            (df_negative, "Three-segment: Negative", c.TRIAL_4_PLOT_ORDER),
        ],
        save_path=output_dir / 'ranking_distribution_combined.png'
    )

    # ==========================================================================
    # LMM PHASE — Tasks 2 & 3
    #   Stage 2 : within-block position effects        (RQ1)
    #   Stage 3 : pooled position × scenario model     (RQ1/RQ2/RQ3 inference)
    #   Stage 4 : aggregation-rule comparison          (RQ3 mechanism)
    #   Stage 4b: leave-one-participant-out CV         (RQ3 mechanism, generalization)
    #   Stage 5 : covariate robustness (exploratory)
    # All models: by-participant random intercept (ML, for valid AIC/BIC/LR).
    # A crossed clip random effect was tested (diagnose_clip_random_effect) and
    # estimated at the variance boundary (~0); measured per-segment values already
    # absorb clip-level variability, so it is omitted here.
    # ==========================================================================
    log.info("\n--- LMM phase: Tasks 2 & 3 ---")

    df_combined = utils.processing_utils.prepare_combined_scenario_df(df_positive, df_negative)

    # --- One-time data integrity checks (not part of the analysis output) ---
    for o in ["valence", "arousal"]:
        seg = df_combined[[f"pos1_{o}", f"pos2_{o}", f"pos3_{o}"]].to_numpy()
        assert np.allclose(df_combined[f"mean_{o}"], seg.mean(axis=1), atol=1e-2, equal_nan=True)
        assert np.allclose(df_combined[f"pos_peak_{o}"], seg.max(axis=1), atol=1e-2)
        assert np.allclose(df_combined[f"neg_peak_{o}"], seg.min(axis=1), atol=1e-2)
        assert np.allclose(df_combined[f"end_{o}"], df_combined[f"pos3_{o}"], atol=1e-2)
    log.info(f"Feature construction verified. N obs = {len(df_combined)}")

    # --- Contrast / EMM specifications (shared across outcomes) ---
    position = "C(spoiler_position)"
    scenario = "C(scenario)[T.Positive]"

    rq2_magnitude_spec = {
        f"{position}[T.1]": -2 / 3, f"{position}[T.2]": -2 / 3, f"{position}[T.3]": -2 / 3,
        f"{position}[T.1]:{scenario}": -1 / 3,
        f"{position}[T.2]:{scenario}": -1 / 3,
        f"{position}[T.3]:{scenario}": -1 / 3,
    }

    contrast_specs = {
        "neg_p1_vs_base": {f"{position}[T.1]": 1},
        "neg_p2_vs_base": {f"{position}[T.2]": 1},
        "neg_p3_vs_base": {f"{position}[T.3]": 1},
        "neg_p3_vs_p1": {f"{position}[T.3]": 1, f"{position}[T.1]": -1},
        "pos_p1_vs_base": {f"{position}[T.1]": 1, f"{position}[T.1]:{scenario}": 1},
        "pos_p2_vs_base": {f"{position}[T.2]": 1, f"{position}[T.2]:{scenario}": 1},
        "pos_p3_vs_base": {f"{position}[T.3]": 1, f"{position}[T.3]:{scenario}": 1},
        "pos_p3_vs_p1": {f"{position}[T.3]": 1, f"{position}[T.1]": -1,
                         f"{position}[T.3]:{scenario}": 1, f"{position}[T.1]:{scenario}": -1},
        # RQ2 negativity bias: |NB-spoiler shift| - |B-spoiler shift| (positive => negativity bias)
        "rq2_magnitude": {f"{position}[T.1]": -2 / 3, f"{position}[T.2]": -2 / 3, f"{position}[T.3]": -2 / 3,
                          f"{position}[T.1]:{scenario}": -1 / 3,
                          f"{position}[T.2]:{scenario}": -1 / 3,
                          f"{position}[T.3]:{scenario}": -1 / 3},
    }
    PLANNED_FAMILIES = {"neg_p3_vs_p1", "pos_p3_vs_p1", "rq2_magnitude"}
    emm_specs = {
        ("Negative", 0): {"Intercept": 1},
        ("Negative", 1): {"Intercept": 1, f"{position}[T.1]": 1},
        ("Negative", 2): {"Intercept": 1, f"{position}[T.2]": 1},
        ("Negative", 3): {"Intercept": 1, f"{position}[T.3]": 1},
        ("Positive", 0): {"Intercept": 1, scenario: 1},
        ("Positive", 1): {"Intercept": 1, f"{position}[T.1]": 1, scenario: 1, f"{position}[T.1]:{scenario}": 1},
        ("Positive", 2): {"Intercept": 1, f"{position}[T.2]": 1, scenario: 1, f"{position}[T.2]:{scenario}": 1},
        ("Positive", 3): {"Intercept": 1, f"{position}[T.3]": 1, scenario: 1, f"{position}[T.3]:{scenario}": 1},
    }

    covariates = ["Gender", "Age", "Cycling_confidence",
                  "Cycling_frequency", "Cycling_purpose", "Cycling_environment"]

    all_contrasts = []

    for OUTCOME in ["valence", "arousal"]:
        log.info(f"\n{'=' * 50}\n{OUTCOME.upper()}\n{'=' * 50}")
        peak, neg, end = f"pos_peak_{OUTCOME}", f"neg_peak_{OUTCOME}", f"end_{OUTCOME}"

        raw = df_combined.apply(
            lambda r: r[f"pos{int(r['spoiler_position'])}_{OUTCOME}"]
            if r['spoiler_position'] > 0 else np.nan, axis=1)
        df_combined[f"spoiler_{OUTCOME}_cb"] = (
                raw - raw.groupby(df_combined["scenario"]).transform("mean")
        ).fillna(0.0)

        # ==================================================================
        # STAGE 2 — Task 2 two-segment order effect (RQ1)
        # ==================================================================
        m_task2 = utils.lmm_utils.run_lmm(
            df=df_two, formula=f"{OUTCOME} ~ C(pair)",
            groups_col=c.PARTICIPANT_ID, convergence_method='powell')
        order_ct = utils.lmm_utils.lmm_contrast(
            m_task2, {"C(pair)[T.NB → B]": 1, "C(pair)[T.B → NB]": -1})
        all_contrasts.append({
            "task": "Task 2 (order)", "outcome": OUTCOME,
            "contrast": "NB_to_B_vs_B_to_NB", "family": "planned",
            "p_holm": pd.NA, **order_ct})

        # ==================================================================
        # STAGE 3 — Task 3 pooled position × scenario (RQ1/RQ2/RQ3)
        # PRIMARY model: no spoiler covariate. All reported contrasts/EMMs
        # come from m_inter. One optimizer ('powell') for all LR pairs.
        # ==================================================================
        # PRIMARY models (both include the covariate -> valid LR test)
        m_main = utils.lmm_utils.run_lmm(
            df=df_combined,
            formula=f"{OUTCOME} ~ C(spoiler_position) + C(scenario) + spoiler_{OUTCOME}_cb",
            groups_col=c.PARTICIPANT_ID, convergence_method='powell')
        m_inter = utils.lmm_utils.run_lmm(
            df=df_combined,
            formula=f"{OUTCOME} ~ C(spoiler_position) * C(scenario) + spoiler_{OUTCOME}_cb",
            groups_col=c.PARTICIPANT_ID, convergence_method='powell')

        rq3 = utils.helper_functions.lr_test(m_main, m_inter, label=f"RQ3 omnibus ({OUTCOME})")
        log.info(f"RQ3 omnibus (main vs interaction): {rq3}")

        # Sensitivity now runs the OTHER way: does dropping the covariate change anything?
        m_inter_nocov = utils.lmm_utils.run_lmm(
            df=df_combined,
            formula=f"{OUTCOME} ~ C(spoiler_position) * C(scenario)",
            groups_col=c.PARTICIPANT_ID, convergence_method='powell')
        keep = [p for p in m_inter.params.index if "spoiler_position" in p or "scenario" in p]
        pd.DataFrame({
            "primary (covariate-adjusted)": m_inter.params[keep],
            "unadjusted": m_inter_nocov.params.reindex(keep),
        }).to_csv(output_dir / f"spoiler_covariate_sensitivity_{OUTCOME}.csv")

        # Planned + descriptive contrasts (Holm within planned family), all on m_inter
        planned_pvals, planned_idx = [], []
        for name, spec in contrast_specs.items():
            ct = utils.lmm_utils.lmm_contrast(m_inter, spec)
            is_planned = name in PLANNED_FAMILIES
            all_contrasts.append({
                "task": "Task 3", "outcome": OUTCOME, "contrast": name,
                "family": "planned" if is_planned else "descriptive",
                "p_holm": pd.NA, **ct})
            if is_planned:
                planned_pvals.append(ct["p"])
                planned_idx.append(len(all_contrasts) - 1)
        if planned_pvals:
            for i, ph in zip(planned_idx, multipletests(planned_pvals, method='holm')[1]):
                all_contrasts[i]["p_holm"] = ph

        # EMMs for the interaction plot
        emm_df = pd.DataFrame([
            {"outcome": OUTCOME, "block": block, "position": pos,
             "emm": (ct := utils.lmm_utils.lmm_contrast(m_inter, spec))["estimate"],
             "ci_low": ct["ci_low"], "ci_high": ct["ci_high"]}
            for (block, pos), spec in emm_specs.items()])
        emm_df.to_csv(output_dir / f"EMM_table_{OUTCOME}.csv", index=False)
        utils.plotting_utils.plot_emm_interaction(emm_df, OUTCOME, output_dir, log)

        # ==================================================================
        # STAGE 4 — Aggregation rules: AIC/BIC + recency tests
        # ==================================================================
        m_additive = utils.lmm_utils.run_lmm(
            df=df_combined, formula=f"{OUTCOME} ~ mean_{OUTCOME}",
            groups_col=c.PARTICIPANT_ID, convergence_method='powell')
        m_seq = utils.lmm_utils.run_lmm(
            df=df_combined, formula=f"{OUTCOME} ~ pos1_{OUTCOME} + pos2_{OUTCOME} + pos3_{OUTCOME}",
            groups_col=c.PARTICIPANT_ID, convergence_method='powell')
        m_peak_end = utils.lmm_utils.run_lmm(
            df=df_combined, formula=f"{OUTCOME} ~ {peak} + {neg} + {end}",
            groups_col=c.PARTICIPANT_ID, convergence_method='powell')
        m_min_end = utils.lmm_utils.run_lmm(
            df=df_combined, formula=f"{OUTCOME} ~ {neg} + {end}",
            groups_col=c.PARTICIPANT_ID, convergence_method='powell')

        pd.DataFrame({
            "Outcome": OUTCOME,
            "Model": ["Additive (mean-value)", "Sequential (positional-value)",
                      "Peak-End (Sym)", "Minimum-End"],
            "AIC": [m_additive.aic, m_seq.aic, m_peak_end.aic, m_min_end.aic],
            "BIC": [m_additive.bic, m_seq.bic, m_peak_end.bic, m_min_end.bic],
        }).sort_values("AIC").to_csv(output_dir / f'stage4_model_comparison_{OUTCOME}.csv', index=False)

        # Recency: Additive is m_seq with b1=b2=b3 (mean = sum/3, same fit) -> direct LR test,
        # plus pairwise weight contrasts. This is THE test of the recency claim.
        recency_out = pd.DataFrame([
            utils.helper_functions.lr_test(m_additive, m_seq, label="equal-weights vs sequential"),
            utils.helper_functions.wald_contrast(m_seq, {f"pos3_{OUTCOME}": 1, f"pos1_{OUTCOME}": -1}, "b3 - b1"),
            utils.helper_functions.wald_contrast(m_seq, {f"pos3_{OUTCOME}": 1, f"pos2_{OUTCOME}": -1}, "b3 - b2"),
            utils.helper_functions.wald_contrast(m_seq, {f"pos2_{OUTCOME}": 1, f"pos1_{OUTCOME}": -1}, "b2 - b1"),
        ])
        recency_out.to_csv(output_dir / f"recency_tests_{OUTCOME}.csv", index=False)
        log.info(f"\nRecency tests ({OUTCOME}):\n{recency_out.to_string(index=False)}")

        # ==================================================================
        # STAGE 4b — LOPO CV with paired comparison
        # ==================================================================
        cv_formulas = {
            "Additive (mean-value)": f"{OUTCOME} ~ mean_{OUTCOME}",
            "Sequential (positional-value)": f"{OUTCOME} ~ pos1_{OUTCOME} + pos2_{OUTCOME} + pos3_{OUTCOME}",
            "Peak-End (Sym)": f"{OUTCOME} ~ {peak} + {neg} + {end}",
            "Minimum-End": f"{OUTCOME} ~ {neg} + {end}",
        }
        cv_fold_rmse = {name: {} for name in cv_formulas}
        for held_out in df_combined[c.PARTICIPANT_ID].unique():
            train = df_combined[df_combined[c.PARTICIPANT_ID] != held_out]
            test = df_combined[df_combined[c.PARTICIPANT_ID] == held_out]
            for name, formula in cv_formulas.items():
                try:
                    m_cv = utils.lmm_utils.run_lmm(df=train, formula=formula,
                                                   groups_col=c.PARTICIPANT_ID,
                                                   convergence_method='powell')
                    resid = test[OUTCOME].to_numpy() - np.asarray(m_cv.predict(exog=test))
                    cv_fold_rmse[name][held_out] = float(np.sqrt(np.mean(resid ** 2)))
                except Exception as e:
                    log.warning(f"CV fold (p={held_out}, {name}) failed: {e}")

        fold_df = pd.DataFrame(cv_fold_rmse).dropna()
        summary = fold_df.mean().sort_values().rename("mean_fold_RMSE").to_frame()
        best = summary.index[0]
        rows = []
        for other in [m for m in fold_df.columns if m != best]:
            diff = fold_df[other] - fold_df[best]
            W, p = stats.wilcoxon(diff)
            rows.append({"best": best, "vs": other,
                         "median_RMSE_diff": float(diff.median()),
                         "W": float(W), "p_raw": float(p)})
        pairs = pd.DataFrame(rows)
        pairs["p_holm"] = multipletests(pairs["p_raw"], method="holm")[1]
        log.info(f"\nStage 4b paired CV ({OUTCOME}):\n{summary.to_string()}\n{pairs.to_string(index=False)}")
        summary.to_csv(output_dir / f"stage4b_cv_summary_{OUTCOME}.csv")
        pairs.to_csv(output_dir / f"stage4b_cv_paired_tests_{OUTCOME}.csv", index=False)

        # ==================================================================
        # STAGE 5 — Covariate robustness (LR vs the PRIMARY model)
        # ==================================================================
        base_formula = f"{OUTCOME} ~ C(spoiler_position) * C(scenario) + spoiler_{OUTCOME}_cb"
        cov_rows = []
        for cov in covariates:
            if cov not in df_combined.columns:
                continue
            sub = df_combined.dropna(subset=[cov])
            if sub[cov].nunique() < 2:
                log.warning(f"{cov}: <2 levels — skipping.")
                continue
            m_ref = (m_inter if len(sub) == len(df_combined)
                     else utils.lmm_utils.run_lmm(df=sub, formula=base_formula,
                                                  groups_col=c.PARTICIPANT_ID,
                                                  convergence_method='powell'))
            m_cov = utils.lmm_utils.run_lmm(
                df=sub, formula=f"{base_formula} + C({cov})",
                groups_col=c.PARTICIPANT_ID, convergence_method='powell')
            res = utils.helper_functions.lr_test(m_ref, m_cov, label=cov)
            if res["note"]:
                log.error(f"Stage 5 ({OUTCOME}, {cov}): {res['note']}")
            cov_rows.append({"outcome": OUTCOME, "covariate": cov, **res})
        if cov_rows:
            cov_df = pd.DataFrame(cov_rows)
            valid = cov_df["p"].notna()
            cov_df.loc[valid, "p_holm"] = multipletests(cov_df.loc[valid, "p"], method='holm')[1]
            log.info(f"\nStage 5 covariates ({OUTCOME}):\n{cov_df.to_string(index=False)}")
            cov_df.to_csv(output_dir / f'stage5_covariates_{OUTCOME}.csv', index=False)

if __name__ == "__main__":
    main()
