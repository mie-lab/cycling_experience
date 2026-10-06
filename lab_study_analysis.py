import configparser
import utils.helper_utils
import utils.lmm_utils
import utils.plot_utils
import utils.process_utils
import utils.physio_utils
import logging
import constants as c
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path
from statsmodels.stats.multitest import multipletests

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
    physio_file = Path(config["filenames"]["physiological_results_file"])

    # Define and create the output directory.
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    OPT = "powell"
    PHYSIO_EXCLUDE = {0, 3, 12, 24}  # recording failure (0, 3, 24), medical reason (12)
    # Raw levels: participant random intercepts absorb individual baselines
    PHYSIO_METRICS = ["SCR_Peaks_Amplitude_Mean", "SCL_Mean", "PPG_Rate_Mean", "HRV_RMSSD"]
    SCR = "SCR_Peaks_Amplitude_Mean"  # skewed: log1p-transformed for the analyses

    # ==============================================================================
    # PHASE 1: LOAD DATA
    # ==============================================================================

    log.info("Phase 1.1: Load online survey (validation reference)")

    survey_df = pd.read_excel(online_results_file).set_index(c.PARTICIPANT_ID)
    online_seq_df = pd.read_csv(online_sequence_file, parse_dates=['seq_start', 'seq_end'])

    survey_results_df = utils.process_utils.transform_to_long_df(survey_df, online_seq_df, id_col=c.PARTICIPANT_ID)
    survey_results_df = utils.process_utils.filter_results(survey_results_df)
    survey_results_df = utils.process_utils.add_valence_arousal(survey_results_df)
    survey_results_df = utils.process_utils.agg_by_chars(
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

    online_video_level_scores = utils.process_utils.calc_video_level_scores(survey_results_df, lab_bool=True)
    online_video_level_scores = online_video_level_scores.rename(
        columns={
            c.VALENCE: 'valence_online',
            c.AROUSAL: 'arousal_online'
        }
    )
    n_online_ratings = survey_results_df.groupby(c.VIDEO_ID_COL).size()
    log.info(f"Online survey: {survey_results_df[c.PARTICIPANT_ID].nunique()} respondents, "
             f"{len(online_video_level_scores)} clips, "
             f"{n_online_ratings.min()}-{n_online_ratings.max()} ratings per clip")

    log.info("Phase 1.2: Load lab study (ratings + demographics)")

    lab_results_df = pd.read_excel(lab_results_file).set_index(c.PARTICIPANT_ID, drop=True)
    lab_results_df = utils.process_utils.agg_by_chars(
        lab_results_df,
        cycling_frequency=True,
        cycling_confidence=True,
        cycling_purpose=True)

    demo_cols = [col for col in c.DEMOGRAPHIC_COLUMNS if col != c.IS_SWISS]
    demographics_df = lab_results_df[demo_cols]

    lab_results_df = (
        lab_results_df
        .replace(r'\s*\((best|worst) experience\)', '', regex=True)
        .drop(columns=demo_cols + [c.START, c.END])
        .apply(pd.to_numeric, errors='coerce')
    )
    log.info(f"Lab study: {len(lab_results_df)} participants, {lab_results_df.shape[1]} rating/ranking columns, "
             f"{int(lab_results_df.isna().sum().sum())} missing or non-numeric values")

    for col in demo_cols:
        shares = demographics_df[col].value_counts(normalize=True, dropna=False).mul(100).round(1)
        log.info(f"  {col} (%): {shares.to_dict()}")

    log.info("Phase 1.3: Load sequence, setup, prediction, and physiology files")

    lab_seq_df = pd.read_csv(lab_sequence_file)
    experiment_setup = pd.read_csv(lab_setup_file, header=None).set_index(0, drop=True)
    video_score_predictions = pd.read_csv(video_predictions_file)

    df_physio = pd.read_csv(physio_file)

    # Physio segment k is mapped to presentation slot k: flag segments whose duration does not fit
    misaligned = utils.physio_utils.check_segment_alignment(df_physio, experiment_setup)
    if not misaligned.empty:
        log.warning(f"Physio segments not matching their slot:\n{misaligned.to_string(index=False)}")

    # Drop excluded participants and misaligned segments; log-transform the skewed SCR
    misaligned_ids = set(zip(misaligned[c.PARTICIPANT_ID], misaligned['slot']))
    keep = [
        pid not in PHYSIO_EXCLUDE and (pid, slot) not in misaligned_ids
        for pid, slot in zip(df_physio[c.PARTICIPANT_ID], df_physio['segment_id'])
    ]
    df_physio = df_physio[keep].copy()
    df_physio[SCR] = np.log1p(df_physio[SCR].clip(lower=0))

    phys3 = utils.physio_utils.get_physio_trial_df(df_physio, experiment_setup,
                                                   c.TRIAL_3_PARAMS["trial_label"], PHYSIO_METRICS)
    phys4 = utils.physio_utils.get_physio_trial_df(df_physio, experiment_setup,
                                                   c.TRIAL_4_PARAMS["trial_label"], PHYSIO_METRICS)
    for label, d in (("bikeable block", phys3), ("non-bikeable block", phys4)):
        log.info(f"Physio {label}: {len(d)} sequences from {d[c.PARTICIPANT_ID].nunique()} participants")

    # ==============================================================================
    # PHASE 2: STIMULUS VALIDATION (Task 1 single-clip ratings)
    # ==============================================================================
    log.info("Phase 2: Stimulus validation")

    task1 = utils.helper_utils.get_trial_dict(
        lab_results_df, experiment_setup, c.TRIAL_1, c.VIDEO_COUNTS
    )
    df1 = utils.helper_utils.trial_dict_to_df(task1)
    df1 = utils.process_utils.add_valence_arousal(df1, ag_col='rating')

    clip_types = utils.helper_utils.get_clip_types(lab_seq_df)
    validated_ids = survey_results_df[c.VIDEO_ID_COL].unique()
    df1['clip_type'] = df1[c.VIDEO_ID_COL].map(clip_types)
    df1['extension'] = (~df1[c.VIDEO_ID_COL].isin(validated_ids)).astype(int)
    df1['group'] = 1

    # NB - B per clip set
    random_effects = {
        'participant': f'0 + C({c.PARTICIPANT_ID})',
        'clip': f'0 + C({c.VIDEO_ID_COL})'
    }
    nb = 'C(clip_type)[T.NB]'
    nb_x_extension = 'C(clip_type)[T.NB]:extension'
    contrasts = {
        'validated': {nb: 1},
        'extension': {nb: 1, nb_x_extension: 1},
        'difference': {nb_x_extension: 1},
    }
    manipulation_check = []
    for outcome in [c.VALENCE, c.AROUSAL]:
        model = utils.lmm_utils.run_lmm(
            df=df1, formula=f"{outcome} ~ C(clip_type) * extension",
            groups_col='group', vc_formula=random_effects,
            convergence_method=OPT, verbose=False
        )
        for clip_set, weights in contrasts.items():
            result = utils.lmm_utils.lmm_contrast(model, weights)
            manipulation_check.append({'outcome': outcome, 'clip_set': clip_set, **result})
    manipulation_check = pd.DataFrame(manipulation_check)
    manipulation_check.to_csv(output_dir / 'task1_manipulation_check.csv', index=False)
    log.info(f"Manipulation check (NB - B):\n{manipulation_check.round(3)}")

    # Clip-level means: lab, online survey, k-NN prediction
    video_level_scores = (
        utils.process_utils.calc_video_level_scores(df1, lab_bool=True)
        .merge(online_video_level_scores, on=c.VIDEO_ID_COL, how='left')
        .merge(video_score_predictions, on=c.VIDEO_ID_COL, how='left')
    )
    video_level_scores.to_csv(output_dir / 'task1_clip_validation.csv', index=False)

    # Valence-arousal coupling on the validated clips
    validated = video_level_scores.dropna(subset=['valence_online'])
    lab_r = validated['valence'].corr(validated['arousal'])
    online_r = validated['valence_online'].corr(validated['arousal_online'])
    log.info(f"Valence-arousal r: lab {lab_r:.2f}, online {online_r:.2f}")

    # Agreement: lab vs online and lab vs k-NN; Bland-Altman on validated clips only
    agreement = {}
    for outcome in [c.VALENCE, c.AROUSAL]:
        online_col = f'{outcome}_online'
        metrics = utils.helper_utils.get_video_level_metrics(
            video_level_scores, outcome, f'{outcome}_prediction', online_col
        )
        bland_altman = utils.plot_utils.plot_bland_altman(
            df=video_level_scores, measurement1=outcome, measurement2=online_col,
            label_col=c.VIDEO_ID_COL,
            save_path=output_dir / f'task1_bland_altman_{outcome}.png'
        )
        agreement[outcome] = {**metrics, **bland_altman}
    agreement = pd.DataFrame(agreement)
    agreement.to_csv(output_dir / 'task1_agreement_metrics.csv')
    log.info(f"Agreement:\n{agreement.round(3)}")

    # Physiology per clip (exploratory)
    phys1 = utils.physio_utils.get_physio_trial_df(df_physio, experiment_setup, c.TRIAL_1, PHYSIO_METRICS)
    phys1[c.VIDEO_ID_COL] = phys1['sequence_code'].str.extract(r'video_(\d+)', expand=False).astype(int)
    df1_physio = df1.merge(
        phys1.drop(columns='sequence_code'), on=[c.PARTICIPANT_ID, c.VIDEO_ID_COL]
    )
    log.info(f"Physio clips: {len(df1_physio)} from {df1_physio[c.PARTICIPANT_ID].nunique()} participants")

    # NB - B per metric (same LMM as the ratings) and repeated-measures correlations;
    # segment_id (presentation slot) absorbs the drift of the signals over the session
    physio_check = []
    for metric in PHYSIO_METRICS:
        model = utils.lmm_utils.run_lmm(
            df=df1_physio.dropna(subset=[metric]), formula=f"{metric} ~ C(clip_type) + segment_id",
            groups_col='group', vc_formula=random_effects,
            convergence_method=OPT, verbose=False
        )
        physio_check.append({'metric': metric, **utils.lmm_utils.lmm_contrast(model, {nb: 1})})
    physio_check = pd.DataFrame(physio_check)
    physio_check['p_holm'] = multipletests(physio_check['p'], method='holm')[1]
    physio_check.to_csv(output_dir / 'task1_physio_manipulation_check.csv', index=False)
    log.info(f"Physio NB - B:\n{physio_check.round(3).to_string(index=False)}")

    physio_corr = utils.helper_utils.rm_corr_table(
        df1_physio, c.PARTICIPANT_ID, PHYSIO_METRICS, [c.VALENCE, c.AROUSAL], control='segment_id'
    )
    physio_corr.to_csv(output_dir / 'task1_physio_rating_correlations.csv', index=False)
    log.info(f"Physio-rating correlations:\n{physio_corr.round(3).to_string(index=False)}")

    # ==============================================================================
    # PHASE 3: TWO-SEGMENT ORDER (Task 2)
    # ==============================================================================
    log.info("Phase 3: Two-segment order")

    df_two = utils.helper_utils.load_and_process_trial_data(
        lab_results_df, experiment_setup, lab_seq_df, c.TRIAL_2_PARAMS, video_level_scores
    )
    df_two['sequence_type'] = df_two['sequence_list'].str.join(' → ')

    utils.plot_utils.plot_sequence_trend_panels(
        df_two, sequence_order=c.TRIAL_2_PLOT_ORDER, estimator='mean',
        save_path=output_dir / 'task2_trends_CI.png'
    )

    # M1: B → NB - NB → B, with participant random intercepts
    order_weights = {
        'C(sequence_type)[T.B → NB]': 1,
        'C(sequence_type)[T.NB → B]': -1
    }
    m1 = []
    for outcome in [c.VALENCE, c.AROUSAL]:
        model = utils.lmm_utils.run_lmm(
            df=df_two, formula=f"{outcome} ~ C(sequence_type)",
            groups_col=c.PARTICIPANT_ID,
            convergence_method=OPT, verbose=False
        )
        result = utils.lmm_utils.lmm_contrast(model, order_weights)
        m1.append({'outcome': outcome, **result})
    m1 = pd.DataFrame(m1)
    m1.to_csv(output_dir / 'task2_M1_participant_intercept.csv', index=False)
    log.info(f"M1 (B → NB - NB → B):\n{m1.round(3)}")

    # Planned contrasts for Table 3, extended with Task 3
    all_contrasts = m1.assign(
        task='Task 2', contrast='B_to_NB_vs_NB_to_B', p_holm=np.nan
    ).to_dict('records')

    # Delayed rankings: B → NB - NB → B (positive = B → NB ranked worse)
    ranking = utils.helper_utils.wilcoxon_pair(
        df_two, c.PARTICIPANT_ID, 'sequence_type', 'ranking', 'B → NB', 'NB → B'
    )
    ranking = pd.DataFrame([ranking])
    ranking.to_csv(output_dir / 'task2_ranking_wilcoxon.csv', index=False)
    log.info(f"Ranking (Wilcoxon):\n{ranking.round(3).to_string(index=False)}")

    # ==============================================================================
    # PHASE 4: THREE-SEGMENT SEQUENCES (Task 3)
    # ==============================================================================
    log.info("Phase 4: Three-segment sequences")

    # Positive block: NB off-type segment in a bikeable route

    df_positive = utils.helper_utils.load_and_process_trial_data(
        lab_results_df, experiment_setup, lab_seq_df,
        c.TRIAL_3_PARAMS, video_level_scores, 'NB'
    )
    df_positive['sequence_type'] = df_positive['sequence_list'].str.join(' \u2192 ')
    df_positive = df_positive.merge(
        phys3, left_on=[c.PARTICIPANT_ID, c.VIDEO_ID_COL],
        right_on=[c.PARTICIPANT_ID, 'sequence_code'], how='left'
    )

    utils.plot_utils.plot_sequence_trend_panels(
        df_positive, sequence_order=c.TRIAL_3_PLOT_ORDER, estimator='mean',
        save_path=output_dir / 'task3_positive_trends_CI.png'
    )

    # Page's trend test on rankings: recency predicts a late NB segment ranked worst
    page_positive = utils.helper_utils.page_trend_ranking(
        df_positive[df_positive['off_type_position'] != 0],
        subject_col=c.PARTICIPANT_ID, condition_col='off_type_position',
        value_col='ranking', predicted_order=[1, 2, 3]
    )
    page_positive = pd.DataFrame([page_positive])
    page_positive.to_csv(output_dir / 'task3_positive_ranking_page.csv', index=False)
    log.info(f"Page's trend test, Positive block:\n{page_positive.round(3).to_string(index=False)}")

    # Negative block: B off-type segment in a non-bikeable route

    df_negative = utils.helper_utils.load_and_process_trial_data(
        lab_results_df, experiment_setup, lab_seq_df,
        c.TRIAL_4_PARAMS, video_level_scores, 'B'
    )
    df_negative['sequence_type'] = df_negative['sequence_list'].str.join(' \u2192 ')
    df_negative = df_negative.merge(
        phys4, left_on=[c.PARTICIPANT_ID, c.VIDEO_ID_COL],
        right_on=[c.PARTICIPANT_ID, 'sequence_code'], how='left'
    )

    utils.plot_utils.plot_sequence_trend_panels(
        df_negative, sequence_order=c.TRIAL_4_PLOT_ORDER, estimator='mean',
        save_path=output_dir / 'task3_negative_trends_CI.png'
    )

    # Page's trend test on rankings: recency predicts a late B segment ranked best
    page_negative = utils.helper_utils.page_trend_ranking(
        df_negative[df_negative['off_type_position'] != 0],
        subject_col=c.PARTICIPANT_ID, condition_col='off_type_position',
        value_col='ranking', predicted_order=[3, 2, 1]
    )
    page_negative = pd.DataFrame([page_negative])
    page_negative.to_csv(output_dir / 'task3_negative_ranking_page.csv', index=False)
    log.info(f"Page's trend test, Negative block:\n{page_negative.round(3).to_string(index=False)}")

    # Pooled position model (M2): both blocks together
    df_combined = utils.process_utils.prepare_combined_scenario_df(df_positive, df_negative)
    df_combined = df_combined.merge(demographics_df, on=c.PARTICIPANT_ID, how='left')
    blocks = ('Negative', 'Positive')
    demographic_tests = []

    for OUTCOME in ["valence", "arousal"]:
        log.info(f"\n{'=' * 50}\n{OUTCOME.upper()}\n{'=' * 50}")
        covariate = utils.process_utils.add_off_type_covariate(df_combined, OUTCOME)

        # M2: Y ~ position x scenario + off-type intensity + (1 | participant)
        m2_formula = f"{OUTCOME} ~ C(off_type_position) * C(scenario) + {covariate}"
        m2 = utils.lmm_utils.run_lmm(
            df=df_combined, formula=m2_formula,
            groups_col=c.PARTICIPANT_ID, convergence_method=OPT, verbose=False
        )
        utils.plot_utils.plot_lmm_diagnostics(
            m2, f"M2 ({OUTCOME})", output_dir / f"diagnostics_M2_{OUTCOME}.png"
        )

        # Contrasts as differences between M2's predicted cell means
        fixed_effects = m2.fe_params.index
        cells = {
            (b, k): utils.lmm_utils.cell_weights(fixed_effects, b, k)
            for b in blocks for k in range(4)
        }
        shift = {(b, k): cells[b, k] - cells[b, 0] for b in blocks for k in (1, 2, 3)}
        nb_sign = -1 if OUTCOME == c.VALENCE else 1  # NB lowers valence, raises arousal

        # Off-type shift from baseline, averaged over positions 1-3 (one Holm family)
        shifts = {
            'nb_shift': sum(shift['Positive', k] for k in (1, 2, 3)) / 3,
            'b_shift': sum(shift['Negative', k] for k in (1, 2, 3)) / 3,
        }
        # RQ1: late vs early off-type segment per block (one Holm family)
        recency = {
            'neg_p3_vs_p1': cells['Negative', 3] - cells['Negative', 1],
            'pos_p3_vs_p1': cells['Positive', 3] - cells['Positive', 1],
        }
        # RQ2: |NB shift| - |B shift|; > 0 = negativity bias
        magnitude = {'rq2_magnitude': nb_sign * (shifts['nb_shift'] + shifts['b_shift'])}
        contrasts = pd.concat([
            utils.lmm_utils.contrast_table(m2, shifts, holm=True),
            utils.lmm_utils.contrast_table(m2, recency, holm=True),
            utils.lmm_utils.contrast_table(m2, magnitude),
        ]).rename_axis('contrast').reset_index()
        all_contrasts += contrasts.assign(task='Task 3', outcome=OUTCOME).to_dict('records')

        # Predicted means per block x position (Figure 5)
        emm_df = utils.lmm_utils.contrast_table(m2, cells).rename(columns={'estimate': 'emm'})
        emm_df = emm_df.rename_axis(['block', 'position']).reset_index().assign(outcome=OUTCOME)
        emm_df.to_csv(output_dir / f"EMM_table_{OUTCOME}.csv", index=False)
        utils.plot_utils.plot_emm_interaction(emm_df, OUTCOME, output_dir, log)

        # Robustness: does any demographic variable improve M2? (LR test)
        for variable in demo_cols:
            m2_variable = utils.lmm_utils.run_lmm(
                df=df_combined, formula=f"{m2_formula} + C({variable})",
                groups_col=c.PARTICIPANT_ID, convergence_method=OPT, verbose=False
            )
            result = utils.helper_utils.lr_test(m2, m2_variable)
            demographic_tests.append({'outcome': OUTCOME, 'variable': variable, **result})

    demographic_tests = pd.DataFrame(demographic_tests).drop(columns=['label', 'note'])
    demographic_tests['p_holm'] = demographic_tests.groupby('outcome')['p'].transform(
        lambda p: multipletests(p, method='holm')[1]
    )
    demographic_tests.to_csv(output_dir / 'task3_M2_demographic_covariates.csv', index=False)
    log.info(f"M2 + demographic variable (LR test):\n{demographic_tests.round(3)}")

    # Planned contrasts (Table 3): Holm p where a family exists, else raw p
    all_contrasts = pd.DataFrame(all_contrasts)
    all_contrasts['p_report'] = all_contrasts['p_holm'].fillna(all_contrasts['p'])
    all_contrasts.to_csv(output_dir / 'planned_contrasts.csv', index=False)
    columns = ['task', 'contrast', 'outcome', 'estimate', 'ci_low', 'ci_high', 'p_report']
    log.info(f"Planned contrasts:\n{all_contrasts[columns].round(3).to_string(index=False)}")

    # Physiology (exploratory): non-bikeable - bikeable block per metric (Holm) and correlations
    # with the ratings; segment_id (presentation slot) absorbs the drift over the session
    physio_block = []
    for metric in PHYSIO_METRICS:
        model = utils.lmm_utils.run_lmm(
            df=df_combined.dropna(subset=[metric]), formula=f"{metric} ~ C(scenario) + segment_id",
            groups_col=c.PARTICIPANT_ID, convergence_method=OPT, verbose=False
        )
        result = utils.lmm_utils.lmm_contrast(model, {'C(scenario)[T.Positive]': -1})
        physio_block.append({'metric': metric, **result})
    physio_block = pd.DataFrame(physio_block)
    physio_block['p_holm'] = multipletests(physio_block['p'], method='holm')[1]
    physio_block.to_csv(output_dir / 'task3_physio_block_contrast.csv', index=False)
    log.info(f"Physio non-bikeable - bikeable block:\n{physio_block.round(3).to_string(index=False)}")

    physio_three_corr = utils.helper_utils.rm_corr_table(
        df_combined, c.PARTICIPANT_ID, PHYSIO_METRICS, [c.VALENCE, c.AROUSAL], control='segment_id'
    )
    physio_three_corr.to_csv(output_dir / 'task3_physio_rating_correlations.csv', index=False)
    log.info(f"Physio-rating correlations (Task 3):\n{physio_three_corr.round(3).to_string(index=False)}")

    # ==============================================================================
    # PHASE 5: AGGREGATION RULES (Task 3)
    # ==============================================================================
    log.info("Phase 5: Aggregation rules")

    # Predictors from the Task 1 clip means of the three segments
    rules = {
        'Additive': 'a_mean',
        'Sequential': 'a_mean + a_recency',
        'Peak-End': 'a_best + a_worst + a_end',
        'Minimum-End': 'a_worst + a_end',
    }
    rule_fits, recency_tests = [], []

    for outcome in [c.VALENCE, c.AROUSAL]:
        segments = df_combined[[f'pos{k}_{outcome}' for k in (1, 2, 3)]].to_numpy()
        low, high = segments.min(axis=1), segments.max(axis=1)
        is_valence = outcome == c.VALENCE
        # Worst moment = lowest valence or highest arousal
        df_rules = df_combined.assign(
            a_mean=segments.mean(axis=1),
            a_recency=segments[:, 2] - segments[:, 0],
            a_end=segments[:, 2],
            a_best=high if is_valence else low,
            a_worst=low if is_valence else high,
        )
        fits = {
            rule: utils.lmm_utils.run_lmm(
                df=df_rules, formula=f"{outcome} ~ {terms}",
                groups_col=c.PARTICIPANT_ID, convergence_method=OPT, verbose=False
            )
            for rule, terms in rules.items()
        }

        # Table 4: AIC, marginal and conditional R2
        for rule, model in fits.items():
            r2m, r2c = utils.lmm_utils.calculate_r2_lmm(model)
            rule_fits.append({
                'outcome': outcome, 'rule': rule, 'AIC': model.aic, 'R2m': r2m, 'R2c': r2c
            })

        # Recency term: LR test Sequential vs Additive
        sequential = fits['Sequential']
        lr = utils.helper_utils.lr_test(fits['Additive'], sequential)
        recency = utils.lmm_utils.lmm_contrast(sequential, {'a_recency': 1})
        recency_tests.append({
            'outcome': outcome, 'term': 'recency', **recency,
            'lr_chi2': lr['lrt'], 'lr_df': lr['df'], 'lr_p': lr['p']
        })
        # Implied segment weights: w_k = b_mean / 3 + (k - 2) * b_recency
        for k in (1, 2, 3):
            weight = utils.lmm_utils.lmm_contrast(sequential, {'a_mean': 1 / 3, 'a_recency': k - 2})
            recency_tests.append({'outcome': outcome, 'term': f'weight_{k}', **weight})
        w1, w3 = recency_tests[-3]['estimate'], recency_tests[-1]['estimate']
        recency_tests.append({'outcome': outcome, 'term': 'weight_3 / weight_1', 'estimate': w3 / w1})

    rule_fits = pd.DataFrame(rule_fits)
    rule_fits.to_csv(output_dir / 'task3_aggregation_rules.csv', index=False)
    log.info(f"Aggregation rules:\n{rule_fits.round(3).to_string(index=False)}")

    recency_tests = pd.DataFrame(recency_tests)
    recency_tests.to_csv(output_dir / 'task3_recency_weights.csv', index=False)
    log.info(f"Recency term and segment weights:\n{recency_tests.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
