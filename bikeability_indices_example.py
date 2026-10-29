import configparser
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import constants as c
import utils.bikeability_utils as B
import utils.helper_utils as H
import utils.plot_utils
import utils.process_utils as P

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BLOCKS = {'Positive': 'bikeable', 'Negative': 'non-bikeable'}


def main():
    config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation()
    )
    config.read("config.ini")
    files, cfg = config['filenames'], config['bikeability']
    output_dir = Path(cfg['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lab data: Task 1 clip ratings and Task 3 routes
    demo_cols = [col for col in c.DEMOGRAPHIC_COLUMNS if col != c.IS_SWISS]
    lab = (
        pd.read_excel(files['lab_study_results_file'])
        .set_index(c.PARTICIPANT_ID)
        .replace(r'\s*\((best|worst) experience\)', '', regex=True)
        .drop(columns=demo_cols + [c.START, c.END])
        .apply(pd.to_numeric, errors='coerce')
    )
    lab_seq = pd.read_csv(files['lab_video_sequence_file'])
    setup = pd.read_csv(files['lab_experiment_setup_file'], header=None)
    setup = setup.set_index(0)

    trials = H.get_trial_dict(lab, setup, c.TRIAL_1, c.VIDEO_COUNTS)
    task1 = P.add_valence_arousal(H.trial_dict_to_df(trials), ag_col='rating')
    clip_ratings = P.calc_video_level_scores(task1, lab_bool=True)

    routes = P.prepare_combined_scenario_df(
        H.load_and_process_trial_data(
            lab, setup, lab_seq, c.TRIAL_3_PARAMS, clip_ratings, 'NB'),
        H.load_and_process_trial_data(
            lab, setup, lab_seq, c.TRIAL_4_PARAMS, clip_ratings, 'B'),
    )
    clip_types = H.get_clip_types(lab_seq)
    routes['block'] = routes['scenario'].map(BLOCKS)
    positions = [f'pos{k}_{c.VIDEO_ID_COL}' for k in (1, 2, 3)]
    routes['sequence'] = (
        routes[positions]
        .apply(lambda col: col.map(clip_types))
        .agg('-'.join, axis=1)
    )

    # Clip attributes and indices
    traces = list(Path(cfg['video_traces_dir']).glob('*.gpkg'))
    clips = P.aggregate_video_level_geometry(traces)
    clips = clips[clips[c.VIDEO_ID_COL].isin(clip_types.index)]
    ground_truth = pd.read_csv(cfg['video_ground_truth_file'])
    ground_truth = ground_truth.drop(columns=['length', 'slope', 'index'])
    clips = clips.merge(ground_truth, on=c.VIDEO_ID_COL)
    clips = B.add_context(clips, cfg)

    ratings = clip_ratings[[c.VIDEO_ID_COL, c.VALENCE, c.AROUSAL]]
    indices = B.calc_indices(clips).merge(ratings, on=c.VIDEO_ID_COL)
    indices.to_csv(output_dir / 'clip_indices.csv', index=False)

    rho = pd.Series({
        i: stats.spearmanr(indices[i], indices[c.AROUSAL])[0]
        for i in B.INDICES
    })
    log.info(f"{len(indices)} clips, {len(routes)} routes; "
             f"rho with Task 1 arousal:\n{rho.round(2).to_string()}")

    # Route values: simple average and Sequential (RQ3 arousal weights)
    recency = pd.read_csv(cfg['recency_weights_file'])
    recency = recency.set_index(['outcome', 'term'])['estimate']
    weights = np.array([recency['arousal', f'weight_{k}'] for k in (1, 2, 3)])
    scores = B.route_scores(routes, indices, weights)
    seq = B.sequence_table(scores, indices)
    seq.to_csv(output_dir / 'route_index_by_sequence.csv', index=False)

    utils.plot_utils.plot_sequence_rules(
        seq[seq['off_type_position'] > 0],
        B.INDICES,
        'vs_baseline_pct',
        "Change from baseline route\n(% of clip range)",
        output_dir / 'sequence_vs_baseline.png',
    )

    # Numbers for the text (% of clip range): effect of the off-type segment
    # under averaging, and the order effect under Sequential
    mixed = seq[(seq['off_type_position'] > 0) & (seq['rule'] == 'average')]
    segment = mixed.groupby(['index', 'block'])['vs_baseline_pct'].mean().unstack()
    summary = pd.concat({'segment': segment, 'order': B.order_effect(seq)}, axis=1)
    summary = summary.loc[B.INDICES]
    summary[('', 'rho_arousal')] = rho
    summary.loc['mean'] = summary.mean()
    summary.to_csv(output_dir / 'order_effect_summary.csv')
    log.info(f"Segment and order effects (% of clip range):\n"
             f"{summary.round(2).to_string()}")


if __name__ == "__main__":
    main()
