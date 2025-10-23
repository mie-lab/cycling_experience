from collections import Counter
import pandas as pd
import re
from scipy.stats import pearsonr, spearmanr
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
import constants as c
import utils.processing_utils


def get_trial_dict(study_results, experiment_setup, trial_label, trial_video_counts):
    """
    Extracts ratings and rankings for a specific trial from study results and experiment setup.
    :param study_results: DataFrame with study results
    :param experiment_setup: DataFrame with experiment setup
    :param trial_label: label of the trial to extract (e.g., 'DJI', '2', '3', '4')
    :param trial_video_counts: dict with number of videos per trial
    :return: Dictionary with participant IDs as keys and their ratings/rankings as values
    """
    trial_dict = {}
    n_total = trial_video_counts[trial_label]

    if trial_label == 'DJI':
        n_ratings = n_total
    else:
        n_ratings = n_total // 2

    for pid in study_results.index:
        if pid not in experiment_setup.index:
            continue

        setup_row = experiment_setup.loc[pid]

        trial_order = list(dict.fromkeys(fn.split('_')[0] for fn in setup_row if pd.notna(fn)))

        idx = trial_order.index(trial_label)
        col_start = sum(trial_video_counts[t] for t in trial_order[:idx])
        col_mid = col_start + n_ratings
        col_end = col_start + n_total
        vals = study_results.loc[pid].values
        files = setup_row[setup_row.str.startswith(f'{trial_label}_') & pd.notna(setup_row)].tolist()
        entry = {'ratings': list(zip(files[:n_ratings], vals[col_start:col_mid].tolist()))}

        if n_ratings < n_total:
            entry['ranks'] = list(zip(files[:n_ratings], vals[col_mid:col_end].tolist()))

        trial_dict[pid] = entry

    return trial_dict


def trial_dict_to_df(trial_dict):
    """
    Converts a trial dictionary to a DataFrame.
    :param trial_dict: Dictionary with participant IDs as keys and their ratings/rankings as values
    :return: DataFrame with columns: participant_id, video_id, rating, ranking (if available)
    """
    records = []
    for pid, data in trial_dict.items():
        rank_map = {fname: r for fname, r in data.get('ranks', [])}

        for fname, rating in data['ratings']:
            if 'DJI' in fname:
                match = re.search(r'video_(\d+)', fname)
                fname = int(match.group(1)) if match else None
            row = {
                'participant_id': pid,
                'video_id': fname,
                'rating': int(rating),
            }
            if 'ranks' in data:
                row['ranking'] = rank_map[fname]
            records.append(row)

    return pd.DataFrame(records)



def get_video_level_metrics(df, lab_sample, prediction, online_sample):
    """
    Calculate and print various metrics comparing lab sample ratings, predictions, and online sample ratings.
    :param df: input DataFrame
    :param lab_sample: lab sample column name
    :param prediction: prediction column name
    :param online_sample: online sample column name
    :return: None
    """
    df_pred = df.dropna(subset=[lab_sample, prediction])
    df_agree = df.dropna(subset=[lab_sample, online_sample])
    y_true, y_pred = df_pred[lab_sample], df_pred[prediction]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    pr, _ = pearsonr(y_true, y_pred)
    sr, _ = spearmanr(y_true, y_pred)
    print(f"Prediction metrics and corr: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}, Pearson r={pr:.3f}, Spearman ρ={sr:.3f}")
    print(f'lab/online corr: Pearson r={pearsonr(df_agree[lab_sample].values, df_agree[online_sample].values)[0]}')
    print(f'lab/online corr: Spearman ρ={spearmanr(df_agree[lab_sample].values, df_agree[online_sample].values)[0]}')


def check_variance_homogeneity(df, group_col, target_col, center='median', alpha=0.05, print_msg=True):
    """
    Levene's test for equal variances across groups.
    :param df: input DataFrame
    :param group_col: column to group by
    :param target_col: column to test variances on
    :param center: 'mean' or 'median' for Levene's test
    :param alpha: significance level
    :param print_msg: whether to print the result message
    :return: LeveneResult object
    """
    groups = [g[target_col].dropna().values for _, g in df.groupby(group_col)]
    res = stats.levene(*groups, center=center)
    if print_msg:
        msg = (f"Variances differ across {group_col}, (statistic={res.statistic:.3g}, p={res.pvalue:.3g})."
               if res.pvalue < alpha else
               f"No evidence variances differ across {group_col} (statistic={res.statistic:.3g}. p={res.pvalue:.3g}).")
        print(msg)
    return res


def add_sequence_info(df, video_mapping, video_sequences, video_level_scores, scenario, valence_col=c.VALENCE):
    """
    Adds sequence information to the main DataFrame.
    :param nb: whether to look for 'NB' or 'B' in sequences, default scenario is NB (non-bikeablre)
    :param df: main DataFrame
    :param video_mapping: mapping of video IDs to sequences
    :param video_sequences: DataFrame with video sequences
    :param video_level_scores: DataFrame with video-level scores
    :param valence_col: name of the valence column
    :return: DataFrame with added sequence information
    """
    df['sequence_list'] = df[c.VIDEO_ID_COL].map(video_mapping)
    cond_df = pd.DataFrame(df['sequence_list'].tolist(), index=df.index)
    cond_df.columns = [f"pos{i + 1}" for i in range(cond_df.shape[1])]
    df[cond_df.columns] = cond_df
    val_map = dict(zip(video_level_scores[c.VIDEO_ID_COL], video_level_scores[valence_col]))

    max_conds = cond_df.shape[1]
    cond_ids = {f"pos{i + 1}_{c.VIDEO_ID_COL}": [] for i in range(max_conds)}
    cond_valences = {f"pos{i + 1}_{c.VALENCE}": [] for i in range(max_conds)}

    for idx, row in df.iterrows():
        pid = row[c.PARTICIPANT_ID]
        vid = row[c.VIDEO_ID_COL]
        conds = row['sequence_list']  # list of conditions for this video

        for i, cond in enumerate(conds):
            fname = video_sequences.loc[pid, f"{vid}_{i + 1}_{cond}"]
            num = int(fname.split('_video_')[1].split('_')[0])
            cond_ids[f"pos{i + 1}_{c.VIDEO_ID_COL}"].append(num)
            cond_valences[f"pos{i + 1}_{c.VALENCE}"].append(val_map[num])

    # Assign new columns back
    for d in (cond_ids, cond_valences):
        for col, lst in d.items():
            df[col] = lst

    df['B_counts'] = df['sequence_list'].apply(lambda l: Counter(l)['B'])
    df['NB_counts'] = df['sequence_list'].apply(lambda l: Counter(l)['NB'])

    if scenario == 'NB':
        # Add NB position column (1-based index, 0 if no NB in sequence)
        df['NB_position'] = df['sequence_list'].apply(lambda x: 0 if 'NB' not in x else x.index('NB') + 1)

    else:
        # Add B position column (1-based index, 0 if no B in sequence)
        df['B_position'] = df['sequence_list'].apply(lambda x: 0 if 'B' not in x else x.index('B') + 1)

    # 2. Dynamically create a list of the new valence columns for calculations
    pos_valence_cols = [f"pos{i + 1}_{valence_col}" for i in range(max_conds)]

    if pos_valence_cols:
        df['mean_valence'] = df[pos_valence_cols].mean(axis=1)
        df['pos_peak_valence'] = df[pos_valence_cols].max(axis=1)
        df['neg_peak_valence'] = df[pos_valence_cols].min(axis=1)
        df['end_valence'] = df[pos_valence_cols[-1]]

    return df


def load_and_process_trial_data(
        study_results: pd.DataFrame,
        experiment_setup: pd.DataFrame,
        video_sequences: pd.DataFrame,
        trial_params: dict,
        video_level_scores: pd.DataFrame,
        scenario='NB'
) -> pd.DataFrame:
    """
    Load and process trial data into a DataFrame and enrich with sequence info.
    :param scenario: 'NB' or 'B' indicating the scenario type
    :param study_results: DataFrame with study results
    :param experiment_setup: DataFrame with experiment setup
    :param video_sequences: DataFrame with video sequences
    :param trial_params: dict with trial parameters
    :param video_level_scores: DataFrame with video-level scores
    :return: Processed DataFrame
    """
    trial_dict = get_trial_dict(study_results, experiment_setup, trial_params['trial_label'], c.VIDEO_COUNTS)
    df = trial_dict_to_df(trial_dict)

    df['video_order'] = (df[c.VIDEO_ID_COL].str.extract(r'_(\d+)\.', expand=False).astype(int))
    df[c.VIDEO_ID_COL] = df[c.VIDEO_ID_COL].str.replace(r'\.mp4$', '', regex=True)
    df = df.sort_values([c.PARTICIPANT_ID, 'video_order'])

    df = utils.processing_utils.add_valence_arousal(df, 'rating')
    df = add_sequence_info(df, trial_params['video_mapping'], video_sequences, video_level_scores, scenario)

    return df











