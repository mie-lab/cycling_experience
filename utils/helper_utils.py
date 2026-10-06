from collections import Counter
import pandas as pd
import re
from scipy.stats import pearsonr, spearmanr, friedmanchisquare, page_trend_test
import numpy as np
from scipy import stats
import constants as c
import utils.process_utils
from scipy.stats import wilcoxon


def get_trial_dict(study_results, experiment_setup, trial_label, trial_video_counts):
    """
    Extract ratings (and rankings) for one trial from the study results.
    `trial_label` is e.g. 'DJI', '2', '3', '4'. Returns {participant_id: {ratings, ranks}}.
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
    """Flatten a trial dict into rows of participant_id, video_id, rating[, ranking]."""
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


def _boot_ci(x, y, corr, n_boot=5000, seed=42):
    """95% percentile bootstrap CI of a correlation, resampling clips."""
    idx = np.random.default_rng(seed).integers(0, len(x), (n_boot, len(x)))
    return np.nanpercentile([corr(x[i], y[i])[0] for i in idx], [2.5, 97.5])


def get_video_level_metrics(df, lab_sample, prediction, online_sample):
    """Agreement of lab clip means with the k-NN prediction and with the online survey."""
    out = {}
    for ref, col in (("pred", prediction), ("online", online_sample)):
        d = df.dropna(subset=[lab_sample, col])
        x, y = d[lab_sample].to_numpy(), d[col].to_numpy()
        diff = x - y
        res = {"n": len(d), "bias": diff.mean(), "rmse": np.sqrt((diff ** 2).mean()), "mae": np.abs(diff).mean()}
        for name, corr in (("pearson", pearsonr), ("spearman", spearmanr)):
            res[name] = corr(x, y)[0]
            res[f"{name}_ci_low"], res[f"{name}_ci_high"] = _boot_ci(x, y, corr)
        out.update({f"{k}_{ref}": v for k, v in res.items()})
    return out


def get_clip_types(lab_seq_df):
    """Map clip id to 'B' or 'NB', parsed from the file names in the lab sequence file."""
    files = lab_seq_df.drop(columns='participant').stack()
    parsed = files.str.extract(r'video_(\d+)_(NB|B)').dropna().drop_duplicates()
    return parsed.set_index(parsed[0].astype(int))[1]


def check_variance_homogeneity(df, group_col, target_col, center='median', alpha=0.05, print_msg=True):
    """Levene's test for equal variances of `target_col` across `group_col`."""
    groups = [g[target_col].dropna().values for _, g in df.groupby(group_col)]
    res = stats.levene(*groups, center=center)
    if print_msg:
        msg = (f"Variances differ across {group_col}, (statistic={res.statistic:.3g}, p={res.pvalue:.3g})."
               if res.pvalue < alpha else
               f"No evidence variances differ across {group_col} (statistic={res.statistic:.3g}. p={res.pvalue:.3g}).")
        print(msg)
    return res


def add_sequence_info(
        df,
        video_mapping,
        video_sequences,
        video_level_scores,
        scenario,
        valence_col=c.VALENCE,
        arousal_col=c.AROUSAL
):
    """
    Expand each sequence into per-position clip ids and valence/arousal columns, plus
    B/NB counts, the off-type `scenario` position, and mean/peak/end aggregates.
    """
    df['sequence_list'] = df[c.VIDEO_ID_COL].map(video_mapping)
    cond_df = pd.DataFrame(df['sequence_list'].tolist(), index=df.index)
    cond_df.columns = [f"pos{i + 1}" for i in range(cond_df.shape[1])]
    df[cond_df.columns] = cond_df

    # 2. Create maps for BOTH valence and arousal
    val_map = dict(zip(video_level_scores[c.VIDEO_ID_COL], video_level_scores[valence_col]))
    aro_map = dict(zip(video_level_scores[c.VIDEO_ID_COL], video_level_scores[arousal_col]))

    max_conds = cond_df.shape[1]

    # 3. Initialize dictionaries to hold the new column data for both axes
    cond_ids = {f"pos{i + 1}_{c.VIDEO_ID_COL}": [] for i in range(max_conds)}
    cond_valences = {f"pos{i + 1}_{valence_col}": [] for i in range(max_conds)}
    cond_arousals = {f"pos{i + 1}_{arousal_col}": [] for i in range(max_conds)}

    for idx, row in df.iterrows():
        pid = row[c.PARTICIPANT_ID]
        vid = row[c.VIDEO_ID_COL]
        conds = row['sequence_list']  # list of conditions for this video

        for i, cond in enumerate(conds):
            fname = video_sequences.loc[pid, f"{vid}_{i + 1}_{cond}"]
            num = int(fname.split('_video_')[1].split('_')[0])

            cond_ids[f"pos{i + 1}_{c.VIDEO_ID_COL}"].append(num)
            # Append to BOTH lists
            cond_valences[f"pos{i + 1}_{valence_col}"].append(val_map.get(num, np.nan))
            cond_arousals[f"pos{i + 1}_{arousal_col}"].append(aro_map.get(num, np.nan))

    # Assign new columns back to the dataframe
    for d in (cond_ids, cond_valences, cond_arousals):
        for col, lst in d.items():
            df[col] = lst

    df['B_counts'] = df['sequence_list'].apply(lambda l: Counter(l)['B'])
    df['NB_counts'] = df['sequence_list'].apply(lambda l: Counter(l)['NB'])

    # Position of the off-type segment (`scenario` = its type): 1-3, or 0 for the baseline
    df['off_type_position'] = df['sequence_list'].apply(
        lambda seq: seq.index(scenario) + 1 if scenario in seq else 0
    )

    # Calculate mean, peak, and end VALENCE
    pos_valence_cols = [f"pos{i + 1}_{valence_col}" for i in range(max_conds)]
    if pos_valence_cols:
        df['mean_valence'] = df[pos_valence_cols].mean(axis=1)
        df['pos_peak_valence'] = df[pos_valence_cols].max(axis=1)
        df['neg_peak_valence'] = df[pos_valence_cols].min(axis=1)
        df['end_valence'] = df[pos_valence_cols[-1]]

    # Calculate mean, peak, and end AROUSAL
    pos_arousal_cols = [f"pos{i + 1}_{arousal_col}" for i in range(max_conds)]
    if pos_arousal_cols:
        df['mean_arousal'] = df[pos_arousal_cols].mean(axis=1)
        df['pos_peak_arousal'] = df[pos_arousal_cols].max(axis=1)
        df['neg_peak_arousal'] = df[pos_arousal_cols].min(axis=1)
        df['end_arousal'] = df[pos_arousal_cols[-1]]

    return df


def load_and_process_trial_data(
        study_results: pd.DataFrame,
        experiment_setup: pd.DataFrame,
        video_sequences: pd.DataFrame,
        trial_params: dict,
        video_level_scores: pd.DataFrame,
        scenario='NB'
) -> pd.DataFrame:
    """Load one trial's ratings/rankings and enrich with sequence info (`scenario` = 'NB'|'B')."""
    trial_dict = get_trial_dict(study_results, experiment_setup, trial_params['trial_label'], c.VIDEO_COUNTS)
    df = trial_dict_to_df(trial_dict)

    df[c.VIDEO_ID_COL] = df[c.VIDEO_ID_COL].str.replace(r'\.mp4$', '', regex=True)
    df = df.sort_values([c.PARTICIPANT_ID, c.VIDEO_ID_COL])

    df = utils.process_utils.add_valence_arousal(df, 'rating')
    df = add_sequence_info(df, trial_params['video_mapping'], video_sequences, video_level_scores, scenario)

    return df


def repeated_measures_corr(df, subject_col, x, y, control=None):
    """
    Repeated-measures correlation (Bakdash & Marusich, 2017): Pearson r of the
    within-subject centred x and y, with df = N - subjects - 1 and a Fisher-z 95% CI.
    `control` (e.g. presentation slot) is partialled out of both (one df less).
    """
    cols = [x, y] + ([control] if control else [])
    d = df[[subject_col] + cols].dropna()
    centred = d[cols] - d.groupby(subject_col)[cols].transform('mean')
    if control:
        z = centred[control]
        centred = centred[[x, y]].apply(lambda v: v - (v @ z) / (z @ z) * z)
    r = centred[x].corr(centred[y])
    dof = len(d) - d[subject_col].nunique() - 1 - bool(control)
    p = 2 * stats.t.sf(abs(r) * np.sqrt(dof / (1 - r ** 2)), dof)
    z, se = np.arctanh(r), 1 / np.sqrt(dof - 1)
    return {'n': len(d), 'df': dof, 'r': r, 'p': p,
            'ci_low': np.tanh(z - 1.96 * se), 'ci_high': np.tanh(z + 1.96 * se)}


def rm_corr_table(df, subject_col, metrics, outcomes, control=None):
    """Repeated-measures r with 95% CI for every metric x outcome pair (descriptive, no p)."""
    rows = []
    for metric in metrics:
        for outcome in outcomes:
            result = repeated_measures_corr(df, subject_col, metric, outcome, control)
            rows.append({'metric': metric, 'outcome': outcome,
                         **{k: result[k] for k in ('n', 'r', 'ci_low', 'ci_high')}})
    return pd.DataFrame(rows)


def wilcoxon_pair(df, subject_col, condition_col, value_col, cond_a, cond_b):
    """
    Paired Wilcoxon signed-rank test of cond_a vs cond_b, plus the mean
    paired difference (a - b) with its 95% t-interval. Zero differences
    are dropped by the test, so n_nonzero is the test's sample size.
    """
    wide = df.pivot_table(index=subject_col, columns=condition_col, values=value_col)
    wide = wide[[cond_a, cond_b]].dropna()
    diff = wide[cond_a] - wide[cond_b]
    half_width = stats.t.ppf(0.975, len(diff) - 1) * diff.sem()
    stat, p = wilcoxon(wide[cond_a], wide[cond_b])
    return {
        "comparison": f"{cond_a} - {cond_b}", "outcome": value_col,
        "n_pairs": len(wide), "n_nonzero": int((diff != 0).sum()),
        "mean_a": wide[cond_a].mean(), "mean_b": wide[cond_b].mean(),
        "mean_diff": diff.mean(), "ci_low": diff.mean() - half_width,
        "ci_high": diff.mean() + half_width, "W": float(stat), "p": float(p),
    }


def wald_contrast(fit, weights: dict, label=""):
    """Linear contrast on a fitted MixedLM: weights = {param_name: w}."""
    params = fit.params
    cov = fit.cov_params()
    L = pd.Series(0.0, index=params.index)
    for name, w in weights.items():
        if name not in params.index:
            raise KeyError(f"Param '{name}' not in model. Available: {list(params.index)}")
        L[name] = w
    est = float(L @ params)
    se = float(np.sqrt(L @ cov @ L))
    z = est / se
    p = 2 * stats.norm.sf(abs(z))
    return {"contrast": label, "estimate": est, "se": se, "z": z, "p": p,
            "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se}


def lr_test(fit_small, fit_big, label=""):
    """LR test with sanity checks for nesting (both must be ML fits)."""
    lrt = 2 * (fit_big.llf - fit_small.llf)
    df = len(fit_big.params) - len(fit_small.params)
    if df <= 0:
        return {"label": label, "lrt": lrt, "df": df, "p": np.nan,
                "note": "DEGENERATE: big model has no extra params — not nested as intended"}
    if lrt < -1e-6:
        return {"label": label, "lrt": lrt, "df": df, "p": np.nan,
                "note": "DEGENERATE: larger model has LOWER loglik — models not nested "
                        "or optimizer failure; refit both with same convergence_method"}
    p = stats.chi2.sf(max(lrt, 0.0), df)
    return {"label": label, "lrt": float(lrt), "df": int(df), "p": float(p), "note": ""}


def friedman_kendall(df, subject_col, condition_col, value_col):
    """
    Friedman test + Kendall's W on repeated rankings (complete cases only).
    Returns chi2, df, p, kendall_w, n_subjects, n_conditions.
    """
    wide = df.pivot_table(index=subject_col, columns=condition_col, values=value_col)
    wide = wide.dropna(axis=0)  # complete cases only — Friedman needs balanced data

    n, k = wide.shape
    if n < 2 or k < 3:
        raise ValueError(f"Need >=2 subjects and >=3 conditions; got {n} x {k}.")

    chi2, p = friedmanchisquare(*[wide[c].values for c in wide.columns])
    kendall_w = chi2 / (n * (k - 1))  # W = chi2 / [n(k-1)]

    return {
        "chi2": float(chi2), "df": k - 1, "p": float(p),
        "kendall_w": float(kendall_w), "n_subjects": int(n), "n_conditions": int(k),
    }


def page_trend_ranking(df, subject_col, condition_col, value_col, predicted_order):
    """
    Page's trend test for an ordered alternative on repeated rankings: is `value_col`
    monotonically increasing across `condition_col` in `predicted_order`? Columns are
    ordered to `predicted_order` (predicted ranks 1..k), complete cases only, and
    scipy's Page L test runs on the within-subject ranks (`ranked=False`, tie-averaged).
    Returns L, p, method, n_subjects, n_dropped, n_conditions, predicted_order,
    mean_rank_by_condition.
    """
    predicted_order = list(predicted_order)

    wide = df.pivot_table(index=subject_col, columns=condition_col, values=value_col)

    missing = [cond for cond in predicted_order if cond not in wide.columns]
    if missing:
        raise ValueError(f"Conditions {missing} not found in '{condition_col}'.")

    wide = wide[predicted_order]        # order columns by the predicted trend
    n_before = len(wide)
    wide = wide.dropna(axis=0)          # complete cases only
    n_after = len(wide)

    n, k = wide.shape
    if n < 2 or k < 3:
        raise ValueError(f"Need >=2 subjects and >=3 conditions; got {n} x {k}.")

    res = page_trend_test(wide.to_numpy(), ranked=False, method="auto")

    mean_rank = df.groupby(condition_col)[value_col].mean()
    mean_rank_by_condition = {cond: round(float(mean_rank[cond]), 3) for cond in predicted_order}

    return {
        "L": float(res.statistic),
        "p": float(res.pvalue),
        "method": res.method,
        "n_subjects": int(n_after),
        "n_dropped": int(n_before - n_after),
        "n_conditions": int(k),
        "predicted_order": predicted_order,
        "mean_rank_by_condition": mean_rank_by_condition,
    }