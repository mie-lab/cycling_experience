import os
from shapely import LineString
import geopandas as gpd
import logging
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
import re
from typing import Sequence
from scipy.spatial.distance import cdist
import matplotlib
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import numpy as np
import constants as c
import ast
import pandas as pd
import statsmodels.formula.api as smf
matplotlib.use("TkAgg")


# --- Functions for Candidate Video Selection ---
log = logging.getLogger(__name__)


def transform_to_long_df(
        df: pd.DataFrame,
        seq_df: pd.DataFrame,
        id_col: str = c.PARTICIPANT_ID
) -> pd.DataFrame:
    """
    Convert a DataFrame with sequence data into a long format DataFrame.
    :param df: DataFrame containing the sequence data.
    :param seq_df: DataFrame containing sequence information with columns 'seq_start', 'seq_end', and 'Sequence'.
    :param id_col: str = "ID"
    :return: DataFrame in long format with columns for ID, demographics, video_id, and question responses.
    """
    questions = [c.OE, c.AG, c.PF, c.NF, c.F]
    demo_cols = [c.START, c.END, c.CONSENT, c.GENDER, c.AGE, c.COUNTRY, c.CYCL_FREQ, c.CYCL_PURP, c.CYCL_CONF,
                 c.CYCL_ENV]

    d = df.copy()
    d[c.START] = pd.to_datetime(d[c.START], errors='coerce')
    bins = list(seq_df['seq_start']) + [seq_df['seq_end'].iloc[-1]]

    labels = [tuple(ast.literal_eval(x)) if isinstance(x, str) else x for x in seq_df['Sequence']]
    d['Sequence'] = pd.cut(d[c.START], bins=bins, right=False, labels=labels)

    pos_to_idx = {
        int(m.group(2)): d.columns.get_indexer([f"{q}{m.group(2)}" for q in questions])
        for col in d.columns if (m := re.fullmatch(f"({c.OE}|{c.AG}|{c.PF}|{c.NF}|{c.F})(\d+)", col))
    }

    rows = []
    for user, row in d.iterrows():
        seq = row.get("Sequence")
        pid = row[id_col] if id_col in d.columns else user
        demo_vals = {col: row[col] for col in demo_cols}

        for pos, video_id in enumerate(seq, start=1):
            idxs = pos_to_idx.get(pos)
            if idxs is None:
                continue
            vals = row.iloc[idxs].tolist()
            qa = dict(zip(questions, vals))
            rows.append({id_col: pid, **demo_vals, c.VIDEO_ID_COL: int(video_id), **qa})

    return pd.DataFrame(rows, columns=[id_col] + demo_cols + [c.VIDEO_ID_COL] + questions)


def filter_results(
        df: pd.DataFrame,
        consent: bool = True,
        duration: bool = True,
        location: bool = True,
        by_country: bool = False
) -> pd.DataFrame:
    """
    Implements filtering logic to clean the survey response data
    :param df: DataFrame containing the data to filter.
    :param consent: bool = True, filter out participants without consent.
    :param duration: bool = True, filter out participants with duration less than 15 minutes.
    :param location: bool = True, filter out participants with missing location data.
    :param by_country: bool = False, if True, apply location filter by country.
    :return: Filtered DataFrame.
    """
    df = df.copy()
    if c.AG in df.columns:
        # small correction for a few participants that typed 0 instead of 1
        df.loc[df[c.AG].eq(0), c.AG] = 1

    mask = pd.Series(True, index=df.index)

    if consent:
        mask &= df[c.CONSENT] != 'Do not consent'

    if duration:
        start = pd.to_datetime(df[c.START], format='%m/%d/%y %H:%M:%S', errors='coerce')
        end = pd.to_datetime(df[c.END], format='%m/%d/%y %H:%M:%S', errors='coerce')
        mask &= (end - start) >= pd.Timedelta(minutes=15) # video duration without breaks for scoring

    if location:
        mask &= df[c.COUNTRY] != '_'

    if by_country:
        mask &= df[c.COUNTRY].isin(['Switzerland', 'CH'])

    return df.loc[mask]


def aggregate_by_characteristics(
        df: pd.DataFrame,
        gender: bool = False,
        age: bool = False,
        cycling_environment: bool = False,
        cycling_frequency: bool = False,
        cycling_confidence: bool = False,
        cycling_purpose: bool = False,
        familiarity: bool = False,
        is_swiss: bool = False
):
    """
    Implements aggregation logic to group categories for demographic and cycling-related characteristics.
    :param is_swiss: if True, create a new binary column 'is_swiss' based on the 'Country' column.
    :param df: DataFrame containing the data to filter.
    :param gender: if True, remove 'Prefer not to say' category.
    :param age: if True, upward aggregate age groups from '46 - 55 years', '56 - 65 years', '+65 years' to '46+ years'.
    :param cycling_environment: if True, remove 'Other' category.
    :param cycling_frequency: if True, aggregate cycling frequency categories into 'Frequent' and 'Infrequent'.
    :param cycling_confidence: if True, aggregate cycling confidence into 'Confident' and 'Not confident'.
    :param cycling_purpose: if True, aggregate cycling purpose categories into 'Commuting' and 'Recreational', 'I do not cycle'.
    :param familiarity: if True, aggregate familiarity categories into 'Unfamiliar', 'Neutral', and 'Familiar'.
    :return: aggregated DataFrame.
    """
    df = df.copy()
    mask = pd.Series(True, index=df.index)

    # if upward aggregation needed to gain some significance power
    if age:
        age_lookup = {
            '46 - 55 years': '46+ years',
            '56 - 65 years': '46+ years',
            '+65 years': '46+ years'
        }

        df[c.AGE] = df[c.AGE].replace(age_lookup)

    if gender:
        mask &= df[c.GENDER].isin(['Male', 'Female'])

    if cycling_environment:
        mask &= df[c.CYCL_ENV] != 'Other'

    if cycling_frequency:
        mask &= df[c.CYCL_FREQ] != 'Other'
        frequency_lookup = {
            "Never": "Infrequent",
            "Less than once a month": "Infrequent",
            "1-3 times/month": "Infrequent",
            "1-2 days/week": "Frequent",
            "3-4 days/week": "Frequent",
            "5-6 days/week": "Frequent",
            "Every day": "Frequent"
        }

        df[c.CYCL_FREQ] = df[c.CYCL_FREQ].replace(frequency_lookup)

    if cycling_confidence:
        frequency_lookup = {
            'Very confident': "Confident",
            'Somewhat confident': "Confident",
            'Slightly not confident': "Not confident",
            'Not confident at all': "Not confident",
        }
        df[c.CYCL_CONF] = df[c.CYCL_CONF].replace(frequency_lookup)

    if cycling_purpose:
        purpose_lookup = {
            'Commuting (e.g., work, school)': 'Commuting',
            'All purposes': 'Commuting',
            'Recreational / leisure': 'Recreational',
            'Exercise / fitness': 'Recreational',
            'Other': 'Recreational'
        }
        df[c.CYCL_PURP] = df[c.CYCL_PURP].replace(purpose_lookup)

    if familiarity:
        familiarity_lookup = {
            'Not at all familiar': 'Unfamiliar',
            'Somewhat unfamiliar': 'Unfamiliar',
            'Equally familiar and unfamiliar': 'Neutral',
            'Somewhat familiar': 'Familiar',
            'Extremely familiar': 'Familiar'
        }
        df[c.F] = df[c.F].replace(familiarity_lookup)

    df = df.loc[mask]

    if is_swiss:
        # Define Switzerland binary (handles potential 'CH' or 'Switzerland' strings)
        swiss_labels = ['Switzerland', 'CH']
        df[c.IS_SWISS] = df[c.COUNTRY].apply(
            lambda x: 'Swiss Resident' if x in swiss_labels else 'Not Swiss Resident'
        )

    return df


def get_marginal_affective_drivers(long_df, label_cols):
    """
    Isolates the 'Marginal Pull' of environmental factors.
    Calculates the displacement vector between the tag-selecting subgroup and the specific video consensus.
    """
    # Video consensus serves as the 'zero-point' for each scenario
    video_consensus = long_df.groupby(c.VIDEO_ID_COL)[[c.VALENCE, c.AROUSAL]].mean()

    driver_data = []
    for tag in label_cols:
        is_pf = tag.startswith('PF:')
        source_col = c.PF if is_pf else c.NF
        label_name = tag.split(': ')[1]

        # Parse semicolon-separated labels to identify the subgroup
        def match_label(val):
            return label_name in [s.strip() for s in str(val).split(';')] if pd.notna(val) else False

        subgroup_df = long_df[long_df[source_col].apply(match_label)]

        if not subgroup_df.empty:
            # Mean of participants who perceived this specific factor
            subgroup_means = subgroup_df.groupby(c.VIDEO_ID_COL)[[c.VALENCE, c.AROUSAL]].mean()
            counts = subgroup_df.groupby(c.VIDEO_ID_COL).size()

            # Local Force = (Subgroup Reaction) - (Video Consensus)
            local_marginal_force = subgroup_means - video_consensus.loc[subgroup_means.index]

            # Aggregate to find the study-wide 'Signature' of the factor
            avg_v_pull = (local_marginal_force[c.VALENCE] * counts).sum() / counts.sum()
            avg_a_pull = (local_marginal_force[c.AROUSAL] * counts).sum() / counts.sum()

            driver_data.append({
                'factor': tag,
                'valence_pull': avg_v_pull,
                'arousal_pull': avg_a_pull,
                'magnitude': np.sqrt(avg_v_pull ** 2 + avg_a_pull ** 2),
                'type': 'Positive' if is_pf else 'Negative'
            })

    return pd.DataFrame(driver_data)


def assign_affective_state(df):
    """
    Assign affective state based on valence and arousal values.
    :param df: DataFrame with 'valence' and 'arousal' columns.
    :return: DataFrame with added 'affective_state' column.
    """
    df["affective_state"] = np.select(
        [
            (df["valence"] > 0) & (df["arousal"] > 0),
            (df["valence"] > 0) & (df["arousal"] <= 0),
            (df["valence"] <= 0) & (df["arousal"] > 0),
            (df["valence"] <= 0) & (df["arousal"] <= 0),
        ],
        c.AFFECTIVE_STATES,
        default="Unknown",
    )
    return df


def add_valence_arousal(
        df: pd.DataFrame,
        ag_col: str = c.AG,
        grid: int = 10
) -> pd.DataFrame:
    """
    Add valence and arousal columns to a DataFrame based on the Arousal and Valence grid ratings.
    :param df: DataFrame to modify.
    :param ag_col: Column name containing the Arousal and Valence grid values.
    :param grid: Size of the grid (default is 10).
    :return: DataFrame with added 'valence' and 'arousal' columns.
    """
    df = df.copy()
    ag = pd.to_numeric(df[ag_col], errors="coerce")
    idx0 = ag - 1
    col = (idx0 % grid).astype("Int64")
    row = (idx0 // grid).astype("Int64")
    vals = np.linspace(-1.0, 1.0, grid + 1)
    VALENCE_LUT = np.delete(vals, grid // 2)
    AROUSAL_LUT = VALENCE_LUT[::-1]

    valence = np.full(len(df), np.nan, float)
    arousal = np.full(len(df), np.nan, float)
    valid = ag.between(1, grid * grid, inclusive="both")
    valence[valid] = VALENCE_LUT[col[valid].to_numpy()]
    arousal[valid] = AROUSAL_LUT[row[valid].to_numpy()]

    df[c.VALENCE] = valence
    df[c.AROUSAL] = arousal

    # assign affective state
    df = assign_affective_state(df)

    return df


def calculate_video_level_scores(
        long_df: pd.DataFrame,
        output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:
    """
    Video-level descriptive affect metrics (RQ1) + Structural Disagreement (RQ2).
    Includes Energy Statistic (Distinctiveness) and GMM (Polarization).
    """
    results = []

    # --- 1. SETUP GLOBAL POOL FOR DISTINCTIVENESS METRIC ---
    global_affect = long_df[[c.VALENCE, c.AROUSAL]].dropna().to_numpy(float)

    # Subsample if too large (>5000) to speed up distance matrix calculation
    if len(global_affect) > 5000:
        rng = np.random.default_rng(42)
        global_sample = rng.choice(global_affect, size=5000, replace=False)
    else:
        global_sample = global_affect

    # Calculate internal energy of the global pool (d_yy) ONCE.
    d_yy = cdist(global_sample, global_sample, metric='euclidean').mean()

    state_cols = [f"{s}_count" for s in c.AFFECTIVE_STATES]
    grouped = long_df.groupby(c.VIDEO_ID_COL, sort=True, dropna=True)

    for video_id, video_df in grouped:
        video_affect = video_df[[c.VALENCE, c.AROUSAL]].dropna()
        n_ratings = len(video_affect)

        if n_ratings == 0:
            continue

        valence_values = video_affect[c.VALENCE].to_numpy(float)
        arousal_values = video_affect[c.AROUSAL].to_numpy(float)
        affect_points = np.column_stack([valence_values, arousal_values])

        # -------------------------------------------------
        # OE: convert ordered labels
        # -------------------------------------------------
        oe_num = pd.Categorical(
            video_df[c.OE], categories=c.OE_ORDER, ordered=True
        ).codes.astype(float)
        oe_num[oe_num < 0] = np.nan
        oe_mean = float(np.nanmean(oe_num))
        oe_mode = (
            video_df[c.OE]
            .value_counts()
            .reindex(c.OE_ORDER, fill_value=0)
            .idxmax()
        )

        # -------------------------------------------------
        # affect_state counts + entropy
        # -------------------------------------------------
        state_vc = (
            video_df[c.AFFECT_STATE]
            .value_counts()
            .reindex(c.AFFECTIVE_STATES, fill_value=0)
        )
        state_counts = state_vc.tolist()
        dominant_quadrant = state_vc.idxmax()

        total = float(state_vc.sum())
        if total > 0:
            p = (state_vc / total).to_numpy(float)
            p = p[p > 0]
            affect_state_entropy = round(float(-(p * np.log(p)).sum()), 2)
        else:
            affect_state_entropy = np.nan

        # -------------------------------------------------
        # per-dimension distribution summaries
        # -------------------------------------------------
        valence_mean = float(np.mean(valence_values))
        arousal_mean = float(np.mean(arousal_values))

        valence_sd = round(float(np.std(valence_values, ddof=1)), 2) if n_ratings > 1 else np.nan
        arousal_sd = round(float(np.std(arousal_values, ddof=1)), 2) if n_ratings > 1 else np.nan

        valence_skew = round(float(pd.Series(valence_values).skew()), 2) if n_ratings >= 3 else np.nan
        arousal_skew = round(float(pd.Series(arousal_values).skew()), 2) if n_ratings >= 3 else np.nan

        # -------------------------------------------------
        # dispersion: mean distance to centroid
        # -------------------------------------------------
        dev = affect_points - np.array([valence_mean, arousal_mean])
        distances = np.linalg.norm(dev, axis=1)
        dispersion_mean_distance = float(distances.mean())

        if n_ratings >= 2:
            # Pairwise Euclidean distances over all i<j (length n(n-1)/2)
            pw = pdist(affect_points, metric="euclidean")
            pairwise_mean_distance = float(pw.mean())

            # RMS pairwise distance (sqrt of mean squared pairwise distance)
            pw2 = pw ** 2
            pairwise_mean_sq_distance = float(pw2.mean())
            pairwise_rms_distance = float(np.sqrt(pairwise_mean_sq_distance))
        else:
            pairwise_mean_distance = np.nan
            pairwise_mean_sq_distance = np.nan
            pairwise_rms_distance = np.nan

        # A. Anisotropy (Structure of Disagreement)
        if n_ratings > 1:
            cov_matrix = np.cov(affect_points.T)
            # Use eigh because the covariance matrix is symmetric
            eigenvalues = np.linalg.eigvalsh(cov_matrix)
            l1, l2 = np.sort(eigenvalues)[::-1]
            # AI ranges from 0 (isotropic/circle) to 1 (linear)
            anisotropy_index = (l1 - l2) / (l1 + l2 + 1e-9)
        else:
            anisotropy_index = 0.0

        # -------------------------------------------------
        # 4. Polarization Index (Structural Bimodality)
        # -------------------------------------------------
        polarization_index = 0.0

        MIN_SEPARATION = 0.5
        BIC_MARGIN = 6.0  # 2-comp must beat 1-comp by >2.0 bits
        MIN_N = 15

        if n_ratings >= MIN_N:
            try:
                # 1. Fit 1-Component Model (Baseline)
                gmm1 = GaussianMixture(n_components=1, n_init=10, random_state=42,
                                       reg_covar=1e-4).fit(affect_points)
                bic1 = gmm1.bic(affect_points)

                # 2. Fit 2-Component Model (Bimodal Hypothesis)
                gmm2 = GaussianMixture(n_components=2, n_init=10, random_state=42,
                                       reg_covar=1e-4).fit(affect_points)
                bic2 = gmm2.bic(affect_points)

                # Calculate metrics
                delta_bic = bic1 - bic2
                mu1 = gmm2.means_[0]
                mu2 = gmm2.means_[1]
                dist = np.linalg.norm(mu1 - mu2)

                polarization_index = float(dist)

                # 3. Robust Decision Logic - ONLY assign if thresholds pass
                #if delta_bic >= BIC_MARGIN and dist >= MIN_SEPARATION:
                #    polarization_index = float(dist)
                #else:
                #    polarization_index = 0.0

            except Exception as e:
                log.warning(f"Video {video_id}: GMM fitting failed - {e}")
                polarization_index = np.nan
                delta_bic = np.nan
        else:
            # Not enough data
            polarization_index = np.nan

        # C. Distinctiveness Metric (Energy Statistic)
        # A simpler implementation of earth mover's distance.
        if n_ratings >= 5:
            d_xy = cdist(affect_points, global_sample, metric='euclidean').mean()
            d_xx = cdist(affect_points, affect_points, metric='euclidean').mean()
            dist_distinctiveness = 2 * d_xy - d_xx - d_yy
        else:
            dist_distinctiveness = np.nan

        # -------------------------------------------------
        # covariance matrix elements
        # -------------------------------------------------
        if n_ratings > 1:
            cov_mat = np.cov(affect_points.T, ddof=1)
            var_valence = float(cov_mat[0, 0])
            var_arousal = float(cov_mat[1, 1])
            cov_valence_arousal = float(cov_mat[0, 1])
            cov_trace = var_valence + var_arousal

            valence_arousal_pearson_r = float(
                np.corrcoef(valence_values, arousal_values)[0, 1]
            )
            dispersion_area_cov = round(float(np.linalg.det(cov_mat)), 6)
        else:
            var_valence, var_arousal = np.nan, np.nan
            cov_valence_arousal, cov_trace = np.nan, np.nan
            valence_arousal_pearson_r, dispersion_area_cov = np.nan, np.nan

        # -------------------------------------------------
        # Correlations (for sanity checks)
        # -------------------------------------------------
        va_rho = np.nan
        if n_ratings >= 3:
            va_rho, _ = spearmanr(valence_values, arousal_values)

        valence_oe_rho, arousal_oe_rho = np.nan, np.nan
        corr_df = video_df[[c.VALENCE, c.AROUSAL, c.OE]].dropna()
        if len(corr_df) >= 3:
            corr_oe_num = pd.Categorical(
                corr_df[c.OE], categories=c.OE_ORDER, ordered=True
            ).codes.astype(float)
            corr_oe_num[corr_oe_num < 0] = np.nan

            if np.isfinite(corr_oe_num).sum() >= 3:
                valence_oe_rho, _ = spearmanr(corr_df[c.VALENCE].to_numpy(float), corr_oe_num)
                arousal_oe_rho, _ = spearmanr(corr_df[c.AROUSAL].to_numpy(float), corr_oe_num)

        results.append([
            video_id,
            round(valence_mean, 2),
            round(valence_sd, 2),
            round(valence_skew, 2),
            round(arousal_mean, 2),
            round(arousal_sd, 2),
            round(arousal_skew, 2),
            round(var_valence, 2),
            round(var_arousal, 2),
            round(cov_valence_arousal, 2),
            round(cov_trace, 2),
            round(polarization_index, 2),
            round(delta_bic, 2),
            round(dist_distinctiveness, 2),
            round(anisotropy_index, 2),
            round(valence_arousal_pearson_r, 2),
            round(dispersion_mean_distance, 2),
            round(pairwise_mean_distance, 2),
            round(pairwise_rms_distance, 2),
            round(pairwise_mean_sq_distance, 6),
            round(dispersion_area_cov, 2),
            round(va_rho, 2),
            round(oe_mean, 2),
            oe_mode,
            round(valence_oe_rho, 2),
            round(arousal_oe_rho, 2),
            round(affect_state_entropy, 2),
            dominant_quadrant,
            *state_counts
        ])

    cols = [
        c.VIDEO_ID_COL,
        "valence_mean",
        "valence_sd",
        "valence_skewness",
        "arousal_mean",
        "arousal_sd",
        "arousal_skewness",
        "valence_variance",
        "arousal_variance",
        "valence_arousal_covariance",
        "covariance_trace",
        "polarization_index",
        "delta_bic",
        "dist_distinctiveness",
        "anisotropy_index",
        "valence_arousal_pearson_r",
        "dispersion_mean_distance",
        "pairwise_mean_distance",
        "pairwise_rms_distance",
        "pairwise_mean_sq_distance",
        "dispersion_area_covariance",
        "valence_arousal_spearman_rho",
        "oe_mean",
        "oe_mode",
        "valence_oe_spearman_rho",
        "arousal_oe_spearman_rho",
        "affect_state_entropy",
        "dominant_quadrant",
        *state_cols
    ]

    results = pd.DataFrame(results, columns=cols)

    if output_path is not None:
        results.to_csv(output_path, index=False)

    return results


def calculate_video_level_scores_by_subgroup(
        long_df: pd.DataFrame,
        subgroup_col: str,
        min_participants: int = 15,
        output_path: Optional[Union[str, Path]] = None
) -> pd.DataFrame:

    results = []

    if subgroup_col not in long_df.columns:
        raise ValueError(f"{subgroup_col} not found in dataframe.")

    for level, df_sub in long_df.groupby(subgroup_col):

        # ensure subgroup has enough participants
        n_participants = df_sub["participant_id"].nunique()
        if n_participants < min_participants:
            continue

        video_scores = calculate_video_level_scores(df_sub)

        video_scores = add_factor_counts_to_scores(
            scores_df=video_scores,
            survey_df=df_sub,
            label_cols=c.LABEL_COLS,
            video_col=c.VIDEO_ID_COL,
            pf_col=c.PF,
            nf_col=c.NF
        )

        video_scores, _ = pf_nf_disagreement_analysis(
            video_level_scores=video_scores,
            label_cols=c.LABEL_COLS,
            out_csv_path=None  # IMPORTANT: don't overwrite global file
        )

        video_scores[subgroup_col] = level
        video_scores["n_participants"] = n_participants
        results.append(video_scores)

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)

    if output_path is not None:
        out.to_csv(output_path, index=False)

    return out


def bayesian_gmm(
    coords: np.ndarray,
    max_components: int = 5,
    weight_concentration_prior: float = 0.1,
    active_weight_thresh: float = 0.01,
    active_min_count: int = 20,
) -> Dict:

    n_samples = len(coords)
    if n_samples < 4:
        return {"n_active_components": 0, "interpretation": "insufficient_data"}

    bgmm = BayesianGaussianMixture(
        n_components=max_components,
        weight_concentration_prior=weight_concentration_prior,
        covariance_type="full",
        max_iter=200,
        random_state=42,
        n_init=5,
    )
    bgmm.fit(coords)

    weights = bgmm.weights_
    responsibilities = bgmm.predict_proba(coords)
    assignments = bgmm.predict(coords)

    counts = np.bincount(assignments, minlength=max_components)
    active_mask = (weights >= active_weight_thresh) & (counts >= active_min_count)
    active_ids = np.where(active_mask)[0]
    n_active = int(len(active_ids))

    if n_active == 0:
        return {"n_active_components": 0, "interpretation": "degenerate"}

    # Remap assignments to active index space: active -> 0..n_active-1, inactive -> -1
    remap = {old: new for new, old in enumerate(active_ids)}
    assignments_remapped = np.array([remap.get(a, -1) for a in assignments], dtype=int)

    active_weights = weights[active_ids]
    active_means = bgmm.means_[active_ids]
    active_covs = bgmm.covariances_[active_ids]

    # Separation between "camps"
    if n_active > 1:
        mean_separation = float(np.mean(pdist(active_means, metric="euclidean")))
        within_spread = float(np.mean([np.sqrt(np.trace(cov)) for cov in active_covs]))
        separation_ratio = mean_separation / (within_spread + 1e-9)
    else:
        mean_separation = 0.0
        separation_ratio = 0.0

    # Interpretation (your logic, unchanged)
    if n_active == 1:
        interp = "strong_consensus"
    elif n_active == 2:
        interp = "clear_polarization" if separation_ratio > 1.5 else "weak_bimodality"
    elif n_active == 3:
        interp = "tripartite_camps"
    else:
        interp = "fragmented_opinions"

    return {
        "n_active_components": n_active,
        "component_weights": [round(float(w), 3) for w in active_weights],
        "component_means": [
            {"valence": round(float(m[0]), 3), "arousal": round(float(m[1]), 3)}
            for m in active_means
        ],
        "mean_separation": round(mean_separation, 3),
        "separation_ratio": round(separation_ratio, 2),
        "assignments": assignments_remapped.tolist(),
        "max_responsibility": round(float(responsibilities.max(axis=1).mean()), 3),
        "interpretation": interp,
    }


def add_factor_counts_to_scores(
        scores_df,
        survey_df,
        label_cols: Sequence[str],
        video_col: str = c.VIDEO_ID_COL,
        pf_col: str = c.PF,
        nf_col: str = c.NF,
   ) -> pd.DataFrame:

    label_counts = build_factor_counts_by_video(
        survey_df,
        label_cols=label_cols,
        video_col=video_col,
        pf_col=pf_col,
        nf_col=nf_col
    )

    # Merge label counts with video-level scores, filling missing values with 0
    video_level_scores = scores_df.merge(label_counts, on=c.VIDEO_ID_COL, how="left")
    video_level_scores[c.LABEL_COLS] = video_level_scores[c.LABEL_COLS].fillna(0).astype(int)
    return video_level_scores


def build_factor_counts_by_video(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    video_col: str = c.VIDEO_ID_COL,
    pf_col: str = c.PF,
    nf_col: str = c.NF,
) -> pd.DataFrame:
    """
    One row per video; columns in `label_cols` filled with counts (ints).
    Assumes PF/NF entries are strings like "A;B;C;" (trailing ';' ok).
    Missing labels remain 0 (never NaN).
    """

    def parse_labels(x):
        if pd.isna(x):
            return []
        parts = [p.strip() for p in re.split(r"[;,]", str(x))]
        return [p for p in parts if p]

    vids = pd.DataFrame({video_col: sorted(df[video_col].dropna().unique())})
    out = vids.copy()
    for col in label_cols:
        out[col] = 0

    def accumulate(source_col: str, prefix: str):
        t = df[[video_col, source_col]].dropna().copy()
        t["label"] = t[source_col].apply(parse_labels)
        t = t.explode("label")
        t = t[t["label"].notna() & (t["label"] != "")]
        if t.empty:
            return

        counts = (t.groupby([video_col, "label"]).size()
                    .rename("count")
                    .reset_index())
        counts["colname"] = prefix + ": " + counts["label"].astype(str)

        for _, r in counts.iterrows():
            colname = r["colname"]
            if colname not in out.columns:
                continue
            out.loc[out[video_col] == r[video_col], colname] = int(r["count"])

    accumulate(pf_col, "PF")
    accumulate(nf_col, "NF")

    return out


def shannon_entropy_from_counts(counts: np.ndarray) -> float:
    """Shannon entropy for a vector of nonnegative counts."""
    counts = np.asarray(counts, dtype=float)
    s = counts.sum()
    if s <= 0:
        return 0.0
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def pf_nf_disagreement_analysis(
            video_level_scores: pd.DataFrame,
            label_cols: List[str],
            out_csv_path=None,
            n_hi_lo: int = 10,
):
    """
    RQ2 analysis: relate affect-disagreement structure to PF/NF cue structure.
    """

    d = video_level_scores.copy()

    # ----------------------------
    # 0) Identify PF / NF columns
    # ----------------------------
    pf_cols = [c for c in label_cols if c.startswith("PF:")]
    nf_cols = [c for c in label_cols if c.startswith("NF:")]

    for col in pf_cols + nf_cols:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0).astype(int)
        else:
            d[col] = 0

    # ----------------------------
    # 1) Disagreement flags
    # ----------------------------
    d["flag_high_dispersion"] = (
        d["dispersion_mean_distance"] > d["dispersion_mean_distance"].median()
        if "dispersion_mean_distance" in d.columns else False
    )

    d["flag_high_polarization"] = (
        d["anisotropy_index"] > d["anisotropy_index"].median()
        if "anisotropy_index" in d.columns else False
    )

    d["flag_multimodal"] = (
        pd.to_numeric(d["polarization_index"], errors="coerce").fillna(0) > 0.0
        if "polarization_index" in d.columns else False
    )

    # ----------------------------
    # 2) PF/NF STRUCTURE metrics
    # ----------------------------
    d["pf_total_count"] = d[pf_cols].sum(axis=1)
    d["nf_total_count"] = d[nf_cols].sum(axis=1)

    # mixed signals (both PF and NF present)
    d["pf_nf_overlap_count"] = d[["pf_total_count", "nf_total_count"]].min(axis=1)

    # how diverse cue selection is
    d["pf_label_entropy"] = d[pf_cols].apply(lambda r: shannon_entropy_from_counts(r.values), axis=1)
    d["nf_label_entropy"] = d[nf_cols].apply(lambda r: shannon_entropy_from_counts(r.values), axis=1)
    d["pf_nf_label_entropy"] = d[pf_cols + nf_cols].apply(
        lambda r: shannon_entropy_from_counts(r.values), axis=1
    )

    # ----------------------------
    # 3) Hi vs Lo dispersion contrasts
    # ----------------------------
    if "dispersion_mean_distance" in d.columns:
        n = min(n_hi_lo, len(d))
        hi = d.nlargest(n, "dispersion_mean_distance")
        lo = d.nsmallest(n, "dispersion_mean_distance")

        hi_lo_diff = hi[label_cols].mean() - lo[label_cols].mean()
        hi_lo_diff = hi_lo_diff.sort_values(ascending=False)
    else:
        hi_lo_diff = pd.Series()

    # ----------------------------
    # 4) Save enriched dataset
    # ----------------------------
    if out_csv_path is not None:
        d.to_csv(out_csv_path, index=False)

    return d, hi_lo_diff


def compute_video_reliability(long_df, n_splits=10):
    """
    Test whether disagreement is systematic or noise.
    Split participants randomly, compute metrics for each split,
    correlate across splits.
    """
    results = []
    for split in range(n_splits):
        # Random split
        participants = long_df['participant_id'].unique()
        np.random.shuffle(participants)
        split_point = len(participants) // 2

        split_A = participants[:split_point]
        split_B = participants[split_point:]

        # Compute metrics for each video in each split
        metrics_A = calculate_video_level_scores(
            long_df[long_df['participant_id'].isin(split_A)]
        )
        metrics_B = calculate_video_level_scores(
            long_df[long_df['participant_id'].isin(split_B)]
        )

        # Correlate
        for metric in ['valence_mean', 'arousal_mean', 'dispersion_mean_distance', 'anisotropy_index', "affect_state_entropy"]:
            merged = metrics_A.merge(metrics_B, on='video_id', suffixes=('_A', '_B'))
            r = spearmanr(merged[f'{metric}_A'], merged[f'{metric}_B'])[0]
            results.append({'split': split, 'metric': metric, 'correlation': r})

    return pd.DataFrame(results)


def fit_lmm_and_extract_metrics(formula, df_model, baseline_aic, baseline_bic):
    model = smf.mixedlm(
        formula,
        df_model,
        groups=df_model[c.PARTICIPANT_ID],
    )

    fit = model.fit(reml=False, method='powell')

    interaction_key = [k for k in fit.params.index if ':' in k][0]
    metric_key = [k for k in fit.params.index if 'z_' in k and ':' not in k][0]

    return {
        "AIC": fit.aic,
        "BIC": fit.bic,
        "Delta_AIC": fit.aic - baseline_aic,
        "Delta_BIC": fit.bic - baseline_bic,
        "Interaction_Beta": fit.params[interaction_key],
        "Interaction_Pval": fit.pvalues[interaction_key],
        "Main_Effect_Beta": fit.params[metric_key],
        "Main_Effect_Pval": fit.pvalues[metric_key],
    }


def prepare_lmm_dataframe(video_level_scores, survey_results_df, metrics):
    clip_features = video_level_scores[[c.VIDEO_ID_COL] + metrics].copy()

    for col in metrics:
        clip_features[f'z_{col}'] = (clip_features[col] - clip_features[col].mean()) / clip_features[col].std(ddof=0)

    df_model = survey_results_df.merge(clip_features, on=c.VIDEO_ID_COL, how='left')

    df_model['z_valence'] = (df_model[c.VALENCE] - df_model[c.VALENCE].mean()) / df_model[c.VALENCE].std(ddof=0)
    df_model['z_arousal'] = (df_model[c.AROUSAL] - df_model[c.AROUSAL].mean()) / df_model[c.AROUSAL].std(ddof=0)

    return df_model



























































#-------------------------------------------------------------
# Additional utility functions
#-------------------------------------------------------------


def filter_by_group_size(
        df: pd.DataFrame,
        group_col: str,
        id_col: str,
        min_size: int = 5
) -> pd.DataFrame:
    """
    Filter a DataFrame to only include groups with a minimum number of unique IDs.
    :param df: DataFrame to filter.
    :param group_col: Column name to group by.
    :param id_col: Column name containing unique IDs.
    :param min_size: Minimum number of unique IDs required to retain a group.
    :return: Filtered DataFrame.
    """
    counts = df.groupby(group_col)[id_col].nunique()
    valid_categories = counts[counts >= min_size].index

    # Filter the original DataFrame and return a copy
    return df[df[group_col].isin(valid_categories)].copy()


def aggregate_video_level_geometry(
        gpx_paths: List[Path]
) -> gpd.GeoDataFrame:
    """
    Aggregate video-level geometries from GPX files.
    :param gpx_paths: List of file paths to GPX files.
    :return: GeoDataFrame with aggregated geometries and slope information.
    """
    results_list = []
    dz_list = []

    for gpx_filepath in gpx_paths:
        gdf = gpd.read_file(gpx_filepath).sort_values('time')
        file_id = os.path.splitext(os.path.basename(gpx_filepath))[0]
        geom = LineString(gdf.geometry.tolist())

        ele0 = pd.to_numeric(gdf['ele'].iloc[0], errors='coerce') if 'ele' in gdf.columns else np.nan
        ele1 = pd.to_numeric(gdf['ele'].iloc[-1], errors='coerce') if 'ele' in gdf.columns else np.nan
        dz = (ele1 - ele0) if pd.notna(ele0) and pd.notna(ele1) else np.nan
        dz_list.append(dz)

        results_list.append({
            'video_id': int(file_id.split('_')[1]),
            'video_name': file_id,
            'geometry': geom
        })

    video_geom = gpd.GeoDataFrame(results_list, crs=gdf.crs).to_crs(2056)
    video_geom['length'] = video_geom.length
    video_geom['slope'] = np.where(video_geom['length'] > 0, pd.Series(dz_list).values / video_geom['length'].values,
                                   np.nan)
    video_geom['index'] = video_geom.index
    return video_geom


def calculate_buffer(
        df: gpd.GeoDataFrame,
        buffer_size: float
) -> gpd.GeoDataFrame:
    """
    Calculate buffer for each geometry in the GeoDataFrame and return a new GeoDataFrame with the buffered geometries.
    :param df: GeoDataFrame to buffer.
    :param buffer_size: buffer size in the units of the GeoDataFrame's CRS.
    :return: GeoDataFrame with buffered geometries and their areas.
    """
    buff_df = df.copy()[['geometry', 'index']]
    buff_df['geometry'] = buff_df.buffer(buffer_size)
    buff_df['buff_area'] = buff_df.area

    return buff_df


def merge_spatial_share(
        buff_edges: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        target_col: str,
        divider_col: str,
        percent: bool = False,
        merge_col: str = "index",
) -> gpd.GeoDataFrame:
    """
    Merge spatial data with share calculation based on overlaps.
    :param buff_edges: GeoDataFrame with buffered edges.
    :param edges: original GeoDataFrame.
    :param spatial_data: GeoDataFrame with spatial data to merge.
    :param target_col: Name of the target column to store share values.
    :param divider_col: Name of the column to use as the denominator for share calculation.
    :param percent: If True, the share will be expressed as a percentage.
    :param merge_col: Name of the column to merge on (default is "index").
    :return: GeoDataFrame with merged data and share values.
    """
    overlaps = gpd.overlay(buff_edges, spatial_data, how="intersection", keep_geom_type=False)

    if overlaps.empty:
        edges[target_col] = 0
        return edges

    if edges.geometry.type.iloc[0] == 'LineString':
        overlaps['overlap'] = overlaps.geometry.length
    elif edges.geometry.type.iloc[0] == 'Polygon':
        overlaps['overlap'] = overlaps.geometry.area

    overlap_sums = overlaps.groupby(merge_col)['overlap'].sum().rename('overlap_sum')
    edges = edges.merge(overlap_sums, on=merge_col, how='left')

    edges[target_col] = (edges['overlap_sum'] / buff_edges[divider_col]).fillna(0)
    if percent:
        edges[target_col] *= 100

    edges.drop(columns=['overlap_sum'], inplace=True)

    return edges


def merge_spatial_count(
        buff_edges: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        target_col: str,
        agg_col: str = None,
        agg_func: Any = 'size',
        merge_col: str = "index",
) -> gpd.GeoDataFrame:
    """
    Merge spatial data with count aggregation based on overlaps.
    :param buff_edges: GeoDataFrame with buffered edges.
    :param edges: original GeoDataFrame.
    :param spatial_data: GeoDataFrame with spatial data to merge.
    :param target_col: Name of the target column to store count values.
    :param agg_col: Name of the column to aggregate (optional).
    :param agg_func: Aggregation function to apply (default is 'size').
    :param merge_col: Name of the column to merge on (default is "index").
    :return: GeoDataFrame with merged data and count values.
    """
    overlaps = gpd.overlay(buff_edges, spatial_data, how="intersection", keep_geom_type=False)

    if agg_func in ['size', 'sum', 'min', 'max']:
        aggregation = overlaps.groupby(merge_col)[agg_col].agg(agg_func) if agg_col else overlaps.groupby(
            merge_col).agg(agg_func)
    else:
        aggregation = overlaps.groupby(merge_col).apply(agg_func)

    aggregation_aligned = buff_edges[merge_col].map(aggregation).fillna(0)
    edges[target_col] = aggregation_aligned

    return edges


def merge_spatial_attribute(
        buff_edges: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        attribute_cols: Union[str, list] = None,
        target_cols: Union[str, list] = None,
        merge_col: str = "index",
        threshold: float = 20,
        divider_col: str = 'length'
) -> gpd.GeoDataFrame:
    """
    Merge spatial data with attributes based on overlap length or area.
    :param buff_edges: GeoDataFrame with buffered edges.
    :param edges: original GeoDataFrame.
    :param spatial_data: GeoDataFrame with spatial data to merge.
    :param attribute_cols: Name of the columns to extract attributes from spatial data.
    :param target_cols: Name of the target columns to store attributes.
    :param merge_col: Name of the column to merge on (default is "index").
    :param threshold: minimum percentage of overlap required to consider a match.
    :param divider_col: 'length' or 'area', determines how overlaps are calculated.
    :return: GeoDataFrame with merged data and attributes.
    """

    if attribute_cols is None:
        attribute_cols = [col for col in spatial_data.columns if col != 'geometry']
    elif not isinstance(attribute_cols, list):
        attribute_cols = [attribute_cols]
    if target_cols is None:
        target_cols = attribute_cols
    elif not isinstance(target_cols, list):
        target_cols = [target_cols]

    overlaps = gpd.overlay(buff_edges, spatial_data, how="intersection", keep_geom_type=False)

    if overlaps.empty:
        for col in target_cols: edges[col] = None
        return edges

    if divider_col == 'area':
        overlaps['overlap_size'] = overlaps.geometry.area
        denom = buff_edges[[merge_col, "geometry"]].copy()
        denom["orig_size"] = denom.geometry.area

    else:
        overlaps['overlap_size'] = overlaps.geometry.length
        denom = edges[[merge_col, "geometry"]].copy()
        denom["orig_size"] = denom.geometry.length

    overlaps = overlaps.merge(denom[[merge_col, "orig_size"]], on=merge_col, how="left")
    overlaps = overlaps[(overlaps["overlap_size"] / overlaps["orig_size"]) * 100 >= threshold]
    idx = overlaps.groupby(merge_col)['overlap_size'].idxmax()
    max_overlaps = overlaps.loc[idx]
    merge_cols = [merge_col] + attribute_cols
    edges = edges.merge(max_overlaps[merge_cols], on=merge_col, how="left")
    rename_dict = dict(zip(attribute_cols, target_cols))
    edges = edges.rename(columns=rename_dict)

    return edges


def merge_spatial_boolean(
        buff_edges: gpd.GeoDataFrame,
        edges: gpd.GeoDataFrame,
        spatial_data: gpd.GeoDataFrame,
        target_col: str,
        divider_col: str,
        merge_col: str = "index",
        threshold: float = 0
) -> gpd.GeoDataFrame:
    """
Merge spatial data with boolean condition based on overlap length or area.
    :param buff_edges: GeoDataFrame with buffered edges.
    :param edges: original GeoDataFrame.
    :param spatial_data: GeoDataFrame with spatial data to merge.
    :param target_col: Name of the target column to store boolean values.
    :param divider_col: Name of the column to use as the denominator for the threshold calculation.
    :param merge_col: Name of the column to merge on (default is "index").
    :param threshold: Threshold value to determine if the condition is met (default is 0).
    :return: GeoDataFrame with merged data and boolean condition.
    """
    overlaps = gpd.overlay(buff_edges, spatial_data, how="intersection", keep_geom_type=False)
    overlaps['overlap_length'] = overlaps.geometry.length
    overlap_sums = overlaps.groupby(merge_col)['overlap_length'].sum().reset_index(name='overlap_length_sum')
    edges = edges.merge(overlap_sums, on=merge_col, how='left')
    edges['overlap_length_sum'] = edges['overlap_length_sum'].fillna(0)
    edges[target_col] = (edges['overlap_length_sum'] / edges[divider_col]) * 100 > threshold
    edges = edges.drop(columns=['overlap_length_sum'])

    return edges


def classify_cycleway_row(gdf):
    """
    Classify cycleway types using an apply function with explicit if/elif/else logic.
    :param gdf: GeoDataFrame containing cycleway-related columns.
    :return: GeoDataFrame with an additional 'infra_class' column.
    """
    gdf_classified = gdf.copy()

    numeric_cols = ['fuss', 'velo', 'veloweg']
    for col in numeric_cols:
        gdf_classified[col] = pd.to_numeric(gdf_classified[col], errors='coerce')
    gdf_classified['velostreifen'] = gdf_classified['velostreifen'].astype(str)

    def _classify_single_row(row):
        if row[['veloweg', 'velostreifen', 'velo', 'fuss']].isna().any():
            return 'shared_path'

        # Priority 1: Separated, exclusive bike lane
        if row['veloweg'] == 1 and row['fuss'] == 0:
            return 'shared_path'

        # Priority 2: Separated, shared with pedestrians
        elif row['veloweg'] == 1 and row['fuss'] == 1:
            return 'shared_path'

        # Priority 3: On-road advisory bike lane
        elif row['velostreifen'] in ['FT', 'TF', 'BOTH', '1']:
            return 'advisory'

        # Priority 4: Sidewalk where bikes are allowed
        elif row['fuss'] == 1 and row['veloweg'] == 0 and row['velostreifen'] == '0' and row['velo'] == 1:
            return 'shared_path'

        # Priority 5: Road with no infrastructure
        elif row['velo'] == 1 and row['veloweg'] == 0 and row['velostreifen'] == '0':
            return 'no_bike_infra'

        # Priority 6: Bikes are explicitly prohibited
        elif row['velo'] == 0:
            return 'no_bike_infra'

        return 'no_bike_infra'

    gdf_classified['bike_infra_type'] = gdf_classified.apply(_classify_single_row, axis=1)
    gdf_classified['bike_infra_type_numeric'] = gdf_classified['bike_infra_type'].map({'shared_path': 1, 'advisory': 0.5, 'no_bike_infra': 0})

    return gdf_classified


def enrich_with_spatial_data(
        video_geom: gpd.GeoDataFrame,
        config: dict
) -> gpd.GeoDataFrame:
    """
    Enriches video geometries with various spatial datasets within a single function.

    This function processes and merges multiple data sources sequentially, adding
    new attributes to the video geometry data at each step.

    :param video_geom: GeoDataFrame containing video geometries.
    :param config: Configuration dictionary with file paths.
    :return: Enriched GeoDataFrame with additional spatial attributes.
    """
    # --- 0. SETUP ---
    log.info("Starting spatial data enrichment process...")
    buffer_geom = calculate_buffer(video_geom, c.BUFFER)

    # Centralize all file path definitions from the config
    paths = {
        "manual_labels": Path(config["filenames"]["manual_labelling_file"]),
        "negar_features": Path(config["filenames"]["negar_features_file"]),
        "bike_network": Path(config["filenames"]["bike_network_file"]),
        "traffic_volume": Path(config["filenames"]["traffic_volume_file"]),
        "road_network": Path(config["filenames"]["road_network_file"]),
        "side_parking": Path(config["filenames"]["side_parking_file"]),
        "tree_canopy": Path(config["filenames"]["tree_canopy_file"]),
        "greenery_share_gis": Path(config["filenames"]["greenery_file"]),
    }

    # Work on a copy to avoid modifying the original DataFrame
    enriched_geom = video_geom.copy()

    # --- 2. MERGE AND CLASSIFY BIKE NETWORK INFRASTRUCTURE ---
    log.info("Enriching with bike network data...")
    bike_network = gpd.read_file(paths["bike_network"], layer='taz_mm.tbl_routennetz')
    ped_only_condition = (
        (bike_network['fuss'] == 1) & (bike_network['veloweg'] == 0) &
        (bike_network['velostreifen'] == '0') & (bike_network['velo'] == 0)
    )
    bike_network = bike_network.loc[~ped_only_condition].copy()
    bike_network_cols = ['velo', 'velostreifen', 'veloweg', 'fuss']
    for attr in bike_network_cols:
        enriched_geom = merge_spatial_attribute(buffer_geom, enriched_geom, bike_network, attr)
    enriched_geom = classify_cycleway_row(enriched_geom)
    enriched_geom = enriched_geom.drop(columns=bike_network_cols, errors='ignore')

    # --- 3.1 MERGE MANUAL DATA ---
    log.info("Enriching with manually labelled data...")
    manual_labels = pd.read_excel(paths["manual_labels"])
    enriched_geom = enriched_geom.merge(manual_labels, on=c.VIDEO_ID_COL, how='left')
    enriched_geom["motor_vehicle_overtakes_presence"] = enriched_geom["motor_vehicle_overtakes_count"] > 0

    # --- 3.2 MERGE NEGAR FEATURES ---
    log.info("Enriching with Negar features data...")
    negar_features = pd.read_csv(paths["negar_features"])
    enriched_geom = enriched_geom.merge(negar_features, on=c.VIDEO_ID_COL, how='left')

    # --- 4. MERGE TRAFFIC VOLUME ---
    log.info("Enriching with traffic volume data...")
    traffic_volume = gpd.read_file(paths["traffic_volume"])
    enriched_geom = merge_spatial_attribute(buffer_geom, enriched_geom, traffic_volume, 'AADT_all_veh', 'traffic_volume')
    enriched_geom['traffic_volume'] = enriched_geom['traffic_volume'].fillna(0)

    # --- 5. MERGE ROAD NETWORK FEATURES (SPEED, TRAMS, ONE-WAY) ---
    log.info("Enriching with road network features (speed, trams, one-way)...")
    road_network_path = paths["road_network"]

    # Speed Limits: Explicitly read only the necessary column
    speed_limits = gpd.read_file(road_network_path, layer='vas.vas_tempo_ist_event', columns=['temporegime_technical'])
    speed_limits['temporegime_technical'] = pd.to_numeric(
        speed_limits['temporegime_technical'].str.extract(r'(\d+)', expand=False), errors='coerce'
    ).astype(int)
    enriched_geom = merge_spatial_attribute(buffer_geom, enriched_geom, speed_limits, 'temporegime_technical', 'motorized_traffic_speed_kmh')

    # Tram Lanes: Explicitly read only the necessary column
    tram_lanes = gpd.read_file(road_network_path, layer='vas.vas_verkehrstraeger_event', columns=['tram_vorhanden'])
    tram_lanes['tram_vorhanden'] = tram_lanes['tram_vorhanden'].map({'ja': True, 'nein': False})
    enriched_geom = merge_spatial_attribute(buffer_geom, enriched_geom, tram_lanes, 'tram_vorhanden', 'tram_lane_presence')

    # One-Way Traffic: Read no attribute columns, only geometry
    one_way_lanes = gpd.read_file(road_network_path, layer='vas.vas_einbahn_ist_event', columns=[])
    enriched_geom = merge_spatial_boolean(buffer_geom, enriched_geom, one_way_lanes, 'one_way', 'length', threshold=20)

    # Fill NaNs and explicitly set the final data type for each column individually.
    enriched_geom['motorized_traffic_speed_kmh'] = enriched_geom['motorized_traffic_speed_kmh'].fillna(0).astype(int)
    enriched_geom['tram_lane_presence'] = enriched_geom['tram_lane_presence'].fillna(False).astype(bool)

    # --- 6. MERGE SIDE PARKING ---
    log.info("Enriching with side parking data...")
    side_parking = gpd.read_file(paths["side_parking"], layer='taz.view_pp_ogd')
    enriched_geom = merge_spatial_count(buffer_geom, enriched_geom, side_parking, 'side_parking_count', agg_func='size')
    enriched_geom['side_parking_presence'] = enriched_geom['side_parking_count'] > 0

    # --- 7. MERGE GREENERY DATA ---
    log.info("Enriching with greenery data...")
    tree_canopy = gpd.read_file(paths["tree_canopy"])
    enriched_geom = merge_spatial_share(buffer_geom, enriched_geom, tree_canopy, 'tree_canopy_share', 'buff_area', percent=True)

    greenery = gpd.read_file(paths["greenery_share_gis"])
    enriched_geom = merge_spatial_share(buffer_geom, enriched_geom, greenery, 'greenery_share_gis', 'buff_area',
                                        percent=True)

    log.info("Spatial data enrichment process finished successfully.")
    return enriched_geom


def prepare_categorical_predictors(df: pd.DataFrame, ordinal_map: dict) -> pd.DataFrame:
    """
    Converts specified ordinal columns into numeric codes and centered versions.
    :param df: The input DataFrame.
    :param ordinal_map: A dictionary mapping column names to their category order list.
                        e.g., {'OE': ['Very negative', ..., 'Very positive']}
    :return: DataFrame with new columns:
             - '{col}_numeric': Integer codes (0, 1, 2, ...).
             - '{col}_centered': Codes centered around zero.
    """
    df_out = df.copy()
    for col, order in ordinal_map.items():
        if col not in df_out.columns:
            continue  # Skip if the column isn't in the dataframe

        # Create the ordered categorical type
        cat_type = pd.CategoricalDtype(categories=order, ordered=True)
        df_out[col] = df_out[col].astype(cat_type)

        # Create a numeric version (useful for correlations, etc.)
        df_out[f'{col}_numeric'] = df_out[col].cat.codes

        # Create a centered version for LMMs
        # The center is the midpoint of the code range (e.g., for 7 levels with codes 0-6, the center is 3)
        center_point = len(order) // 2
        df_out[f'{col}_centered'] = df_out[f'{col}_numeric'] - center_point

    return df_out


def prepare_combined_scenario_df(
        df_positive: pd.DataFrame,
        df_negative: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare a combined DataFrame for both positive and negative scenarios.
    :param df_positive: DataFrame for the positive scenario.
    :param df_negative: DataFrame for the negative scenario.
    :return: Combined DataFrame with standardized columns and additional useful columns.
    """
    # Create copies to avoid modifying original dataframes
    df_pos = df_positive.copy()
    df_neg = df_negative.copy()

    # Standardize column names for combining
    df_pos['scenario'] = 'Positive'
    df_pos['spoiler_position'] = df_pos['NB_position']

    df_neg['scenario'] = 'Negative'
    df_neg['spoiler_position'] = df_neg['B_position']

    # Combine and set appropriate data types for modeling
    df_combined = pd.concat([df_pos, df_neg])
    df_combined['scenario'] = pd.Categorical(df_combined['scenario'])
    df_combined['spoiler_position'] = pd.Categorical(df_combined['spoiler_position'])

    # Add the quantity of spoilers in the sequence. Counting B would result in the same but inverse reults.
    df_combined['NB_count'] = df_combined['sequence_list'].apply(lambda seq: seq.count('NB'))

    return df_combined