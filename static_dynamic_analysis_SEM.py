import utils.helper_functions
import utils.processing_utils
from scipy.stats import zscore
from semopy import Model, semplot
from semopy.stats import calc_stats
from sklearn.decomposition import PCA
import configparser
import utils.helper_functions
import utils.processing_utils
from utils.physiological_data_utils import map_physiological_segments_to_videos
import matplotlib
import lingam
from lingam.utils import make_dot, evaluate_model_fit
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from statsmodels.formula.api import ols
import constants as c
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.feature_selection import mutual_info_regression
import logging
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Suppress sklearn future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# -------------------------------------------------------------------
# UTILS: TRANSFORMATIONS
# -------------------------------------------------------------------

def extract_interaction_wald(model, interaction_name, target_variable):
    """
    Extracts Wald stats for a specific target ~ interaction path.
    Usage: extract_interaction_wald(model, 'Infra_Dynamic_Interaction', 'valence')
    """
    est_df = model.inspect()

    # Filter by BOTH predictor (rval) AND target (lval)
    row = est_df[
        (est_df["lval"] == target_variable) &
        (est_df["op"] == "~") &
        (est_df["rval"] == interaction_name)
    ]

    if row.empty:
        return None

    stats = row.iloc[0]

    return {
        "Estimate": stats["Estimate"],
        "SE": stats["Std. Err"],
        "z": stats["z-value"],
        "p": stats["p-value"]
    }


def plot_valence_arousal_scatter(df, output_dir):

    stats = df.groupby(c.VIDEO_ID_COL).agg(
        mean_valence=("valence", "mean"),
        mean_arousal=("arousal", "mean"),
        sem_valence=("valence", "sem"),
        sem_arousal=("arousal", "sem")
    ).reset_index()

    stats["color_metric"] = stats["mean_arousal"] - stats["mean_valence"]

    # -----------------------------
    # FIGURE SETTINGS
    # -----------------------------
    fig, ax = plt.subplots(figsize=(9, 9))

    scatter = ax.scatter(
        stats["mean_valence"],
        stats["mean_arousal"],
        c=stats["color_metric"],
        cmap="vlag",
        s=230,
        edgecolor="black",
        linewidth=1.3
    )

    # Error bars (lighter)
    for i in range(len(stats)):
        ax.errorbar(
            stats.loc[i, "mean_valence"],
            stats.loc[i, "mean_arousal"],
            xerr=stats.loc[i, "sem_valence"],
            yerr=stats.loc[i, "sem_arousal"],
            fmt="none",
            ecolor="gray",
            elinewidth=1,
            capsize=3,
            alpha=0.5
        )

    # Text labels (offset)
    for i, vid in enumerate(stats[c.VIDEO_ID_COL]):
        ax.text(
            stats.loc[i, "mean_valence"] + 0.02,
            stats.loc[i, "mean_arousal"] + 0.02,
            str(vid),
            fontsize=10
        )

    # -----------------------------
    # AXIS RANGE AND GRID
    # -----------------------------
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    ax.set_xticks(np.arange(-1, 1.1, 0.2))
    ax.set_yticks(np.arange(-1, 1.1, 0.2))
    ax.tick_params(labelsize=14)

    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_aspect("equal", adjustable="box")

    ax.axhline(0, color="gray", linestyle="--", alpha=0.6)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.6)

    # -----------------------------
    # LABELS AND TITLE
    # -----------------------------
    ax.set_xlabel("Mean Valence", fontsize=18)
    ax.set_ylabel("Mean Arousal", fontsize=18)
    ax.set_title("Avg. Video Ratings in Valence–Arousal Space", fontsize=18)

    # -----------------------------
    # PROPER COLORBAR MATCHING GRID HEIGHT
    # -----------------------------
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label("Affect Grid Scale", fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    fig.tight_layout()
    out_path = Path(output_dir, "video_mean_valence_arousal_scatter.png")
    plt.savefig(out_path, dpi=300)
    plt.close()


def mutual_info(df, cols):
    log.info("--- Calculating Mutual Information ---")
    X = df[cols].fillna(0)
    y_val = df["valence"].fillna(0)
    y_ar = df["arousal"].fillna(0)

    MI_val = mutual_info_regression(X, y_val, random_state=42)
    MI_ar = mutual_info_regression(X, y_ar, random_state=42)

    return MI_val, MI_ar


def check_vif(df, features):
    log.info("--- Calculating Variance Inflation Factor (VIF) ---")
    X = df[features].dropna()
    X = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                       for i in range(len(X.columns))]
    return vif_data


def assess_lingam_model_fit(model, df, cols, output_dir, scenario_name):
    log.info(f"--- Assessing LiNGAM Fit ({scenario_name}) ---")

    X = df[cols].dropna()
    # Force Data to strictly 2D float NumPy array
    X_np = np.ascontiguousarray(X.values, dtype=np.float64)

    # ------------------------------------------------------
    # 1. Residual Independence (The Critical Test)
    # ------------------------------------------------------
    try:
        p_mat = model.get_error_independence_p_values(X_np)
        p_mat = np.asarray(p_mat, dtype=float)
        p_mat = p_mat.reshape(len(cols), len(cols))
        np.fill_diagonal(p_mat, 1.0)

        min_p = p_mat.min()
        pd.DataFrame(p_mat, index=cols, columns=cols).to_csv(
            output_dir / f"lingam_independence_{scenario_name}.csv"
        )

        status = "PASS" if min_p > 0.05 else "VIOLATION (Confounders likely)"
        print(f"[LiNGAM Check: {scenario_name}] Residual Independence: {status} (min p={min_p:.4f})")

    except Exception as e:
        log.warning(f"Residual independence check failed: {e}")

    # ------------------------------------------------------
    # 2. Global Fit Statistics (The Fix)
    # ------------------------------------------------------
    try:
        if hasattr(model, "adjacency_matrix_"):
            adj = model.adjacency_matrix_

            if hasattr(adj, "values"):
                adj = adj.values

            adj = np.asarray(adj, dtype=np.float64)
            if adj.ndim > 2:
                adj = adj[0]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stats = evaluate_model_fit(adj, X_np)

            pd.DataFrame([stats]).to_csv(
                output_dir / f"lingam_fit_stats_{scenario_name}.csv", index=False
            )
            print("  Global Fit Stats: Saved")

    except Exception as e:
        print(f"  Global Fit Stats: Skipped ({e})")


def check_lingam_data_assumptions(df, cols, output_dir, scenario_name=""):
    log.info(f"--- Checking Assumptions ({scenario_name}) ---")
    results = []

    df_clean = df[cols].dropna()

    for target in cols:
        predictors = [c for c in cols if c != target]
        if not predictors:
            continue

        # ------------------------------------------------------
        # 1. Non-Gaussianity (Shapiro-Wilk)
        # ------------------------------------------------------
        y = df_clean[target]
        sw_stat, sw_p = stats.shapiro(y)
        non_gaussian = sw_p < 0.05

        # ------------------------------------------------------
        # 2. Linearity: compare x vs x^2 for strongest predictor
        # ------------------------------------------------------
        best_pred = max(predictors, key=lambda c: abs(df_clean[c].corr(y)))
        x = df_clean[best_pred]

        # Build quadratic model: y ~ x + x^2
        df_quad = pd.DataFrame({
            best_pred: x,
            f"{best_pred}_2": x ** 2
        })
        df_quad = sm.add_constant(df_quad)

        try:
            model_quad = sm.OLS(y, df_quad).fit()

            # Extract quadratic p-value safely using label lookup
            quad_p = float(model_quad.pvalues.loc[f"{best_pred}_2"])
            is_linear = quad_p > 0.01

        except Exception:
            quad_p = np.nan
            is_linear = True

        results.append({
            "Variable": target,
            "Non-Gaussian (p<.05)": "PASS" if non_gaussian else "FAIL",
            "Linearity (Quad p>.01)": "PASS" if is_linear else "WARN",
            "Strongest_Link": best_pred,
            "Shapiro_p": round(sw_p, 4),
            "Quad_p": round(quad_p, 4) if not np.isnan(quad_p) else "NA"
        })

    out = pd.DataFrame(results)
    out.to_csv(output_dir / f"lingam_assumptions_{scenario_name}.csv", index=False)
    return out


def generate_descriptives(df, output_dir=None):
    """
    Generates Table 1 (Demographics), Table 2 (Ratings), and text statistics
    (Retention, Raters/Video, Demographic Bias P-values).
    """
    log.info("--- Generating All Descriptive Statistics ---")

    # =========================================================
    # 1. TABLE 1: DEMOGRAPHICS (Share % and Mean Valence/Arousal)
    # =========================================================
    demo_map = {
        "Gender": "Gender",
        "Age": "Age",
        "Cycling_frequency": "Cycling Frequency",
        "Cycling_confidence": "Cycling Confidence",
        "Cycling_purpose": "Cycling Purpose",
        "Cycling_environment": "Cycling Environment"
    }

    # Calculate Share % based on UNIQUE participants (N=60)
    total_participants = df[c.PARTICIPANT_ID].nunique()

    results = []
    for col, label in demo_map.items():
        if col in df.columns:
            for option in df[col].dropna().unique():
                sub_df = df[df[col] == option]
                n_part = sub_df[c.PARTICIPANT_ID].nunique()

                results.append({
                    "Category": label,
                    "Option": option,
                    "Share (%)": f"{(n_part / total_participants) * 100:.1f}",
                    "Valence": f"{sub_df['valence'].mean():.2f} ± {sub_df['valence'].std():.2f}",
                    "Arousal": f"{sub_df['arousal'].mean():.2f} ± {sub_df['arousal'].std():.2f}"
                })

    print("\n=== TABLE 1: DEMOGRAPHICS ===")
    print(pd.DataFrame(results).sort_values(["Category", "Option"]).to_string(index=False))

    # =========================================================
    # 2. TABLE 2: RATING DESCRIPTIVES (Mean, ICC, etc.)
    # =========================================================
    metrics = []
    for var in ["valence", "arousal"]:
        series = df[var].dropna()

        # ICC Calculation
        tmp = df[[var, c.VIDEO_ID_COL]].dropna()
        model = ols(f"{var} ~ C({c.VIDEO_ID_COL})", data=tmp).fit()
        ms_b = sm.stats.anova_lm(model, typ=1).loc[f"C({c.VIDEO_ID_COL})", 'mean_sq']
        ms_w = sm.stats.anova_lm(model, typ=1).loc['Residual', 'mean_sq']
        k = stats.hmean(tmp.groupby(c.VIDEO_ID_COL)[var].count())
        icc1 = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w)
        icc_k = (ms_b - ms_w) / ms_b

        metrics.append({
            "Dimension": var.capitalize(),
            "Mean (SD)": f"{series.mean():.3f} ({series.std():.3f})",
            "IQR": f"[{series.quantile(0.25):.2f}, {series.quantile(0.75):.2f}]",
            "Skew": f"{stats.skew(series):.2f}",
            "ICC(1)": f"{icc1:.2f}",
            "ICC(k)": f"{icc_k:.2f}"
        })

    print("\n=== TABLE 2: RATINGS ===")
    print(pd.DataFrame(metrics).to_string(index=False))

    # =========================================================
    # 3. MANUSCRIPT TEXT STATISTICS (Retention, Raters, OLS)
    # =========================================================
    print("\n=== TEXT STATISTICS ===")

    # A. Retention Rate
    expected_obs = 600
    actual_obs = len(df)
    print(f"Retention: {actual_obs}/{expected_obs} ({(actual_obs / expected_obs) * 100:.1f}%)")

    # B. Raters per Video
    raters = df.groupby(c.VIDEO_ID_COL)[c.PARTICIPANT_ID].nunique()
    print(f"Raters per Video: Mean={raters.mean():.2f}, SD={raters.std():.2f}")

    # C. Robust Demographic Bias Check
    print("\n--- Demographic Bias Checks (Major Groups Only) ---")

    # Define the "Major Groups" we want to test against each other
    # This excludes "Non-binary", "Other", "I do not cycle" to prevent skew
    keep_groups = {
        "Gender": ["Male", "Female"],
        "Cycling_environment": ["Urban area", "Rural area"],
        "Cycling_purpose": ["Commuting (e.g., work, school)", "Recreational / leisure", "Exercise / fitness"],
        "Cycling_frequency": ["Infrequent", "Ocasional", "Regular"],
    }

    demo_vars = ["Gender", "Age", "Cycling_frequency", "Cycling_confidence", "Cycling_purpose", "Cycling_environment"]
    available_demos = [d for d in demo_vars if d in df.columns]

    for y in ["valence", "arousal"]:
        print(f"\nOutcome: {y.capitalize()}")
        for x in available_demos:
            try:
                # FILTERING STEP:
                # If this variable has a restriction list, keep only those rows
                if x in keep_groups:
                    df_test = df[df[x].isin(keep_groups[x])].copy()
                else:
                    df_test = df.copy()

                # Check if we still have at least 2 groups
                if df_test[x].nunique() < 2:
                    print(f"  vs. {x:20s}: Skipped (Only 1 group remaining)")
                    continue

                # Run OLS on the FILTERED data
                formula = f"{y} ~ C({x})"
                model = ols(formula, data=df_test).fit(
                    cov_type='cluster',
                    cov_kwds={'groups': df_test[c.PARTICIPANT_ID]}
                )
                print(f"  vs. {x:20s}: p = {model.f_pvalue:.3f} (N={len(df_test)})")

            except Exception as e:
                print(f"  vs. {x:20s}: Error ({e})")


def analyze_street_morphology(df_ground):
    def _estimate_width(row):
        # Base: Car lanes + Sidewalks
        if row.get('car_lanes_total_count', 0) == 0:
            # Likely a dedicated path/pedestrian zone
            width = 4.0
        else:
            width = (row['car_lanes_total_count'] * 3.0) + 4.0

        # Add Tram Infrastructure
        if row.get('tram_lane_presence', False):
            width += 3.0

        # Add Bus Infrastructure
        # Handles both boolean (True/False) and numeric (0/1) types
        if row.get('bus_lane_presence', 0):
            width += 3.0

        # Add Side Parking (approx 2.0m for one side)
        if row.get('side_parking_presence', False):
            width += 2.0

        # Add Separated Bike Infrastructure
        # If it's a 'shared_path' NEXT to a road (car_lanes > 0), add width for the path.
        # If car_lanes == 0, the base width of 4.0m already covers the path.
        if row.get('bike_infra_type') == 'shared_path' and row.get('car_lanes_total_count', 0) > 0:
            width += 2.0

        return width

    df_ground['estimated_street_width'] = df_ground.apply(_estimate_width, axis=1)
    mean_w = df_ground['estimated_street_width'].mean()
    max_w = df_ground['estimated_street_width'].max()

    log.info(f"Buffer Justification analysis")
    log.info(f"Mean Street Width: {mean_w:.2f} m, Max Street Width:  {max_w:.2f} m")
    log.info(f"Implication: A 15m buffer (30m diameter) covers 100% of the max width ({max_w}m).")

    return df_ground


def create_aligned_index(df, cols, index_name, reference_col=None):
    log.info(f"--- Generating Aligned PCA Index: {index_name} ---")
    data = df[cols].fillna(0)
    pca = PCA(n_components=1, svd_solver='full')
    pc1 = pca.fit_transform(data).flatten()

    # Check Variance
    variance_expl = pca.explained_variance_ratio_[0]
    loadings = pca.components_[0]
    loading_dict = dict(zip(cols, loadings))

    log.info(f"{index_name}: PC1 explains {variance_expl * 100:.2f}% variance")

    # Alignment
    flip_factor = 1
    if reference_col and reference_col in loading_dict:
        ref_loading = loading_dict[reference_col]
        if ref_loading < 0:
            flip_factor = -1
    else:
        # If no ref, ensure the strongest variable is positive
        max_var = max(loading_dict, key=lambda k: abs(loading_dict[k]))
        if loading_dict[max_var] < 0:
            flip_factor = -1

    final_scores = pc1 * flip_factor
    loading_dict = {k: v * flip_factor for k, v in loading_dict.items()}

    return pd.Series(final_scores, index=df.index), variance_expl, loading_dict


def run_sensitivity_analysis(df, scenarios_config, outcomes, output_dir):
    log.info("--- Running Systematic Sensitivity Analysis (PC1 vs PC2) ---")
    results = []

    for scenario_name, config in scenarios_config.items():
        log.info(f"[Sensitivity] Evaluating scenario: {scenario_name}")

        for domain in ["infra", "landuse", "visual", "dynamic"]:

            domain_cols = config.get(f"{domain}_cols")
            ref_var = config.get(f"{domain}_ref")  # reference var for sign direction

            if not domain_cols or len(domain_cols) < 2:
                log.warning(f"[Sensitivity] {scenario_name}/{domain} — insufficient columns")
                continue

            # Extract data
            data = df[domain_cols].fillna(0)

            # PCA
            pca = PCA(n_components=2, svd_solver='full', random_state=42)
            components = pca.fit_transform(data)
            loadings = pca.components_
            var_expl = pca.explained_variance_ratio_

            # =====================================================
            # PC1 ORIENTATION
            # =====================================================
            pc1 = components[:, 0]
            load1 = loadings[0]

            if ref_var in domain_cols:
                ref_idx = domain_cols.index(ref_var)
                if load1[ref_idx] < 0:
                    pc1 = -pc1
                    load1 = -load1
            else:
                if load1[np.argmax(np.abs(load1))] < 0:
                    pc1 = -pc1
                    load1 = -load1

            # =====================================================
            # PC2 ORIENTATION
            # =====================================================
            pc2 = components[:, 1]
            load2 = loadings[1]

            if ref_var in domain_cols:
                ref_idx = domain_cols.index(ref_var)
                if load2[ref_idx] < 0:
                    pc2 = -pc2
                    load2 = -load2
            else:
                if load2[np.argmax(np.abs(load2))] < 0:
                    pc2 = -pc2
                    load2 = -load2

            # Make interpretable loading dict
            load1_dict = dict(zip(domain_cols, load1))
            load2_dict = dict(zip(domain_cols, load2))

            res = {
                "Scenario": scenario_name,
                "Domain": domain,
                "Reference_Var": ref_var,
                "Var_expl_PC1": var_expl[0],
                "Var_expl_PC2": var_expl[1],
                "Loadings_PC1": load1_dict,
                "Loadings_PC2": load2_dict
            }

            for out in outcomes:
                corr_pc1, p_pc1 = stats.spearmanr(pc1, df[out])
                corr_pc2, p_pc2 = stats.spearmanr(pc2, df[out])

                res[f"Corr_{out}_PC1"] = corr_pc1
                res[f"Pval_{out}_PC1"] = p_pc1

                res[f"Corr_{out}_PC2"] = corr_pc2
                res[f"Pval_{out}_PC2"] = p_pc2

                # Dominance test: stronger AND significant
                res[f"PC2_dominates_{out}"] = (
                        abs(corr_pc2) > abs(corr_pc1)
                        and p_pc2 < 0.05
                )

            results.append(res)

    df_sensitivity = pd.DataFrame(results)
    df_sensitivity.to_csv(output_dir / "sensitivity_analysis.csv", index=False)
    log.info(f"Sensitivity analysis written to: {output_dir / 'sensitivity_analysis.csv'}")

    return df_sensitivity


def run_sem_model(model_desc, df, name, output_dir, plot_graph=False):
    try:
        model = Model(model_desc)
        model.fit(df)
        est = model.inspect(std_est=True)
        fit = calc_stats(model)
        est.to_csv(output_dir / f"{name}_estimates.csv", index=False)
        if plot_graph:
            try:
                semplot(
                    model,
                    str(output_dir / f"{name}.png"),
                    plot_covs=False,
                    std_ests=True,
                    engine="dot",
                    show="estimates")
            except:
                pass
        return model, fit
    except Exception as e:
        log.error(f"Model {name} failed: {e}")
        return None


def run_lingam_discovery(df, cols, output_dir, scenario_name):
    try:
        X = df[cols].dropna()
        # -1 = Unknown, 0 = No Path, 1 = Path Exists
        prior = np.full((len(cols), len(cols)), -1)
        idx = {n: i for i, n in enumerate(cols)}

        # --- PRIOR KNOWLEDGE CONSTRAINTS ---
        env_vars = ['InfraIndex', 'VisualIndex', 'DynamicIndex']
        human_vars = [c for c in cols if c not in env_vars]

        for h in human_vars:
            for e in env_vars:
                prior[idx[e], idx[h]] = 0  # Human -> Env is IMPOSSIBLE

        # Enforce Env Structure (Infra -> Visual/Dynamic)
        prior[idx['InfraIndex'], idx['VisualIndex']] = 0
        prior[idx['InfraIndex'], idx['DynamicIndex']] = 0
        prior[idx['DynamicIndex'], idx['VisualIndex']] = 0
        prior[idx['VisualIndex'], idx['DynamicIndex']] = 0

        # -------------------------------------------------------
        # 1. Run Standard Model (for Coefficients)
        # -------------------------------------------------------
        model = lingam.DirectLiNGAM(prior_knowledge=prior)
        model.fit(X)

        # Save Fit Statistics
        scenario_out_dir = output_dir / f"suitability_{scenario_name}"
        scenario_out_dir.mkdir(exist_ok=True)
        assess_lingam_model_fit(model, df, cols, scenario_out_dir, scenario_name)

        # Save Coefficient Matrix (The "Effect Size")
        adj_df = pd.DataFrame(model.adjacency_matrix_, columns=cols, index=cols)
        adj_df.to_csv(output_dir / f"lingam_matrix_{scenario_name}.csv")


        log.info(f"--- Running LiNGAM Bootstrap for {scenario_name} ---")

        res = model.bootstrap(X, n_sampling=500)
        probs = res.get_probabilities(min_causal_effect=0.01)
        probs_df = pd.DataFrame(probs, columns=cols, index=cols)
        probs_df.to_csv(output_dir / f"lingam_bootstrap_probs_{scenario_name}.csv")

        # -------------------------------------------------------
        # 3. Save Graphs
        # -------------------------------------------------------
        try:
            # Graph 1: Standard (Coefficients)
            dot = make_dot(model.adjacency_matrix_, labels=cols)
            dot.format = 'png'
            dot.render(str(output_dir / f"lingam_graph_{scenario_name}"), view=False)

            # Graph 2: Bootstrap Probabilities (Optional visualization)
            # This helps you visualize which edges are stable (>0.8) vs unstable (<0.5)
            # We pass the probability matrix instead of the adjacency matrix
            dot_prob = make_dot(probs, labels=cols, lower_limit=0.5)  # Only show edges > 50% freq
            dot_prob.format = 'png'
            dot_prob.render(str(output_dir / f"lingam_graph_probs_{scenario_name}"), view=False)

            log.info(f"LiNGAM results and bootstrap probabilities saved for {scenario_name}")

        except Exception as e:
            log.warning(f"Could not render LiNGAM graph: {e}")

    except ImportError:
        log.warning("LiNGAM library not installed. Skipping.")
    except Exception as e:
        log.error(f"LiNGAM failed for {scenario_name}: {e}")


def calculate_sem_r2(model, df):
    """
    Calculates R2 using the 'lval', 'rval', 'Estimate' columns from your semopy version.
    R2 = 1 - (Error Variance / Total Variance)
    """
    try:
        stats = model.inspect()
        total_variances = df.var(numeric_only=True)

        r2_results = {}

        for index, row in stats.iterrows():
            if row['op'] == '~~' and row['lval'] == row['rval']:
                var_name = row['lval']

                if var_name in total_variances:
                    error_variance = row['Estimate']
                    total_variance = total_variances[var_name]

                    if total_variance == 0:
                        r2_results[var_name] = 0.0
                        continue

                    r2 = 1 - (error_variance / total_variance)
                    r2 = max(0.0, min(1.0, r2))
                    r2_results[var_name] = r2

        return r2_results

    except Exception as e:
        log.warning(f"Could not calculate R2 from estimates: {e}")
        return {}


def plot_correlation_matrix(
        corr_plot,
        pretty_names,
        df_raw,
        output_dir,
        file_name="correlation_matrix.png"
):

    plt.figure(figsize=(18, 14))
    mask = np.triu(np.ones_like(corr_plot, dtype=bool))

    ax = sns.heatmap(
        corr_plot,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1, vmax=1,
        linewidths=0.4,
        square=True,
        cbar=False
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="3%", pad=2.5)
    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.set_xlabel("Spearman correlation (ρ)", fontsize=14, labelpad=10)
    cb.ax.tick_params(labelsize=14)
    cb.outline.set_visible(False)

    ax.set_aspect('equal')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')

    ax.set_xticklabels(
        [pretty_names.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()],
        fontsize=14,
        rotation=90
    )

    ax.set_yticklabels(
        [pretty_names.get(y.get_text(), y.get_text()) for y in ax.get_yticklabels()],
        fontsize=14,
        rotation=0
    )

    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

    xlabels = ax.get_xticklabels()
    if xlabels:
        xlabels[-1].set_visible(False)

    fig = plt.gcf()
    fig.suptitle(f"Multi-Modal Correlation Matrix (N={len(df_raw)})", fontsize=14, y=0.98)
    plt.tight_layout()

    plt.savefig(Path(output_dir, file_name), dpi=300, bbox_inches="tight")
    plt.close()


def plot_explained_variance(output_dir):

    df = pd.read_csv(Path(output_dir, "SEM_model_comparison.csv"))

    # We select the specific Model + Scenario combinations that tell the story
    # logic: (Scenario, Model, Display Label)
    target_chain = [
        ("Planner_View", "04_Full_Environment", "M1-Static\n(GIS Only)"),
        ("Visual_Cyclist", "04_Full_Environment","M2-Visual\n(Segmentation)"),
        ("Temporal_Cyclist", "04_Full_Environment", "M3-Temporal\n(MLLM Events)"),
        ("Visual_Temporal_Cyclist", "04_Full_Environment", "M4-Integrated\n(Segmentation+MLLM Events)")
    ]

    plot_data = []

    for scenario, model, label in target_chain:
        # Find the specific row
        row = df[(df["Scenario"] == scenario) & (df["Model"] == model)]

        if not row.empty:
            plot_data.append({
                "Label": label,
                "R2_Valence": row.iloc[0]["R2_Valence"],
                "R2_Arousal": row.iloc[0]["R2_Arousal"]
            })
        else:
            log.warning(f"Could not find result for {scenario} - {model}")

    df_plot = pd.DataFrame(plot_data)

    # --- Plotting ---
    labels = df_plot["Label"]
    x = np.arange(len(labels))
    width = 0.35

    cmap = plt.get_cmap("vlag")
    COLOR_VALENCE = cmap(0.05)
    COLOR_AROUSAL = cmap(0.95)

    fig, ax = plt.subplots(figsize=(8, 4))

    bars1 = ax.bar(x - width / 2, df_plot["R2_Valence"], width, label="Valence", color=COLOR_VALENCE)
    bars2 = ax.bar(x + width / 2, df_plot["R2_Arousal"], width, label="Arousal", color=COLOR_AROUSAL)

    # Annotate bars
    for b in bars1:
        yval = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, yval + 0.01, f"{yval:.2f}", ha='center', fontsize=10)

    for b in bars2:
        yval = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, yval + 0.01, f"{yval:.2f}", ha='center', fontsize=10)

    ax.set_ylabel("Explained variance ($R^2$)", fontsize=12)
    ax.set_title("Information Scenarios", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 0.6)  # Set limit slightly higher than max value (0.46)

    # Grid and Layout
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=10, loc='upper left')

    plt.tight_layout()
    plt.savefig(Path(output_dir, "information_scenarios.png"), dpi=300)
    plt.close()

    print("Saved corrected: information_scenarios.png")


# -------------------------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------------------------
def main():
    # ============================================================
    # 0. LOAD RAW DATA
    # ============================================================
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("config.ini")
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.read_excel(config["filenames"]["lab_study_results_file"]).set_index(c.PARTICIPANT_ID)
    df_physio = pd.read_csv(config["filenames"]["physiological_results_file"])
    df_ground = pd.read_csv(config["filenames"]["video_info_ground_truth"])
    df_setup = pd.read_csv(config["filenames"]["lab_experiment_setup_file"], header=None).set_index(0)

    # This will print the mean/max width to your console/log
    df_ground = analyze_street_morphology(df_ground)

    log.info(f"Initial results dataframe size: {len(df_results)}, NaNs per column: {df_results.isna().sum()}")
    # ============================================================
    # 1. PARTICIPANT FILTERING
    # ============================================================
    df_results = utils.processing_utils.filter_aggregate_results(
        df_results,
        consent=False,
        duration=False,
        location=False,
        gender=False,
        cycling_environment=False,
        cycling_frequency=True,
        cycling_confidence=True
    )

    # ============================================================
    # 2. MAP SUBJECTIVE RATINGS TO VIDEOS
    # ============================================================
    scores_df = df_results.drop(columns=c.DEMOGRAPHIC_COLUMNS + [c.START, c.END], errors='ignore')
    scores_df = scores_df.apply(pd.to_numeric, errors='coerce')

    trial_dict = utils.helper_functions.get_trial_dict(scores_df, df_setup, c.TRIAL_1, c.VIDEO_COUNTS)
    df_ratings = utils.helper_functions.trial_dict_to_df(trial_dict)
    df_ratings = utils.processing_utils.add_valence_arousal(df_ratings, "rating")

    # ============================================================
    # 3. MAP PHYSIOLOGY TO VIDEOS
    # ============================================================
    eda_cols = ['SCL_Delta', 'SCR_Peaks_N', 'SCR_Peaks_Amplitude_Mean']
    ppg_cols = ['PPG_Rate_Mean', 'HRV_RMSSD']

    df_physio_mapped = map_physiological_segments_to_videos(df_physio, df_setup, c.TRIAL_1, c.VIDEO_COUNTS)
    df_physio_mapped = df_physio_mapped[[c.PARTICIPANT_ID, c.VIDEO_ID_COL] + ppg_cols + eda_cols]

    demo_cols = [
        'Gender',
        'Age',
        'Cycling_frequency',
        'Cycling_confidence',
        'Cycling_purpose',
        'Cycling_environment'
    ]
    df_demo = df_results[demo_cols].reset_index()
    df_demo[c.PARTICIPANT_ID] = df_results.index

    # ============================================================
    # 4. MERGE INTO MASTER DATAFRAME (RAW)
    # ============================================================
    df_raw = (df_ratings
              .merge(df_physio_mapped, on=[c.PARTICIPANT_ID, c.VIDEO_ID_COL], how="left")
              .merge(df_ground, on=c.VIDEO_ID_COL, how="left")
              .merge(df_demo, on=c.PARTICIPANT_ID, how="left"))

    df_raw = df_raw[~df_raw[c.PARTICIPANT_ID].isin([0, 3, 12, 24])].copy()

    generate_descriptives(df_raw, output_dir)
    plot_valence_arousal_scatter(df_raw, output_dir)

    subj_cols = [
        "valence",
        "arousal"
    ]
    visual_cols = [
        "average_greenery_share",
        "average_sky_share",
        "average_building_share",
        "average_road_share",
        "pois_count",
    ]
    infra_cols = [
        "bike_infra_type_numeric",
        "tram_lane_presence",
        "bus_lane_presence",
        "side_parking_count",
        "intersection_count",
        "tree_canopy_share",
        "greenery_share_gis",
        "commercial_share",
        "residential_share",
        "building_count",
        "car_lanes_total_count",
        "traffic_volume",
        "motorized_traffic_speed_kmh",
    ]
    dynamic_cols = [
        "ped_and_cycl_count",
        "motor_vehicle_overtakes_count",
        "unique_motor_vehicles_count"
    ]

    log.info("Visualization complete.")

    # ============================================================
    # 5. TRANSFORMATIONS
    # ============================================================
    # A. Log Transform - These variables follow a Poisson distribution (skewed, many zeros).
    skewed_count_vars = [
        # dynamic
        "ped_and_cycl_count",
        "motor_vehicle_overtakes_count",
        "unique_motor_vehicles_count",

        # physiological
        "SCR_Peaks_N",
        'SCR_Peaks_Amplitude_Mean',

        # infra
        "intersection_count",
        "building_count",

        # visual
        "pois_count",
    ]

    for col in skewed_count_vars:
        if col in df_raw.columns:
            df_raw[col] = np.log1p(df_raw[col])

    # C. Mean centering within participants AND Z-scoring within participants
    dep_vars = subj_cols + ppg_cols + eda_cols

    # D. Mean Center (Remove Baseline Bias)
    for col in dep_vars:
        df_raw[col] = df_raw[col] - df_raw.groupby(c.PARTICIPANT_ID)[col].transform('mean')

    # E. Global Z-score for all environment variables
    env_vars = list(set(infra_cols + visual_cols + dynamic_cols))
    df_raw[env_vars] = df_raw[env_vars].apply(zscore)

    # ============================================================
    # 6. CORRELATION MATRIX (RAW VARIABLES, TRIANGULAR HEATMAP)
    # ============================================================
    log.info("--- Building Correlation Matrix ---")

    pretty_names = {
        # Subjective
        "valence": "Valence",
        "arousal": "Arousal",
        # PPG
        "PPG_Rate_Mean": "Mean Heart Rate",
        "HRV_RMSSD": "HRV RMSSD",
        # EDA
        "SCL_Delta": "SCL Δ",
        "SCR_Peaks_N": "SCR Peaks",
        "SCR_Peaks_Amplitude_Mean": "SCR Amplitude (mean)",
        # Infrastructure (static)
        "bike_infra_type_numeric": "Bike Infra. Level",
        "bus_lane_presence": "Bus Lane Presence",
        "intersection_count": "Intersection Count",
        "tree_canopy_share": "Tree Canopy (%)",
        "greenery_share_gis": "Greenery Share (GIS)",
        "building_count": "Building Count",
        "car_lanes_total_count": "Car Lanes Count",
        "traffic_volume": "Traffic Volume (AADT)",
        "motorized_traffic_speed_kmh": "Speed Limit",
        # Visual
        "pois_count": "POIs Count",
        "average_greenery_share": "Greenery share (Seg.)",
        "average_sky_share": "Sky share (Seg.)",
        "average_building_share": "Building share (Seg.)",
        "average_road_share": "Road share (Seg.)",
        # Dynamic
        "ped_and_cycl_count": "Ped. & Cyclists Count",
        "motor_vehicle_overtakes_count": "Overtakes Count",
        "unique_motor_vehicles_count": "Vehicles Count"
    }

    corr_cols = list(pretty_names.keys())
    corr_df = df_raw[corr_cols].copy()
    corr_matrix = corr_df.corr(method="spearman")
    corr_path = output_dir / "correlation_matrix_raw.csv"
    corr_matrix.to_csv(corr_path)

    plot_cols = corr_cols.copy()
    corr_plot = corr_matrix.loc[plot_cols, plot_cols]
    plot_correlation_matrix(corr_plot, pretty_names, df_raw, output_dir)

    # ============================================================
    # 7. SCENARIOS
    # ============================================================

    # SCENARIOS
    standardized_infra_cols = ["bike_infra_type_numeric", "bus_lane_presence", "intersection_count",
                               "car_lanes_total_count"
                               ]

    SCENARIOS_CONFIG = {
        "Planner_View": {
            "infra_cols": standardized_infra_cols,
            "infra_ref": "bike_infra_type_numeric",
            "visual_cols": ["tree_canopy_share", "greenery_share_gis", "building_count", "pois_count"],
            "visual_ref": "greenery_share_gis",
            "dynamic_cols": ["traffic_volume", "motorized_traffic_speed_kmh"],
            "dynamic_ref": "traffic_volume"
        },
        "Visual_Cyclist": {
            "infra_cols": standardized_infra_cols,
            "infra_ref": "bike_infra_type_numeric",
            "visual_cols": ["average_greenery_share", "average_building_share", "average_sky_share",
                            "average_road_share"],
            "visual_ref": "average_greenery_share",
            "dynamic_cols": ["traffic_volume", "motorized_traffic_speed_kmh"],
            "dynamic_ref": "traffic_volume"
        },
        "Temporal_Cyclist": {
            "infra_cols": standardized_infra_cols,
            "infra_ref": "bike_infra_type_numeric",
            "visual_cols": ["tree_canopy_share", "greenery_share_gis", "building_count", "pois_count"],
            "visual_ref": "greenery_share_gis",
            "dynamic_cols": ["ped_and_cycl_count", "motor_vehicle_overtakes_count", "unique_motor_vehicles_count"],
            "dynamic_ref": "motor_vehicle_overtakes_count"
        },
        "Visual_Temporal_Cyclist": {
            "infra_cols": standardized_infra_cols,
            "infra_ref": "bike_infra_type_numeric",
            "visual_cols": ["average_greenery_share", "average_building_share", "average_sky_share",
                            "average_road_share"],
            "visual_ref": "average_greenery_share",
            "dynamic_cols": ["ped_and_cycl_count", "motor_vehicle_overtakes_count", "unique_motor_vehicles_count"],
            "dynamic_ref": "motor_vehicle_overtakes_count"
        }
    }
    # ============================================================
    # 8. INDEX CREATION (SIGN–ALIGNED)
    # ============================================================
    log.info("--- Calculating Sign-Aligned Equal-Weighted Indices ---")

    index_loading_report = []

    for scenario_name, config in SCENARIOS_CONFIG.items():
        print(f"\n\n=== BUILDING PCA INDICES FOR SCENARIO: {scenario_name} ===")

        # INFRA
        df_raw[f"InfraIndex_{scenario_name}"], var_inf, load_inf = create_aligned_index(
            df_raw, config["infra_cols"], f"InfraIndex_{scenario_name}", reference_col=config["infra_ref"]
        )
        index_loading_report.append(
            {"Scenario": scenario_name, "Type": "Infra", "Var_PC1": var_inf, "Loadings": load_inf}
        )

        # VISUAL
        df_raw[f"VisualIndex_{scenario_name}"], var_vis, load_vis = create_aligned_index(
            df_raw, config["visual_cols"], f"VisualIndex_{scenario_name}", reference_col=config["visual_ref"]
        )
        index_loading_report.append(
            {"Scenario": scenario_name, "Type": "Visual", "Var_PC1": var_vis, "Loadings": load_vis}
        )

        # DYNAMIC
        df_raw[f"DynamicIndex_{scenario_name}"], var_dyn, load_dyn = create_aligned_index(
            df_raw, config["dynamic_cols"], f"DynamicIndex_{scenario_name}", reference_col=config["dynamic_ref"]
        )
        index_loading_report.append(
            {"Scenario": scenario_name, "Type": "Dynamic", "Var_PC1": var_dyn, "Loadings": load_dyn}
        )

    # PPG: Invert HRV so it aligns with Arousal
    ppg_signed_cols = []
    for v in ppg_cols:
        if v == "HRV_RMSSD":
            df_raw[v + "_signed"] = df_raw[v] * -1
        else:
            df_raw[v + "_signed"] = df_raw[v]
        ppg_signed_cols.append(v + "_signed")

    # Create EDA Index via PCA
    df_raw["EDAIndex"], var_eda, load_eda = create_aligned_index(
        df_raw,
        eda_cols,
        "EDAIndex",
        reference_col='SCR_Peaks_N'
    )

    # Create PPG Index via PCA
    df_raw["PPGIndex"], var_ppg, load_ppg = create_aligned_index(
        df_raw,
        ppg_cols,
        "PPGIndex",
        reference_col='PPG_Rate_Mean'
    )

    corr = df_raw[["EDAIndex", 'SCR_Peaks_N', "PPGIndex", 'PPG_Rate_Mean', "valence", "arousal"]].corr()
    corr.to_csv(output_dir / "physio_index_correlation_raw.csv")

    # Save loadings and perform sensitivity analysis
    pd.DataFrame(index_loading_report).to_csv(output_dir / "index_loading_report.csv", index=False)
    run_sensitivity_analysis(df_raw, SCENARIOS_CONFIG, ["valence", "arousal"], output_dir)

    df_raw['PPGIndex'] = df_raw['PPG_Rate_Mean']
    df_raw['EDAIndex'] = df_raw['SCR_Peaks_N']
    log.info("Saved PCA loading and sensitivity report")

    # ============================================================
    # Run MI for each domain and scenario
    # ============================================================
    for scenario_name, config in SCENARIOS_CONFIG.items():
        log.info(f" Mutual Information: {scenario_name}")

        for domain in ["infra", "visual", "dynamic"]:
            cols = config[f"{domain}_cols"]

            MI_val, MI_ar = mutual_info(df_raw, cols)
            log.info("\nMutual Information (Valence/Arousal):")
            for f, mv, ma in zip(cols, MI_val, MI_ar):
                log.info(f"{f:30s} val={mv:.3f} ar={ma:.3f}")

    # ============================================================
    # 12. DEFINE MODEL CATALOG (Final & Complete)
    # ============================================================
    MODEL_CATALOG = {

        # FAMILY 1: THE BASICS (Direct Effects)
        "04_Full_Environment": """
            valence ~ InfraIndex + VisualIndex + DynamicIndex
            arousal ~ InfraIndex + VisualIndex + DynamicIndex
            
            valence ~~ arousal
            InfraIndex  ~~ VisualIndex + DynamicIndex
            VisualIndex ~~ DynamicIndex
        """,

        # FAMILY 2: MEDIATION (The Mechanism)
        "05_Full_Mediation": """
            VisualIndex  ~ InfraIndex
            DynamicIndex ~ InfraIndex
            
            valence      ~ VisualIndex + DynamicIndex
            arousal      ~ VisualIndex + DynamicIndex
            
            valence ~~ arousal
            VisualIndex ~~ DynamicIndex
        """,

        # FAMILY 3: PHYSIOLOGY (Mind-Body Connection)
        "06_Physio_Parallel": """
            # Inputs
            InfraIndex ~~ VisualIndex
            InfraIndex ~~ DynamicIndex
            VisualIndex ~~ DynamicIndex

            # Env -> Mind
            valence ~ InfraIndex + VisualIndex + DynamicIndex
            arousal ~ InfraIndex + VisualIndex + DynamicIndex

            # Env -> Body (Both Stress and Safety)
            PPGIndex    ~ InfraIndex + VisualIndex + DynamicIndex
            EDAIndex    ~ InfraIndex + VisualIndex + DynamicIndex
           
            # Residual Correlations
            valence ~~ arousal
            PPGIndex ~~ EDAIndex
        """,

        "07_Causal_Body_to_Mind": """
            # Inputs correlated
            InfraIndex ~~ VisualIndex
            InfraIndex ~~ DynamicIndex
            VisualIndex ~~ DynamicIndex

            # Env -> Body
            PPGIndex ~ InfraIndex + VisualIndex + DynamicIndex
            EDAIndex ~ InfraIndex + VisualIndex + DynamicIndex

            # Body -> Mind
            valence ~ PPGIndex + EDAIndex
            arousal ~ PPGIndex + EDAIndex

            # Residual correlations
            valence ~~ arousal
            PPGIndex ~~ EDAIndex
            """,

        "08_Causal_Mind_to_Body": """
            # Inputs correlated
            InfraIndex ~~ VisualIndex
            InfraIndex ~~ DynamicIndex
            VisualIndex ~~ DynamicIndex

            # Env -> Mind
            valence ~ InfraIndex + VisualIndex + DynamicIndex
            arousal ~ InfraIndex + VisualIndex + DynamicIndex

            # Mind -> Body
            PPGIndex ~ valence + arousal
            EDAIndex ~ valence + arousal

            # Residual correlations
            valence ~~ arousal
            PPGIndex ~~ EDAIndex
        """,

        # FAMILY 4: INTERACTIONS (The Buffering Effects)
        "09_Mod_Infra_Buffers_Traffic": """
            # Exogenous covariances
            InfraIndex ~~ VisualIndex
            InfraIndex ~~ DynamicIndex
            VisualIndex ~~ DynamicIndex

            # Affect
            valence ~ InfraIndex + DynamicIndex + Infra_Dynamic_Interaction + VisualIndex
            arousal ~ InfraIndex + DynamicIndex + Infra_Dynamic_Interaction + VisualIndex

            # Residual covariances
            valence ~~ arousal
        """,

        "10_Mod_Beauty_Buffers_Traffic": """
            # Exogenous covariances
            InfraIndex ~~ VisualIndex
            InfraIndex ~~ DynamicIndex
            VisualIndex ~~ DynamicIndex

            # Affect
            valence ~ VisualIndex + DynamicIndex + Visual_Dynamic_Interaction + InfraIndex
            arousal ~ VisualIndex + DynamicIndex + Visual_Dynamic_Interaction + InfraIndex

            # Residual correlations
            valence ~~ arousal
        """,

        "11_Mod_Beauty_Enhances_Infra": """
            # Exogenous covariances
            InfraIndex ~~ VisualIndex
            InfraIndex ~~ DynamicIndex
            VisualIndex ~~ DynamicIndex

            # Affect
            valence ~ InfraIndex + VisualIndex + Infra_Visual_Interaction + DynamicIndex
            arousal ~ InfraIndex + VisualIndex + Infra_Visual_Interaction + DynamicIndex

            # Residual covariances
            valence ~~ arousal
        """,

        "12_Combined_Interactions": """
            valence ~ InfraIndex + VisualIndex + DynamicIndex + Infra_Dynamic_Interaction + Infra_Visual_Interaction
            arousal ~ InfraIndex + VisualIndex + DynamicIndex + Infra_Dynamic_Interaction + Infra_Visual_Interaction

            valence ~~ arousal
            InfraIndex ~~ VisualIndex + DynamicIndex
            VisualIndex ~~ DynamicIndex
        """
        }

    # ============================================================
    # 13. EXECUTION LOOP (With Full Fit Metrics)
    # ============================================================
    log.info("--- Running Scenario Comparison Loop ---")

    FIT_METRICS = ["DoF", "chi2", "chi2 p-value", "CFI", "TLI", 'AGFI', "RMSEA", "AIC", "BIC", 'LogLik']
    comparison_results = []

    for scenario_name in SCENARIOS_CONFIG.keys():
        log.info(f"\n>> PROCESSING SCENARIO: {scenario_name}")
        df_model = df_raw.copy()

        # Rename indices for modeling convenience
        rename_map = {
            f"InfraIndex_{scenario_name}": "InfraIndex",
            f"VisualIndex_{scenario_name}": "VisualIndex",
            f"DynamicIndex_{scenario_name}": "DynamicIndex"
        }
        df_model = df_model.rename(columns=rename_map)

        df_model["Infra_Dynamic_Interaction"] = df_model["InfraIndex"] * df_model["DynamicIndex"]
        df_model["Visual_Dynamic_Interaction"] = df_model["VisualIndex"] * df_model["DynamicIndex"]
        df_model["Infra_Visual_Interaction"] = df_model["InfraIndex"] * df_model["VisualIndex"]

        # -------------------------------------------------------
        # RUN LiNGAM FOR THIS SCENARIO
        # -------------------------------------------------------
        lingam_cols = [
            'InfraIndex',
            'VisualIndex',
            'DynamicIndex',
            'EDAIndex',
            'PPGIndex',
            'valence',
            'arousal'
        ]

        scenario_out_dir = output_dir / f"suitability_{scenario_name}"
        scenario_out_dir.mkdir(exist_ok=True)

        check_lingam_data_assumptions(df_model, lingam_cols, scenario_out_dir, scenario_name)
        run_lingam_discovery(df_model, lingam_cols, output_dir, scenario_name)

        # -------------------------------------------------------
        # RUN SEM MODELS
        # -------------------------------------------------------
        for model_name, syntax in MODEL_CATALOG.items():
            run_name = f"{scenario_name}_{model_name}"
            model, fit_stats = run_sem_model(syntax, df_model, run_name, output_dir)

            if "Mod" in model_name:
                log.info(f"\n--- Running VIF Check for {model_name} ---")
                if "09_Mod" in model_name:
                    predictors = ["InfraIndex", "DynamicIndex", "Infra_Dynamic_Interaction"]
                elif "10_Mod" in model_name:
                    predictors = ["VisualIndex", "DynamicIndex", "Visual_Dynamic_Interaction"]
                elif "11_Mod" in model_name:
                    predictors = ["InfraIndex", "VisualIndex", "Infra_Visual_Interaction"]

                # Run the VIF check function
                vif_df = check_vif(df_model, predictors)
                max_vif = vif_df["VIF"].max()

                if max_vif > 5:
                    log.info(f" > Caution: Moderate multicollinearity detected (Max VIF = {max_vif: .2f})")
                else:
                    log.info(f" > VIF OK (Max VIF = {max_vif:.2f})")

            if fit_stats is not None:
                res = {
                    "Scenario": scenario_name,
                    "Model": model_name
                }

                for m in FIT_METRICS:
                    res[m] = fit_stats[m].iloc[0]

                r2_data = calculate_sem_r2(model, df_model)
                res["R2_Valence"] = r2_data.get("valence", None)
                res["R2_Arousal"] = r2_data.get("arousal", None)

                # Extract Physiology R2 (returns None if variable is not in model)
                res["R2_PPG"] = r2_data.get("PPGIndex", None)
                res["R2_EDA"] = r2_data.get("EDAIndex", None)

                # --- WALD TESTS FOR INTERACTION MODELS ONLY ---
                if model_name.startswith(("09_", "10_", "11_")):
                    if "09_" in model_name:
                        interaction = "Infra_Dynamic_Interaction"
                    elif "10_" in model_name:
                        interaction = "Visual_Dynamic_Interaction"
                    elif "11_" in model_name:
                        interaction = "Infra_Visual_Interaction"

                    wald = extract_interaction_wald(model, interaction, "valence")

                    if wald is not None:
                        res[f"{interaction}_beta"] = wald["Estimate"]
                        res[f"{interaction}_z"] = wald["z"]
                        res[f"{interaction}_p"] = wald["p"]

                comparison_results.append(res)

    df_compare = pd.DataFrame(comparison_results)
    df_compare.to_csv(output_dir / "SEM_model_comparison.csv", index=False)

    log.info("SEM modeling complete. Check output directory for results.")

    plot_explained_variance(output_dir)


if __name__ == "__main__":
    main()
