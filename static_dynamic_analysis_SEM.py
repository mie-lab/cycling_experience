import utils.helper_functions
import utils.processing_utils
from scipy.stats import zscore
from semopy import Model, semplot
from semopy.stats import calc_stats
from sklearn.decomposition import PCA
import pandas as pd
import configparser
import utils.helper_functions
import utils.processing_utils
import logging
import constants as c
from utils.physiological_data_utils import map_physiological_segments_to_videos
import scipy.stats as stats
import seaborn as sns
import matplotlib
import lingam
from lingam.utils import make_dot
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.use("Agg")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# -------------------------------------------------------------------
# UTILS: TRANSFORMATIONS
# -------------------------------------------------------------------


def create_aligned_index(df, cols, index_name, reference_col=None):
    data = df[cols].fillna(0)
    pca = PCA(n_components=1, svd_solver='full')
    pc1 = pca.fit_transform(data).flatten()

    # Check Variance
    variance_expl = pca.explained_variance_ratio_[0]
    loadings = pca.components_[0]
    loading_dict = dict(zip(cols, loadings))

    print(f"{index_name}: PC1 explains {variance_expl * 100:.2f}% variance")

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


def build_index_from_corr(df, cols, index_name, corr_matrix, valence_col="valence"):
    """
    Builds an equal-weight index for given variables using only the
    sign from correlation with valence.
    """

    signs = {}
    print(f"\n[Index construction: {index_name}]")
    print("Variable  | corr(valence)   SIGN")

    # fetch valence correlations
    for v in cols:
        r = corr_matrix.loc[v, valence_col]
        signs[v] = 1 if r >= 0 else -1
        print(f"{v:20s}   {r:+.3f}        {signs[v]:+d}")

        # apply sign to the dataframe
        df[v + "_signed"] = df[v] * signs[v]

    # compute equal-weighted arithmetic mean index
    signed_cols = [v + "_signed" for v in cols]
    df[index_name] = df[signed_cols].mean(axis=1)

    return df, signs


def sensitivity_leave_one_out(df, cols, outcome_vars, index_name):
    results = []

    # full index first:
    full_index = df[cols].mean(axis=1)

    for outcome in outcome_vars:
        corr_full, _ = stats.spearmanr(full_index, df[outcome])

        for drop_col in cols:
            subset = [c for c in cols if c != drop_col]
            idx = df[subset].mean(axis=1)
            corr_subset, _ = stats.spearmanr(idx, df[outcome])

            results.append({
                "Index": index_name,
                "Outcome": outcome,
                "Dropped_Feature": drop_col,
                "Corr_Full_Index": corr_full,
                "Corr_Drop_One": corr_subset,
                "Change_in_corr": corr_subset - corr_full
            })

    return pd.DataFrame(results)


def run_sensitivity_analysis(df, scenarios_config, outcomes, output_dir):
    log.info("--- Running Systematic Sensitivity Analysis (PC1 vs PC2) ---")
    results = []

    for scenario_name, config in scenarios_config.items():
        log.info(f"[Sensitivity] Evaluating scenario: {scenario_name}")

        for domain in ["infra", "visual", "dynamic"]:

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
    """Robust runner for SEM models with error handling and plotting."""
    try:
        model = Model(model_desc)
        model.fit(df)
        est = model.inspect(std_est=True)
        fit = calc_stats(model)
        est.to_csv(output_dir / f"{name}_estimates.csv", index=False)
        if plot_graph:
            try:
                semplot(model,
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
    """
    Runs LiNGAM causal discovery with strict constraints and saves CSVs.
    """
    try:
        X = df[cols].dropna()
        # -1 = Unknown, 0 = No Path, 1 = Path Exists
        prior = np.full((len(cols), len(cols)), -1)
        idx = {n: i for i, n in enumerate(cols)}

        # --- CONSTRAINT GROUP A: HUMAN vs. ENVIRONMENT ---
        # Humans (Physio/Rating) cannot cause the Environment
        env_vars = ['InfraIndex', 'VisualIndex', 'DynamicIndex']
        human_vars = [c for c in cols if c not in env_vars]

        for h in human_vars:
            for e in env_vars:
                prior[idx[e], idx[h]] = 0  # Human -> Env is IMPOSSIBLE

        # --- CONSTRAINT GROUP B: INFRA IS ROOT ---
        # Visuals/Dynamics don't build roads.
        prior[idx['InfraIndex'], idx['VisualIndex']] = 0  # Vis -> Infra impossible
        prior[idx['InfraIndex'], idx['DynamicIndex']] = 0  # Dyn -> Infra impossible

        # --- CONSTRAINT GROUP C: INDEPENDENT CHANNELS ---
        # We enforce that Visual and Dynamic are parallel outcomes of Infra, not causes of each other.
        prior[idx['DynamicIndex'], idx['VisualIndex']] = 0  # Vis -> Dyn IMPOSSIBLE
        prior[idx['VisualIndex'], idx['DynamicIndex']] = 0  # Dyn -> Vis IMPOSSIBLE

        # 3. Run Model
        model = lingam.DirectLiNGAM(prior_knowledge=prior)
        model.fit(X)

        # A. Save Adjacency Matrix
        # The matrix shows coefficients: Column causes Row.
        adj_df = pd.DataFrame(model.adjacency_matrix_, columns=cols, index=cols)
        adj_df.to_csv(output_dir / f"lingam_matrix_{scenario_name}.csv")

        # B. Save Causal Order List
        causal_order_names = [cols[i] for i in model.causal_order_]
        order_df = pd.DataFrame(causal_order_names, columns=["Causal_Rank"])
        order_df.to_csv(output_dir / f"lingam_order_{scenario_name}.csv", index=False)

        try:
            dot = make_dot(model.adjacency_matrix_, labels=cols)
            dot.format = 'png'
            dot.render(str(output_dir / f"lingam_graph_{scenario_name}"), view=False)
            log.info(f"LiNGAM results saved for {scenario_name}")
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
        # 1. Get parameter estimates
        stats = model.inspect()

        # 2. Calculate Total Variance from data (Numeric columns only to avoid 'video_id' error)
        total_variances = df.var(numeric_only=True)

        r2_results = {}

        # 3. Find Error Variances
        for index, row in stats.iterrows():
            # Look for variance term: operator '~~' where Left Value == Right Value
            if row['op'] == '~~' and row['lval'] == row['rval']:
                var_name = row['lval']

                # Only calculate for endogenous variables we have data for
                if var_name in total_variances:
                    error_variance = row['Estimate']
                    total_variance = total_variances[var_name]

                    # Safety check for zero variance
                    if total_variance == 0:
                        r2_results[var_name] = 0.0
                        continue

                    # Standard R2 formula
                    r2 = 1 - (error_variance / total_variance)

                    # Clip to valid 0-1 range
                    r2 = max(0.0, min(1.0, r2))

                    r2_results[var_name] = r2

        return r2_results

    except Exception as e:
        log.warning(f"Could not calculate R2 from estimates: {e}")
        return {}


def plot_raw_distributions(df, subjective_cols, physio_arousal_vars, physio_cog_vars, infra_cols, dynamic_cols,
                           visual_cols, output_dir):
    """
    Generates KDE plots for all raw, numeric variables of interest.
    """
    log.info("Generating raw variable distribution plots...")

    # Define columns exactly as in your transformation section (before any transformation)
    plotting_groups = {
        "1. Subjective Ratings": subjective_cols,
        "2. Physiological Metrics (Arousal & Effort)": physio_arousal_vars + physio_cog_vars,
        "3. Static Infrastructure (OSM)": infra_cols,
        "4. Raw Dynamic Counts (MLLM)": dynamic_cols,
        "5. Raw Visual Shares (Segmentation)":visual_cols
    }

    # Iterate through groups and plot
    for title, cols in plotting_groups.items():
        plot_cols = [col for col in cols if col in df.columns]

        if not plot_cols:
            log.warning(f"No columns found for group: {title}")
            continue

        n_plots = len(plot_cols)
        n_cols = min(n_plots, 3)
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()  # Flatten array for easy indexing

        for i, col in enumerate(plot_cols):
            ax = axes[i]
            # Use KDE plot to visualize distribution shape (for skewness and modality)
            sns.kdeplot(df[col].dropna(), ax=ax, fill=True, color='skyblue', alpha=0.6)
            # Add a vertical line for the mean
            ax.axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
            ax.set_title(f'Distribution of: {col}', fontsize=12)
            ax.set_xlabel(col)
            ax.set_ylabel("Density")
            ax.legend()

        for i in range(n_plots, len(axes)):
            fig.delaxes(axes[i])

        plt.suptitle(f'Raw Data Distributions: {title}', fontsize=16, y=1.02)
        plt.tight_layout()
        plot_path = output_dir / f"raw_distribution_{title.split('.')[0].replace(' ', '_')}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        log.info(f"Saved plot: {plot_path}")


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
    cax = divider.append_axes("bottom", size="3%", pad=1.75)
    norm = plt.Normalize(vmin=-1, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="vlag", norm=norm)
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cb.ax.set_xlabel("Spearman correlation (ρ)", fontsize=16, labelpad=10)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_visible(False)

    ax.set_aspect('equal')
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    ax.tick_params(axis='x', which='both', length=0)
    ax.tick_params(axis='y', which='both', length=0)

    ax.set_xticklabels([pretty_names.get(x.get_text(), x.get_text()) for x in ax.get_xticklabels()])
    ax.set_yticklabels([pretty_names.get(y.get_text(), y.get_text()) for y in ax.get_yticklabels()])

    xlabels = ax.get_xticklabels()
    if xlabels:
        xlabels[-1].set_visible(False)

    fig = plt.gcf()
    fig.suptitle(f"Multi-Modal Correlation Matrix (N={len(df_raw)})", fontsize=16, y=0.98)
    plt.tight_layout()

    plt.savefig(Path(output_dir, file_name), dpi=300, bbox_inches="tight")
    plt.close()


def plot_explained_variance(output_dir):
    """
    Plots the R2 for the specific narrative progression described in the text:
    1. Static (Planner View)
    2. Visual (Visual Cyclist)
    3. Dynamic (Temporal Cyclist - MLLM events)
    4. Combined (Visual-Temporal Cyclist)
    """
    df = pd.read_csv(Path(output_dir, "SEM_model_comparison.csv"))

    # We select the specific Model + Scenario combinations that tell the story
    # logic: (Scenario, Model, Display Label)
    target_chain = [
        ("Planner_View", "01_Static_Only", "M1-Static\n(GIS Only)"),
        ("Visual_Cyclist", "02_Visual_Only", "M2-Visual\n(Segmentation)"),
        ("Temporal_Cyclist", "03_Dynamic_Only", "M3-Temporal\n(MLLM Events)"),
        ("Visual_Temporal_Cyclist", "04_Full_Environment", "M4-Integrated\n(Vis+Temp)")
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

    fig, ax = plt.subplots(figsize=(6, 6))

    bars1 = ax.bar(x - width / 2, df_plot["R2_Valence"], width, label="Valence", color=COLOR_VALENCE)
    bars2 = ax.bar(x + width / 2, df_plot["R2_Arousal"], width, label="Arousal", color=COLOR_AROUSAL)

    # Annotate bars
    for b in bars1:
        yval = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, yval + 0.01, f"{yval:.2f}", ha='center', fontsize=10)

    for b in bars2:
        yval = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, yval + 0.01, f"{yval:.2f}", ha='center', fontsize=10)

    ax.set_ylabel("Explained variance ($R^2$)", fontsize=16)
    ax.set_title("Information Scenarios", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 0.6)  # Set limit slightly higher than max value (0.46)

    # Grid and Layout
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(fontsize=16, loc='upper left')

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

    print(f"Initial results dataframe size: {len(df_results)}, NaNs per column: {df_results.isna().sum()}")
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
    physio_arousal_cols = [
        'PPG_Rate_Mean',
        'SCL_Delta',
        'SCR_Peaks_N',
        'SCR_Peaks_Amplitude_Mean'
    ]
    physio_reg_cols = [
        'HRV_RMSSD'
    ]

    df_physio_mapped = map_physiological_segments_to_videos(df_physio, df_setup, c.TRIAL_1, c.VIDEO_COUNTS)
    df_physio_mapped = df_physio_mapped[[c.PARTICIPANT_ID, c.VIDEO_ID_COL] + physio_arousal_cols + physio_reg_cols]
    # ============================================================
    # 4. MERGE INTO MASTER DATAFRAME (RAW)
    # ============================================================
    df_raw = (df_ratings
              .merge(df_physio_mapped, on=[c.PARTICIPANT_ID, c.VIDEO_ID_COL], how="left")
              .merge(df_ground, on=c.VIDEO_ID_COL, how="left"))

    # TODO: Check how to fix this.
    # temporary fix for typo in cyclist count column
    df_raw['ped_and_cycl_count'] = df_raw['unique_pedestrians_count'] + df_raw['unique_cyclists_count']

    df_raw = df_raw[~df_raw[c.PARTICIPANT_ID].isin([0, 3, 12, 24])].copy()

    subj_cols = [
        "valence",
        "arousal"
    ]
    visual_cols = [
        "average_greenery_share",
        "average_sky_share",
        "average_building_share",
        "average_road_share"
    ]
    infra_cols = [
        "bike_infra_type_numeric",
        "tram_lane_presence",
        "bus_lane_presence",
        "one_way",
        "side_parking_count",
        "intersection_count",
        "tree_canopy_share",
        "building_count",
        "car_lanes_total_count",
        "slope",
        "traffic_volume",
        "motorized_traffic_speed_kmh",
        "pois_count",
    ]
    dynamic_cols = [
        "ped_and_cycl_count",
        "motor_vehicle_overtakes_count",
        "unique_motor_vehicles_count"
    ]

    #plot_raw_distributions(
    #    df_raw,
    #    subj_cols,
    #    physio_arousal_cols,
    #    physio_reg_cols,
    #    infra_cols,
    #    dynamic_cols,
    #    visual_cols,
    #    output_dir
    #)
    log.info("Visualization complete.")

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
                fontsize=10,
                fontweight="bold"
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
        ax.set_title("Videos in Valence–Arousal Space", fontsize=18)

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

        print(f"Saved improved visualization → {out_path}")

    plot_valence_arousal_scatter(df_raw, output_dir)
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
        "side_parking_count",
        "intersection_count",
        "building_count",
        "pois_count",
    ]

    for col in skewed_count_vars:
        if col in df_raw.columns:
            df_raw[col] = np.log1p(df_raw[col])

    # C. Mean centering within participants
    dep_vars = subj_cols + physio_arousal_cols + physio_reg_cols
    for col in dep_vars:
        df_raw[col] = df_raw[col] - df_raw.groupby(c.PARTICIPANT_ID)[col].transform('mean')

    # D. Global Z-score for all environment variables
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
        # PhysioArousal
        "PPG_Rate_Mean": "Heart rate (BPM)",
        "SCL_Delta": "Skin Conductance Δ",
        "SCR_Peaks_N": "SCR peaks (count)",
        "SCR_Peaks_Amplitude_Mean": "SCR amplitude (mean)",
        "HRV_RMSSD": "HRV RMSSD",
        # Infrastructure (static)
        "bike_infra_type_numeric": "Bike Infrastructure",
        "tram_lane_presence": "Tram lane presence",
        "bus_lane_presence": "Bus lane presence",
        "one_way": "One-way street",
        "side_parking_count": "Side Parking Count",
        "intersection_count": "Intersection Density",
        "tree_canopy_share": "Tree Canopy (%)",
        "building_count": "Building Counts",
        "car_lanes_total_count": "Car Lanes Count",
        "slope": "Street Slope (°)",
        "traffic_volume": "Traffic Volume (AADT)",
        "motorized_traffic_speed_kmh": "Speed Limit (km/h)",
        "pois_count": "POI Density",
        # Visual
        "average_greenery_share": "Greenery share",
        "average_sky_share": "Sky share",
        "average_building_share": "Building share",
        "average_road_share": "Road share",
        # Dynamic
        "ped_and_cycl_count": "Ped. & Cyclists Count",
        "motor_vehicle_overtakes_count": "Overtakes Count",
        "unique_motor_vehicles_count": "Vehicles Count"
    }

    corr_cols = subj_cols + physio_arousal_cols + physio_reg_cols + infra_cols + visual_cols + dynamic_cols
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

    SCENARIOS_CONFIG = {
        "Planner_View": {
            "infra_cols": ["bike_infra_type_numeric", "tram_lane_presence", "bus_lane_presence", "one_way",
                           "side_parking_count", "intersection_count", "pois_count"],
            "infra_ref": "bike_infra_type_numeric",  # High = Good Infra

            "visual_cols": ["tree_canopy_share", "building_count", "car_lanes_total_count", "slope"],
            "visual_ref": "tree_canopy_share",  # High = Greenery

            "dynamic_cols": ["traffic_volume", "motorized_traffic_speed_kmh"],
            "dynamic_ref": "traffic_volume"  # High = Heavy Traffic
        },

        "Visual_Cyclist": {
            "infra_cols": ["bike_infra_type_numeric", "tram_lane_presence", "bus_lane_presence", "one_way",
                           "side_parking_count", "intersection_count", "pois_count"],
            "infra_ref": "bike_infra_type_numeric",

            "visual_cols": ["average_greenery_share", "average_sky_share", "average_building_share",
                            "average_road_share"],
            "visual_ref": "average_greenery_share", # High = Greenery

            "dynamic_cols": ["traffic_volume", "motorized_traffic_speed_kmh"],
            "dynamic_ref": "traffic_volume"
        },

        "Temporal_Cyclist": {
            "infra_cols": ["bike_infra_type_numeric", "tram_lane_presence", "bus_lane_presence", "one_way",
                           "side_parking_count", "intersection_count", "pois_count"],
            "infra_ref": "bike_infra_type_numeric",

            "visual_cols": ["tree_canopy_share", "building_count", "car_lanes_total_count", "slope"],
            "visual_ref": "tree_canopy_share",

            "dynamic_cols": ["ped_and_cycl_count", "motor_vehicle_overtakes_count", "unique_motor_vehicles_count"],
            "dynamic_ref": "motor_vehicle_overtakes_count"  # High = More Danger
        },

        "Visual_Temporal_Cyclist": {
            "infra_cols": ["bike_infra_type_numeric", "tram_lane_presence", "bus_lane_presence", "one_way",
                           "side_parking_count", "intersection_count", "pois_count"],
            "infra_ref": "bike_infra_type_numeric",

            "visual_cols": ["average_greenery_share", "average_sky_share", "average_building_share",
                            "average_road_share"],
            "visual_ref": "average_greenery_share", # High = Greenery

            "dynamic_cols": ["ped_and_cycl_count", "motor_vehicle_overtakes_count", "unique_motor_vehicles_count"],
            "dynamic_ref": "motor_vehicle_overtakes_count"  # High = More Danger
        }
    }

    # ============================================================
    # 8. INDEX CREATION (SIGN–ALIGNED, EQUAL WEIGHTED)
    # ============================================================
    log.info("--- Calculating Sign-Aligned Equal-Weighted Indices ---")

    # Outcome Indices — unchanged
    physio_cols = physio_arousal_cols + physio_reg_cols

    physio_signs = {}
    for v in physio_cols:
        r = corr_matrix.loc[v, "valence"]
        physio_signs[v] = 1 if r >= 0 else -1
        df_raw[v + "_signed"] = df_raw[v] * physio_signs[v]

    df_raw["Physio_Index"] = df_raw[[v + "_signed" for v in physio_cols]].mean(axis=1)

    index_loading_report = []

    for scenario_name, config in SCENARIOS_CONFIG.items():
        print(f"\n\n=== BUILDING PCA INDICES FOR SCENARIO: {scenario_name} ===")

        # INFRA
        df_raw[f"InfraIndex_{scenario_name}"], var_inf, load_inf = create_aligned_index(
            df_raw, config["infra_cols"], f"InfraIndex_{scenario_name}", reference_col=config["infra_ref"]
        )
        index_loading_report.append(
            {"Scenario": scenario_name, "Type": "Infra", "Variance_PC1": var_inf, "Loadings": load_inf}
        )

        # VISUAL
        df_raw[f"VisualIndex_{scenario_name}"], var_vis, load_vis = create_aligned_index(
            df_raw, config["visual_cols"], f"VisualIndex_{scenario_name}", reference_col=config["visual_ref"]
        )
        index_loading_report.append(
            {"Scenario": scenario_name, "Type": "Visual", "Variance_PC1": var_vis, "Loadings": load_vis}
        )

        # DYNAMIC
        df_raw[f"DynamicIndex_{scenario_name}"], var_dyn, load_dyn = create_aligned_index(
            df_raw, config["dynamic_cols"], f"DynamicIndex_{scenario_name}", reference_col=config["dynamic_ref"]
        )
        index_loading_report.append(
            {"Scenario": scenario_name, "Type": "Dynamic", "Variance_PC1": var_dyn, "Loadings": load_dyn}
        )

    # Save loadings
    pd.DataFrame(index_loading_report).to_csv(output_dir / "index_loading_report.csv", index=False)

    sensitivity = run_sensitivity_analysis(df_raw, SCENARIOS_CONFIG, ["valence", "arousal"], output_dir)

    print("Nesting correction added: valence_wp, arousal_wp, Physio_Index_wp")
    log.info("--- Saved index_loading_report.csv ---")
    # ============================================================
    # 12. DEFINE MODEL CATALOG (Final & Complete)
    # ============================================================
    MODEL_CATALOG = {
        # ---------------------------------------------------------
        # FAMILY 1: THE BASICS (Direct Effects)
        # ---------------------------------------------------------
        "01_Static_Only": """
            valence ~ InfraIndex
            arousal ~ InfraIndex
            valence ~~ arousal
        """,
        "02_Visual_Only": """
            valence ~ VisualIndex
            arousal ~ VisualIndex
            valence ~~ arousal
        """,
        "03_Dynamic_Only": """
            valence ~ DynamicIndex
            arousal ~ DynamicIndex
            valence ~~ arousal
        """,
        "04_Full_Environment": """
            valence ~ InfraIndex + VisualIndex + DynamicIndex
            arousal ~ InfraIndex + VisualIndex + DynamicIndex
            valence ~~ arousal
        """,

        # ---------------------------------------------------------
        # FAMILY 2: MEDIATION (The Mechanism)
        # ---------------------------------------------------------
        "05_Full_Mediation": """
            VisualIndex  ~ InfraIndex
            DynamicIndex ~ InfraIndex
            valence      ~ VisualIndex + DynamicIndex
            arousal      ~ VisualIndex + DynamicIndex
            valence ~~ arousal
        """,

        # ---------------------------------------------------------
        # FAMILY 3: PHYSIOLOGY (Mind-Body Connection)
        # ---------------------------------------------------------

        "06_Physio_Parallel": """
            # Inputs
            InfraIndex ~~ VisualIndex
            InfraIndex ~~ DynamicIndex
            VisualIndex ~~ DynamicIndex

            # Env -> Mind
            valence ~ InfraIndex + VisualIndex + DynamicIndex
            arousal ~ InfraIndex + VisualIndex + DynamicIndex

            # Env -> Body (Both Stress and Safety)
            Physio_Index    ~ InfraIndex + VisualIndex + DynamicIndex
           
            # Residual Correlations
            valence ~~ arousal
            valence ~~ Physio_Index
            arousal ~~ Physio_Index
        """,

        "07_Causal_Body_to_Mind": """
            # Env -> Body
            Physio_Index    ~ InfraIndex + VisualIndex + DynamicIndex

            # Body -> Mind
            valence ~ Physio_Index
            arousal ~ Physio_Index

            valence ~~ arousal
        """,

        "08_Causal_Mind_to_Body": """
            # Env -> Mind
            valence ~ InfraIndex + VisualIndex + DynamicIndex
            arousal ~ InfraIndex + VisualIndex + DynamicIndex

            # Mind -> Body
            Physio_Index    ~ arousal + valence
    
            valence ~~ arousal
        """,

        # ---------------------------------------------------------
        # FAMILY 4: INTERACTIONS (The Buffering Effects)
        # ---------------------------------------------------------

        "09_Mod_Infra_Buffers_Traffic": """
            # Interaction Effects
            valence                 ~ InfraIndex + DynamicIndex + Infra_Dynamic_Interaction
            arousal                 ~ InfraIndex + DynamicIndex + Infra_Dynamic_Interaction
            Physio_Index    ~ InfraIndex + DynamicIndex + Infra_Dynamic_Interaction

            # Control
            valence ~ VisualIndex
            arousal ~ VisualIndex
        """,

        "10_Mod_Beauty_Buffers_Traffic": """
            # Interaction Effects
            valence                 ~ VisualIndex + DynamicIndex + Visual_Dynamic_Interaction
            arousal                 ~ VisualIndex + DynamicIndex + Visual_Dynamic_Interaction
            Physio_Index            ~ VisualIndex + DynamicIndex + Visual_Dynamic_Interaction

            # Control
            valence ~ InfraIndex
            arousal ~ InfraIndex
        """,

        "11_Mod_Beauty_Enhances_Infra": """
            # Interaction Effects
            valence                 ~ InfraIndex + VisualIndex + Infra_Visual_Interaction
            arousal                 ~ InfraIndex + VisualIndex + Infra_Visual_Interaction
            Physio_Index            ~ InfraIndex + VisualIndex + Infra_Visual_Interaction

            # Control
            valence ~ DynamicIndex
            arousal ~ DynamicIndex
        """
    }

    # ============================================================
    # 13. EXECUTION LOOP (With Full Fit Metrics)
    # ============================================================
    log.info("--- Running Scenario Comparison Loop ---")

    FIT_METRICS = ["DoF", "chi2", "chi2 p-value", "CFI", "TLI", "RMSEA", "AIC", "BIC"]
    comparison_results = []

    for scenario_name in SCENARIOS_CONFIG.keys():
        print(f"\n>> PROCESSING SCENARIO: {scenario_name}")

        # A. Prepare Data
        df_model = df_raw.copy()
        rename_map = {
            f"InfraIndex_{scenario_name}": "InfraIndex",
            f"VisualIndex_{scenario_name}": "VisualIndex",
            f"DynamicIndex_{scenario_name}": "DynamicIndex"
        }
        df_model = df_model.rename(columns=rename_map)

        # -------------------------------------------------------
        # RUN LiNGAM FOR THIS SCENARIO
        # -------------------------------------------------------
        lingam_cols = [
            'InfraIndex',
            'VisualIndex',
            'DynamicIndex',
            'Physio_Index',
            'valence',
            'arousal'
        ]
        run_lingam_discovery(df_model, lingam_cols, output_dir, scenario_name)
        # -------------------------------------------------------

        # B. Calculate Interactions
        df_model["Infra_Dynamic_Interaction"] = df_model["InfraIndex"] * df_model["DynamicIndex"]
        df_model["Visual_Dynamic_Interaction"] = df_model["VisualIndex"] * df_model["DynamicIndex"]
        df_model["Infra_Visual_Interaction"] = df_model["InfraIndex"] * df_model["VisualIndex"]

        # 2. Run Models
        for model_name, syntax in MODEL_CATALOG.items():
            run_name = f"{scenario_name}_{model_name}"
            model, fit_stats = run_sem_model(syntax, df_model, run_name, output_dir, plot_graph=True)

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

                comparison_results.append(res)

    df_compare = pd.DataFrame(comparison_results)
    df_compare.to_csv(output_dir / "SEM_model_comparison.csv", index=False)

    log.info("SEM modeling complete. Check output directory for results.")

    plot_explained_variance(output_dir)

if __name__ == "__main__":
    main()
