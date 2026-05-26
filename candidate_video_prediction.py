import configparser
from typing import Optional, Tuple
import numpy as np
from matplotlib.lines import Line2D
from scipy.spatial import ConvexHull
from sklearn.cluster import KMeans
from sklearn.metrics import root_mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import sys
import constants as c
import utils.helper_functions
import seaborn as sns
import utils.processing_utils
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


sys.path.append(str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

sns.set_context("paper", font_scale=1.0)
sns.set_style("white")


# ==============================================================================
# UNIFIED PLOT STYLE
# ==============================================================================

STYLE = {
    'palette': 'YlGnBu',
    'cat_palette': 'Set2',
    'figsize_wide': (10, 4),
    'figsize_square': (6, 6),
    'font_title': 14,
    'font_axis_label': 12,
    'font_tick': 9,
    'font_legend': 9,
    'font_annotation': 7,
    'dpi': 200,
    'grid_alpha': 0.25,
    'line_color_neutral': 'grey',
    'line_color_boundary': 'grey',
    'marker_size': 60,
}


def _apply_axis_style(ax, title=None, xlabel=None, ylabel=None):
    """Apply consistent title/label/tick styling to an Axes."""
    if title:
        ax.set_title(title, fontsize=STYLE['font_title'], pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=STYLE['font_axis_label'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=STYLE['font_axis_label'])
    ax.tick_params(axis='both', labelsize=STYLE['font_tick'])
    ax.grid(True, linestyle='--', alpha=STYLE['grid_alpha'])


def perform_clustering(video_df, data_cols, cluster_target, corr_list):
    """
    Perform KMeans clustering and print cluster compositions and correlations.
    :param video_df: DataFrame with video data
    :param data_cols: list of columns to use for clustering
    :param cluster_target: name of the new cluster column
    :param corr_list: list of columns to compute correlations with
    :return: DataFrame with cluster assignments
    """
    X = video_df[data_cols].copy()
    kmeans = KMeans(n_clusters=2, random_state=42)
    video_df[cluster_target] = kmeans.fit_predict(X)

    # Cluster video composition
    cluster_videos = video_df.groupby(cluster_target)[c.VIDEO_ID_COL].apply(list)
    for cluster_label, videos in cluster_videos.items():
        print(f"\nCluster {cluster_label} contains {len(videos)} videos:")
        print(videos)

    # Create separate sorted correlation DataFrames
    for i in corr_list:
        print(video_df[data_cols].corrwith(video_df[i]).sort_values(ascending=False).to_frame(name=f'{i} corr'))
        print()
    return video_df


def compute_k_rmse(X, video_df, target_col=c.VALENCE):
    """
    Compute RMSE for k-NN predictions across different k values to find the optimal k.
    :param X: feature DataFrame
    :param video_df: DataFrame with video data
    :param target_col: name of the target column
    :return: DataFrame with RMSE values for each k, best RMSE, and best k
    """
    rmses = []
    for k in range(1, len(X) - 1):
        preds = []
        gt = []
        for vid in X.index:
            pred = get_neighbor_prediction(X, vid, video_df, k, target_col)
            preds.append(pred)
            gt.append(video_df.loc[vid, target_col])
        rmse_k = root_mean_squared_error(gt, preds)
        rmses.append(rmse_k)

    rmses_df = (pd.DataFrame({'k': range(1, len(X) - 1), 'rms': rmses})
                .set_index('k'))

    best_k = rmses_df['rms'].idxmin()
    best_rmse = rmses_df.loc[best_k, 'rms']
    print(f"Global best k = {best_k}")
    print(f"Global best RMSE = {best_rmse}")

    return rmses_df, best_rmse, best_k


def predict_candidates(X, video_df, candidate_df, feature_cols, target_col, k, id_col=c.VIDEO_ID_COL, negate=False):
    """
    Predict valence for candidate videos using k-NN trained on existing video data.
    :param X: feature DataFrame for training videos
    :param video_df: DataFrame with training video data
    :param candidate_df: DataFrame with candidate video data
    :param feature_cols: list of feature columns to use
    :param target_col: name of the target column
    :param k: number of neighbors
    :param id_col: name of the ID column
    :param negate: whether to negate predictions
    :return: DataFrame with candidate IDs and predicted valence
    """
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', metric='manhattan')
    knn.fit(X, video_df.loc[X.index, target_col])

    rows = []
    for vid in candidate_df.index:
        pred = knn.predict(candidate_df.loc[vid, feature_cols].to_frame().T)[0]
        if negate:
            pred = -pred
        rows.append({id_col: candidate_df.loc[vid, id_col], 'valence_prediction': pred})

    results_df = (pd.DataFrame(rows).sort_values(by='valence_prediction', ascending=False).reset_index(drop=True)
                  )
    return results_df


def get_neighbor_prediction(
        X: pd.DataFrame,
        video: str,
        video_df: pd.DataFrame,
        k: int,
        target_col: str = c.VALENCE
) -> float:
    """
    Get k-NN prediction for a specific video.
    :param X: feature DataFrame
    :param video: video ID to predict
    :param video_df: DataFrame with video data
    :param k: number of neighbors
    :param target_col: name of the target column
    :return: predicted value
    """
    curr_X = X.drop(index=[video])
    ratings_X = video_df.loc[curr_X.index, target_col]
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance', metric='manhattan')
    knn.fit(curr_X, ratings_X)
    prediction = knn.predict(X.loc[video, :].to_frame().T)[0]

    return prediction


def get_top_labels(df, target_col, label_cols, top_n=5):
    """
    Finds the top correlating survey labels (PF or NF).
    If a core feature (e.g., 'Overtakes') appears as both PF and NF,
    it keeps only the variant with the highest absolute correlation,
    returning exactly the top N highest-correlating unique features.
    """

    corrs = df[label_cols].corrwith(df[target_col]).abs()
    corr_df = corrs.reset_index()
    corr_df.columns = ['original_col', 'abs_corr']
    corr_df['core_feature'] = corr_df['original_col'].str.replace(r'^(PF|NF):\s*', '', regex=True)
    corr_df = corr_df.sort_values(by='abs_corr', ascending=False)
    deduped_df = corr_df.drop_duplicates(subset=['core_feature'], keep='first')
    top_features_df = deduped_df.head(top_n)
    final_label_cols = top_features_df['original_col'].tolist()

    log.info(f"Selected top {len(final_label_cols)} unique PF/NF columns:")

    for _, row in top_features_df.iterrows():
        log.info(f" - {row['original_col']} (|r| = {row['abs_corr']:.3f})")

    return final_label_cols


def filter_by_distance_to_boundary(
        df: pd.DataFrame,
        cluster_col: str,
        valence_col: str,
        threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Retain rows whose valence is at least `threshold` from the cluster boundary
    (defined as the midpoint between the two cluster centroids in valence).

    Returns (retained, excluded, boundary).
    """
    if df[cluster_col].nunique() != 2:
        raise ValueError(f"Expected 2 clusters in {cluster_col}, got {df[cluster_col].nunique()}.")

    centroids = df.groupby(cluster_col)[valence_col].mean()
    boundary = float(centroids.mean())

    df = df.copy()
    df['dist_to_boundary'] = (df[valence_col] - boundary).abs()
    retained = df[df['dist_to_boundary'] >= threshold].copy()
    excluded = df[df['dist_to_boundary'] < threshold].copy()

    return retained, excluded, boundary


def plot_cluster_separation(
        df: pd.DataFrame,
        cluster_col: str,
        valence: str,
        arousal: str,
        id_col: str = c.VIDEO_ID_COL,
        save_path: Optional[Path] = None,
) -> None:
    """
    Plot clusters in valence-arousal space with convex hulls and point labels.
    """
    cmap = plt.get_cmap(STYLE['palette'])
    clusters = sorted(df[cluster_col].dropna().unique())
    cluster_colors = {clusters[0]: cmap(0.25), clusters[-1]: cmap(0.85)}

    fig, ax = plt.subplots(figsize=STYLE['figsize_square'])

    # Convex hulls (drawn first so points sit on top)
    for cl in clusters:
        sub = df[df[cluster_col] == cl]
        if len(sub) >= 3:
            pts = sub[[valence, arousal]].to_numpy()
            try:
                hull = ConvexHull(pts)
                hp = pts[hull.vertices]
                ax.fill(hp[:, 0], hp[:, 1],
                        color=cluster_colors[cl], alpha=0.15, edgecolor='none',
                        zorder=1)
            except Exception:
                pass

    # Cluster points (one seaborn call for both clusters)
    plot_df = df.copy()
    plot_df['_cluster_str'] = plot_df[cluster_col].apply(lambda x: f'Cluster {x}')
    palette_named = {f'Cluster {cl}': cluster_colors[cl] for cl in clusters}

    sns.scatterplot(
        data=plot_df, x=valence, y=arousal,
        hue='_cluster_str',
        palette=palette_named,
        s=STYLE['marker_size'] + 30,
        edgecolor='white', linewidth=0.8,
        alpha=0.9,
        ax=ax, zorder=3,
    )

    # Point labels
    if id_col in plot_df.columns:
        for _, r in plot_df.iterrows():
            ax.text(r[valence], r[arousal], str(r[id_col]),
                    fontsize=STYLE['font_annotation'], ha='center', va='bottom',
                    color=STYLE['line_color_boundary'], zorder=4)

    ax.axhline(0, color=STYLE['line_color_neutral'], lw=0.6, zorder=0)
    ax.axvline(0, color=STYLE['line_color_neutral'], lw=0.6, zorder=0)

    _apply_axis_style(
        ax,
        title='Valence–arousal clusters',
        xlabel=valence.capitalize(),
        ylabel=arousal.capitalize(),
    )

    ax.legend(title='', loc='best',
              fontsize=STYLE['font_legend'], frameon=False)

    sns.despine(ax=ax)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=STYLE['dpi'], bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_correlation_heatmap(
        df: pd.DataFrame,
        feature_cols: list,
        title: str,
        save_path: Path = None,
) -> None:
    """
    Plot a correlation matrix heatmap with a diverging palette centred on zero,
    which is the right idiom for correlation values that span both signs.
    """
    corr_matrix = df[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))

    diverging = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=np.triu(np.ones_like(corr_matrix, dtype=bool), k=1),
        annot=True,
        fmt=".2f",
        cmap=diverging,
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        square=True,
        cbar_kws={'shrink': 0.7, 'label': 'Correlation'},
        annot_kws={'size': STYLE['font_tick']},
        ax=ax,
    )

    ax.set_title(title,
                 fontsize=STYLE['font_title'], pad=12)
    ax.tick_params(axis='both', labelsize=STYLE['font_tick'])
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=STYLE['dpi'], bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


def plot_paired_valence_panels(
        validated_df: pd.DataFrame,
        candidates_df: pd.DataFrame,
        threshold: float,
        boundary: float,
        valence_col_validated: str = c.VALENCE,
        valence_col_candidates: str = 'valence_prediction',
        id_col: str = c.VIDEO_ID_COL,
        save_path: Optional[Path] = None,
):
    """
    Two-panel figure: validated clips (measured valence) on top,
    archive candidates (predicted valence) on bottom. Same boundary,
    same threshold, same colour grammar — the visual symmetry mirrors
    the methodological symmetry described in the text.
    """
    cmap = plt.get_cmap(STYLE['palette'])
    upper = boundary + threshold
    lower = boundary - threshold

    # Sort BOTH ascending so axis grammar matches
    v_df = validated_df.sort_values(valence_col_validated).reset_index(drop=True)
    c_df = candidates_df.sort_values(valence_col_candidates).reset_index(drop=True)

    def _categorise(values):
        return np.select(
            [values > upper, values < lower],
            ['Bikeable', 'Non-bikeable'],
            default='Ambiguous',
        )

    v_df['rank'] = range(len(v_df))
    v_df['category'] = _categorise(v_df[valence_col_validated].to_numpy())
    c_df['rank'] = range(len(c_df))
    c_df['category'] = _categorise(c_df[valence_col_candidates].to_numpy())

    palette = {
        'Bikeable': cmap(0.85),
        'Non-bikeable': cmap(0.25),
        'Ambiguous': cmap(0.55),
    }
    markers = {'Bikeable': '^', 'Non-bikeable': 's', 'Ambiguous': 'o'}
    hue_order = ['Non-bikeable', 'Ambiguous', 'Bikeable']

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharey=True)

    for ax, df_panel, vcol, label, ylabel in [
        (axes[0], v_df, valence_col_validated, "Validated clips (measured valence)", "Measured valence"),
        (axes[1], c_df, valence_col_candidates, "Archive candidates (predicted valence)", "Predicted valence"),
    ]:
        sns.scatterplot(
            data=df_panel, x='rank', y=vcol,
            hue='category', hue_order=hue_order,
            style='category', style_order=hue_order,
            palette=palette, markers=markers,
            s=STYLE['marker_size'] + 20,
            edgecolor='white', linewidth=0.5,
            ax=ax, legend=(ax is axes[0]),  # legend on top panel only
        )
        ax.axhline(boundary, color=STYLE['line_color_boundary'], linestyle='-', lw=1.2, zorder=0)
        ax.axhline(upper, color=STYLE['line_color_neutral'], linestyle=(0, (8, 4)), lw=0.9, zorder=0)
        ax.axhline(lower, color=STYLE['line_color_neutral'], linestyle=(0, (8, 4)), lw=0.9, zorder=0)

        ax.set_ylim(-1, 1)
        ax.set_xticks(range(len(df_panel)))
        ax.set_xticklabels(df_panel[id_col].astype(str).tolist(),
                           rotation=90, ha='center', fontsize=STYLE['font_tick'])

        _apply_axis_style(ax, title=label, xlabel="Clip ID", ylabel=ylabel)
        sns.despine(ax=ax)

    # Single legend on top panel, extended with boundary + RMSE handles
    handles, labels = axes[0].get_legend_handles_labels()
    handles.extend([
        Line2D([0], [0], color=STYLE['line_color_boundary'], lw=1.2, label='Boundary'),
        Line2D([0], [0], color=STYLE['line_color_neutral'], linestyle=(0, (8, 4)), lw=0.9, label='± 1 RMSE'),
    ])
    labels.extend(['Boundary', '± 1 RMSE'])
    axes[0].legend(handles=handles, labels=labels,
                   loc='upper left', fontsize=STYLE['font_legend'], frameon=False, ncol=1)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=STYLE['dpi'], bbox_inches='tight')
        plt.close()
    else:
        plt.show()


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

    # ==============================================================================
    # PHASE 1: LOAD DATA
    # ==============================================================================
    survey_results_file = Path(config["filenames"]["survey_results_file"])
    sequence_file = Path(config["filenames"]["online_sequence_file"])
    predicted_valences_file = Path(config['filenames']['video_predictions_file'])
    ground_truth_file = Path(config["filenames"]["video_info_ground_truth"])
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==============================================================================
    # PHASE 2: PROCESS SURVEY DATA
    # ==============================================================================
    log.info("Phase 2: Processing survey data")

    survey_df = pd.read_excel(survey_results_file).set_index(c.PARTICIPANT_ID)
    seq_df = pd.read_csv(sequence_file, parse_dates=['seq_start', 'seq_end'])
    video_ground_truth_features = pd.read_csv(ground_truth_file)

    survey_results_df = utils.processing_utils.transform_to_long_df(survey_df, seq_df, id_col=c.PARTICIPANT_ID)
    survey_results_df = utils.processing_utils.filter_results(survey_results_df)
    survey_results_df = utils.processing_utils.add_valence_arousal(survey_results_df)

    video_level_scores = utils.processing_utils.calculate_video_level_scores(survey_results_df)

    #feature_cols = ['bike_infra_type', 'slope', 'car_lanes_total_count', 'traffic_volume',
    #                'motorized_traffic_speed_kmh', 'average_building_share', 'average_greenery_share',
    #                'motor_vehicle_overtakes_count', 'ped_and_cycl_count', 'surface_type']

    #video_ids = survey_results_df[c.VIDEO_ID_COL].unique()
    #filtered_features = video_ground_truth_features[video_ground_truth_features[c.VIDEO_ID_COL].isin(video_ids)].copy()

    #utils.plotting_utils.plot_transposed_scenario_heatmap(
    #    df=filtered_features,
    #    feature_cols=feature_cols,
    #    id_col=c.VIDEO_ID_COL,
    #    categorical_maps=c.CATEGORICAL_MAPPINGS,
    #    save_path=output_dir / "scenario_typology_heatmap.png"
    #)

    # ==============================================================================
    # PHASE 3: SELECT FEATURES FROM TOP TAG CORRELATES
    # ==============================================================================
    log.info("Phase 3: Selecting features from top PF/NF tag correlates")

    video_level_scores = utils.processing_utils.add_factor_counts_to_scores(
        scores_df=video_level_scores,
        survey_df=survey_results_df,
        label_cols=c.LABEL_COLS,
        video_col=c.VIDEO_ID_COL,
        pf_col=c.PF,
        nf_col=c.NF,
    )
    video_level_scores = video_level_scores.rename(
        columns={'valence_mean': c.VALENCE, 'arousal_mean': c.AROUSAL}
    )

    top_label_cols = get_top_labels(video_level_scores, c.VALENCE, c.LABEL_COLS, 5)

    plot_correlation_heatmap(
        video_level_scores,
        top_label_cols + [c.VALENCE, c.AROUSAL],
        'Correlation matrix of environmental features',
        save_path=output_dir / "Label_Correlation_Heatmap.png",
    )

    # ==============================================================================
    # PHASE 4: CLUSTER VIDEOS BY MEASURABLE ENVIRONMENTAL FEATURES
    # ==============================================================================
    log.info("Phase 4: Clustering videos on measurable environmental features")

    scaler = MinMaxScaler()
    video_ground_truth_features[c.DATA_COLS] = scaler.fit_transform(
        video_ground_truth_features[c.DATA_COLS]
    )

    data_cluster_df = video_level_scores.merge(
        video_ground_truth_features, on=c.VIDEO_ID_COL, how='left'
    )
    plot_correlation_heatmap(
        data_cluster_df,
        c.DATA_COLS + [c.VALENCE, c.AROUSAL],
        'Correlation matrix of environmental features',
        save_path=output_dir / "Correlation_Heatmap.png",
    )

    data_cluster_df = perform_clustering(
        data_cluster_df, c.DATA_COLS, 'Cluster_data', c.DEPENDENT_VARIABLES
    )
    plot_cluster_separation(
        data_cluster_df, 'Cluster_data', c.VALENCE, c.AROUSAL,
        id_col=c.VIDEO_ID_COL,
        save_path=output_dir / "Data Clusters scatterplots.png",
    )

    # ==============================================================================
    # PHASE 5: PREDICTIVE MODEL + SYMMETRIC THRESHOLD FILTERING
    # ==============================================================================
    log.info("Phase 5: Training predictive model and applying symmetric threshold")

    # Fit k-NN once; use the LOO-RMSE as the symmetric threshold
    X = data_cluster_df[c.DATA_COLS].copy()
    rmses_df, best_rmse, best_k = compute_k_rmse(X, data_cluster_df, c.TARGET_COL)
    log.info(f"LOO RMSE: {best_rmse:.3f} (k={best_k})")

    # Threshold filter on validated set (1 x RMSE from boundary in valence)
    retained_validated, excluded_validated, boundary = filter_by_distance_to_boundary(
        data_cluster_df,
        cluster_col='Cluster_data',
        valence_col=c.VALENCE,
        threshold=best_rmse,
    )
    log.info(f"Cluster boundary in valence: {boundary:.3f}")
    log.info(f"Validated clips retained: {len(retained_validated)}/{len(data_cluster_df)}")
    if len(excluded_validated) > 0:
        log.info(
            f"Excluded validated clips (near boundary): "
            f"{excluded_validated[[c.VIDEO_ID_COL, c.VALENCE, 'dist_to_boundary']].to_dict('records')}"
        )

    # Predict valence for archive candidates
    ids_to_exclude = data_cluster_df[c.VIDEO_ID_COL].unique()
    candidate_video_df = video_ground_truth_features.loc[
        ~video_ground_truth_features[c.VIDEO_ID_COL].isin(ids_to_exclude)
    ]
    predicted_valences = predict_candidates(
        X, data_cluster_df, candidate_video_df,
        c.DATA_COLS, c.TARGET_COL, best_k,
    )

    # Same threshold on archive predictions
    predicted_valences['dist_to_boundary'] = (predicted_valences['valence_prediction'] - boundary).abs()
    predicted_valences['classification'] = np.select(
        [
            (predicted_valences['valence_prediction'] > boundary)
            & (predicted_valences['dist_to_boundary'] >= best_rmse),
            (predicted_valences['valence_prediction'] < boundary)
            & (predicted_valences['dist_to_boundary'] >= best_rmse),
        ],
        ['bikeable_extension', 'non_bikeable_extension'],
        default='ambiguous',
    )
    log.info(f"Archive classification:\n{predicted_valences['classification'].value_counts()}")

    n_extensions = (predicted_valences['classification'] != 'ambiguous').sum()
    log.info(
        f"Final stimulus set composition: {len(retained_validated)} validated "
        f"+ {n_extensions} extensions = {len(retained_validated) + n_extensions} clips"
    )

    # ==============================================================================
    # PHASE 6: SAVE AND VISUALIZE RESULTS
    # ==============================================================================
    log.info("Phase 6: Saving outputs")
    #plot_candidate_predictions(
    #    predicted_valences, best_rmse,
    #    bounds=(boundary, boundary),
    #    save_path=output_dir / "Candidate videos valence predictions.png",
    #)

    #plot_clip_valences(
    #    data_cluster_df.sort_values(c.VALENCE),
    #    valence_col=c.VALENCE,
    #    threshold=best_rmse,
    #    boundary=boundary,
    #    title="Validated clips: measured valence",
    #    ylabel="Measured valence",
    #    save_path=output_dir / "Validated_clips_valences.png",
    #)

    plot_paired_valence_panels(
        validated_df=data_cluster_df,
        candidates_df=predicted_valences,
        threshold=best_rmse,
        boundary=boundary,
        save_path=output_dir / "Valence_filter_panels.png",
    )

    predicted_valences.to_csv(predicted_valences_file, index=False)
    retained_validated.to_csv(output_dir / "validated_clips_retained.csv", index=False)
    excluded_validated.to_csv(output_dir / "validated_clips_excluded.csv", index=False)
    log.info(f"Saved predictions to {predicted_valences_file}")

    log.info("Analysis pipeline finished successfully!")


if __name__ == "__main__":
    main()
