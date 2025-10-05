import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from typing import Tuple, Dict, Optional, List
from collections import Counter
import constants as c
from PIL import Image


# ---  Plotting functions for survey data analysis ---
def plot_overall_experience(
        df: pd.DataFrame,
        oe_col: str,
        oe_order: List[str],
        video_col: str = c.VIDEO_ID_COL,
        save_path: Optional[Path] = None
) -> None:
    """
    Plot a stacked bar chart showing the distribution of Overall Experience (OE) ratings across videos.
    :param df: DataFrame containing the video and overall experience data.
    :param oe_col: Overall Experience rating column name.
    :param oe_order: order of overall experience rating categories.
    :param video_col: Video ID column name.
    :param save_path: Optional file path to save the figure.
    :return: None
    """
    df[oe_col] = pd.Categorical(df[oe_col], categories=oe_order, ordered=True)
    counts = df.groupby([video_col, oe_col], observed=False).size().unstack(fill_value=0)[oe_order]
    ax = counts.plot(kind="bar", stacked=True, figsize=(len(counts) * 0.6 + 2, 6),
                     color=sns.color_palette("YlGnBu", n_colors=len(oe_order)))
    ax.set_ylabel("Count", fontsize=14)
    ax.set_xlabel("Video ID", fontsize=14)
    ax.set_title("Overall Experience", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.legend(title=oe_col, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=len(oe_order), fontsize=12)
    fig = plt.gcf()

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        fig.tight_layout()
        plt.show()


def plot_affect_grid(
        df: pd.DataFrame,
        video_id: Optional[str] = None,
        save_path: Optional[Path] = None
) -> None:
    """
    Plot a heatmap of the Affect Grid for a given video or all videos in the DataFrame.
    :param save_path: Optional file path to save the figure.
    :param df: DataFrame containing the Affect Grid data.
    :param video_id: Optional video ID to filter the DataFrame. If None, all videos will be plotted.
    :return: None
    """
    if video_id is None:
        vids = sorted(df[c.VIDEO_ID_COL].dropna().unique())
    elif isinstance(video_id, (list, tuple, set, pd.Series, np.ndarray)):
        vids = sorted(set(int(v) for v in video_id))
    else:
        vids = [int(video_id)]

    for vid in vids:
        ag = pd.to_numeric(df.loc[df[c.VIDEO_ID_COL] == vid, c.AG], errors='coerce')
        ag = ag[(ag >= 1) & (ag <= 100)].astype(int)

        counts = ag.value_counts().reindex(range(1, 101), fill_value=0).sort_index()
        grid = counts.values.reshape(10, 10)
        heatmap_data = pd.DataFrame(grid, index=range(1, 11), columns=range(1, 11))

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            cbar_kws={'label': 'Choice Count'},
            xticklabels=False,
            yticklabels=False
        )
        plt.title(f'Affect Grid Heatmap (Video {vid})', fontsize=14)
        fig = plt.gcf()

        if save_path:
            # Construct a unique path for THIS video inside the loop
            filename = f"affect_grid_video_{vid}.png"
            full_path = Path(save_path) / filename

            fig.tight_layout()
            fig.savefig(full_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            fig.tight_layout()
            plt.show()


def plot_oe_distribution_by_fam(
        df: pd.DataFrame,
        f_order: List[str],
        oe_order: List[str],
        f_col: str = c.F,
        oe_col: str = c.OE,
        save_path: Optional[Path] = None
) -> None:
    """
    Plot the distribution of Overall Experience (OE) ratings by Familiarity (F).
    If a save_path is provided, the plot is saved; otherwise, it is shown.

    :param df: DataFrame containing the familiarity and overall experience data.
    :param f_order: Order of familiarity categories.
    :param oe_order: Order of overall experience rating categories.
    :param f_col: Familiarity rating column name.
    :param oe_col: Overall experience rating column name.
    :param save_path: Optional file path to save the figure.
    """
    d = df[[f_col, oe_col]].dropna().copy()
    d[f_col] = pd.Categorical(d[f_col], categories=f_order, ordered=True)
    d[oe_col] = pd.Categorical(d[oe_col], categories=oe_order, ordered=True)

    tab = (d.groupby([f_col, oe_col], observed=False)
           .size()
           .unstack(fill_value=0)
           .reindex(index=f_order, columns=oe_order, fill_value=0))

    pct = tab.div(tab.sum(axis=1), axis=0) * 100
    ax = pct.plot(kind="barh", stacked=True, figsize=(10, 3),
                  color=sns.color_palette("YlGnBu", n_colors=len(oe_order)))

    fig = ax.get_figure()

    ax.set_xlabel("Percentage")
    ax.set_ylabel("Familiarity")
    ax.set_title("OE distribution by Familiarity (normalized)")
    ax.legend(title=c.OE, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
    else:
        fig.tight_layout()
        plt.show()


def generate_demographic_summary(
        df: pd.DataFrame,
        columns: List[str],
        orders: Optional[Dict[str, List[str]]] = None,
        save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate a demographic summary table and corresponding bar plots for specified columns.
    :param df: DataFrame containing the demographic data.
    :param columns: List of column names to include in the summary and plots.
    :param orders: Optional dictionary specifying the order of categories for certain columns.
    :param save_path: Optional file path to save the figure.
    :return: DataFrame summarizing counts and percentages for each category in the specified columns.
    """

    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8), sharex=False)
    axes = axes.ravel()
    k = min(6, len(columns))
    summary_rows = []

    for i, col in enumerate(columns[:k]):
        ax = axes[i]

        unique_participants_df = df.drop_duplicates(subset=[c.PARTICIPANT_ID])
        s = unique_participants_df[col].dropna().astype(str)

        if orders and col in orders:
            cat_order = [x for x in orders[col] if x in s.unique()]
            s = s.astype(pd.CategoricalDtype(cat_order, ordered=True))
            vc = s.value_counts(sort=False)
        else:
            vc = s.value_counts()

        summary_chunk = pd.DataFrame({
            "variable": col,
            "level": vc.index,
            "counts": vc.values,
            "pct": (vc / vc.sum() * 100).round(1).values
        })
        summary_rows.append(summary_chunk)

        plot_data = summary_chunk[['level', 'pct']]
        sns.barplot(data=plot_data, x="pct", y="level", hue="level",
                    palette=sns.color_palette("viridis", n_colors=len(plot_data)),
                    dodge=False, legend=False, ax=ax, orient="h")

        ax.set(title=col.replace('_', ' ').title(), xlabel="Percent of Participants", ylabel="")
        ax.set_xlim(0, 100)
        ax.grid(axis="x", alpha=.2)
        for sp in ax.spines.values(): sp.set_visible(False)
        for y, v in enumerate(plot_data.pct):
            ax.text(min(v, 98), y, f"{v:.1f}%", va="center",
                    ha=("right" if v > 15 else "left"),
                    color=("white" if v > 15 else "black"), fontsize=9)

    for j in range(k, 6):
        axes[j].axis("off")

    if save_path:
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)

    final_summary_table = pd.concat(summary_rows, ignore_index=True)
    return final_summary_table


# --- Plotting functions for video clustering and selection ---

def plot_target_comparisons(
        video_df: pd.DataFrame,
        oe_metrics: pd.DataFrame,
        cluster_col: pd.DataFrame,
        save_path: Optional[Path] = None
) -> None:
    """
    Plot boxplots of Overall Experience (OE) metrics by cluster and video id.
    :param save_path: Optional file path to save the figure.
    :param video_df: DataFrame containing video data with clusters and overall experience metrics.
    :param oe_metrics: List of Overall Experience metrics to plot.
    :param cluster_col: Column name for the cluster labels in the video DataFrame.
    """

    plt.figure(figsize=(15, 5))
    clusters = sorted(video_df[cluster_col].unique())
    cluster_palette = sns.color_palette("Set2", len(clusters))
    cluster_colors = {label: cluster_palette[i] for i, label in enumerate(clusters)}

    for i, metric in enumerate(oe_metrics):
        plt.subplot(1, 3, i + 1)
        for cluster in clusters:
            cluster_data = video_df[video_df[cluster_col] == cluster]
            sns.boxplot(
                x=[cluster] * len(cluster_data), y=cluster_data[metric],
                color=cluster_colors[cluster],
                showcaps=True,
                boxprops=dict(edgecolor='gray'),
                whiskerprops=dict(color='gray'),
                flierprops=dict(marker='o', color='gray', alpha=0.5),
                medianprops=dict(color='gray'),
                width=0.6
            )
        for cluster in clusters:
            cluster_data = video_df[video_df[cluster_col] == cluster]
            for idx, row in cluster_data.iterrows():
                y = row[metric]
                x = cluster + np.random.uniform(-0.35, 0.35)
                plt.plot(x, y, 'o', color='black', markersize=4, alpha=0.85)
                plt.text(x, y, str(row[c.VIDEO_ID_COL]), fontsize=6, ha='center', va='bottom', alpha=0.85)

        plt.title(f'{metric} by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel(metric)
        plt.xticks(ticks=clusters, labels=[f"Cluster {clust}" for clust in clusters])

    if save_path:
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_cluster_separation(
        df: pd.DataFrame,
        cluster_col: str,
        valence: str,
        arousal: str,
        id_col: str = c.VIDEO_ID_COL,
        save_path: Optional[Path] = None
) -> None:
    """
    Plot clusters in a 2D space defined by x_col and y_col, with convex hulls and point labels.
    :param save_path: Optional file path to save the figure.
    :param df: DataFrame containing the data to plot.
    :param cluster_col: Column name for the cluster labels.
    :param valence: valence column name (x axis).
    :param arousal: arousal column name (y axis).
    :param id_col: column name for video IDs to label points.
    """

    clusters = sorted(df[cluster_col].dropna().unique())
    colors = sns.color_palette("Set2", n_colors=len(clusters))
    cmap = dict(zip(clusters, colors))

    plt.figure(figsize=(5, 5))

    for cl in clusters:
        sub = df[df[cluster_col] == cl]
        plt.scatter(sub[valence], sub[arousal],
                    s=70, c=[cmap[cl]], edgecolor='k',
                    alpha=0.85, label=f'C{cl}')
        # point labels
        if id_col in sub.columns:
            for _, r in sub.iterrows():
                plt.text(r[valence], r[arousal], str(r[id_col]),
                         fontsize=7, ha='center', va='bottom')
        # convex hull (only if ≥3 points)
        if len(sub) >= 3:
            pts = sub[[valence, arousal]].to_numpy()
            try:
                hull = ConvexHull(pts)
                hp = pts[hull.vertices]
                plt.fill(hp[:, 0], hp[:, 1],
                         color=cmap[cl], alpha=0.12, edgecolor='none')
            except Exception:
                pass

    plt.axhline(0, color='gray', lw=0.8)
    plt.axvline(0, color='gray', lw=0.8)
    plt.xlabel(valence.capitalize())
    plt.ylabel(arousal.capitalize())
    plt.title('Valence–Arousal Clusters')
    plt.legend(title='Cluster', fontsize=8)

    if save_path:
        plt.tight_layout(rect=(0, 0, 1, 0.98))
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    feature_cols: list,
    save_path: Path = None
) -> None:
    """
    Calculates and plots a correlation matrix heatmap for the specified feature columns.

    :param df: The DataFrame containing the data.
    :param feature_cols: A list of column names for which to compute the correlation.
    :param save_path: The file path to save the resulting plot. If None, the plot is displayed.
    """
    corr_matrix = df[feature_cols].corr()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        corr_matrix,
        mask=np.triu(np.ones_like(corr_matrix, dtype=bool)),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu_r",
        linewidths=.5,
        ax=ax
    )

    ax.set_ylim(len(corr_matrix), 1)
    ax.set_xlim(0, len(corr_matrix) - 1)

    ax.set_title('Correlation Matrix of Environmental Features', fontsize=16, pad=20)
    ax.grid(False)
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        print(f"Saving correlation heatmap to {save_path}")
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()

    plt.close(fig)


def plot_candidate_predictions(
        df: pd.DataFrame,
        best_rmse: float,
        bounds: Tuple[float, float] = (0, 0),
        target_col: str = c.VALENCE,
        id_col: str = c.VIDEO_ID_COL,
        save_path: Optional[Path] = None
):
    """
    Plot predicted values for candidate videos, categorizing them based on their deviation from expected bounds.
    :param save_path: Optional file path to save the figure.
    :param df: DataFrame containing the predictions and video IDs.
    :param best_rmse: best root mean square error (RMSE) for the predictions.
    :param bounds: Tuple of lower and upper bounds for the expected values.
    :param target_col: name of the target column for predictions.
    :param id_col: name of the column containing video IDs.
    """

    cmap = plt.get_cmap('viridis')
    figsize = (8, 4)
    video_ids = df[id_col].astype(str)
    predictions = df['valence_prediction']
    error_margin = best_rmse
    lower, upper = bounds[0] - error_margin, bounds[1] + error_margin

    # Categorize
    categories = np.select(
        [predictions > upper, predictions < lower],
        ['above', 'below'],
        default='ambiguous'
    )
    marker_map = {'above': '^', 'below': 's', 'ambiguous': 'o'}
    color_map = {'above': cmap(0.9), 'below': cmap(0.2), 'ambiguous': cmap(0.5)}

    fig, ax = plt.subplots(figsize=figsize)
    for i, (pred, cat) in enumerate(zip(predictions, categories)):
        ax.scatter(i, pred, marker=marker_map[cat], color=color_map[cat])
    ax.axhline(upper, color='darkgrey', linestyle='--', label='Expected Error')
    ax.axhline(lower, color='darkgrey', linestyle='--')
    ax.set_ylim(-0.6, 0.6)
    ax.set_xticks(range(len(video_ids)))
    ax.set_xticklabels(video_ids, rotation=90, ha='center', fontsize=8)
    ax.set_xlabel("Video ID")
    ax.set_ylabel(f"Predicted {target_col}")
    ax.set_title(f"Predicted {target_col} per Candidate Video")
    ax.grid(True, linestyle='--', alpha=0.5)
    legend_elements = [
        Line2D([0], [0], marker='^', color=color_map['above'], linestyle='', label='Above range'),
        Line2D([0], [0], marker='s', color=color_map['below'], linestyle='', label='Below range'),
        Line2D([0], [0], marker='o', color=color_map['ambiguous'], linestyle='', label='Ambiguous'),
        Line2D([0], [0], color='darkgrey', linestyle='--', label='Expected Error')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    if save_path:
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# --- Lab study plotting functions ---

def plot_violin_panels(
        df: pd.DataFrame,
        sequence_order: Optional[list] = None, # MODIFIED: New parameter
        palette_name="Set1",
        mean_marker_size=40,
        save_path: Optional[Path] = None
) -> None:
    """
    Plot violin plots for valence, arousal, and ranking by sequence type.
    :param df: DataFrame containing the data.
    :param sequence_order: List specifying the order of sequences to display on the x-axis. If None, sequences are sorted alphabetically.
    :param palette_name: Name of the seaborn color palette to use.
    :param mean_marker_size: Size of the mean marker on the plots.
    :param save_path: Optional file path to save the figure.
    :return: None
    """
    df_plot = df.copy()
    grouping_col = 'sequence_type'

    df_plot[grouping_col] = df_plot['sequence_list'].apply(lambda seq: ' → '.join(map(str, seq)))

    metrics = [('valence', (-1, 1), 'Valence'), ('arousal', (-1, 1), 'Arousal'), ('ranking', (1, 4), 'Ranking')]
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6), sharex=True)

    if len(metrics) == 1:
        axes = [axes]

    for ax, (ycol, ylim, title) in zip(axes, metrics):
        d = df_plot[[grouping_col, ycol]].dropna()

        if sequence_order:
            order = [s for s in sequence_order if s in d[grouping_col].unique()]
        else:
            order = sorted(d[grouping_col].unique())

        # Create palette and color map based on the final order
        palette = sns.color_palette(palette_name, len(order))
        color_map = {seq: color for seq, color in zip(order, palette)}

        sns.violinplot(data=d, x=grouping_col, y=ycol, order=order,
                       hue=grouping_col,
                       palette=color_map,
                       cut=0, inner='quartile', legend=False, ax=ax)

        means = d.groupby(grouping_col)[ycol].mean()

        xs = range(len(order))
        colors = [color_map[seq] for seq in order]
        ax.scatter(xs, means.loc[order], marker='D', s=mean_marker_size,
                   facecolors='white', edgecolors=colors, linewidths=2,
                   zorder=5, clip_on=False)
        ax.set_ylim(*ylim)
        ax.set_title(f'{title}')
        ax.set_xlabel('Sequence')
        ax.set_ylabel(title)
        ax.grid(True, axis='y', alpha=.3)

    fig.suptitle('Overall Ratings by Sequence Type', fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_ranking_by_nb_pos(
        df,
        position_col: str,
        params: Dict[str, Dict[int, str]],
        ranking_col='ranking',
        save_path: Optional[Path] = None
) -> None:
    """
    Boxplot of ranking by position in sequence (NB position).
    :param df: DataFrame containing the data.
    :param position_col: Column name for position values.
    :param params: Mapping of position values to labels.
    :param ranking_col: Column name for ranking values.
    :param save_path: Optional file path to save the figure.
    :return:
    """
    df = df.copy()
    df['pos_label'] = df[position_col].map(params['outlier_position'])

    plt.figure(figsize=(5, 5))
    sns.boxplot(data=df, x='pos_label', y=ranking_col, hue='pos_label',
                palette='Set1', whis=1.5, legend=False)
    plt.ylim(0, 5)
    plt.xlabel('position in sequence')
    plt.ylabel('Ranking')
    plt.title(f'{ranking_col} by Position')
    plt.grid(axis='y', alpha=.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# --- Miscellaneous plotting functions ---


def plot_video_rating_heatmap(
        df: pd.DataFrame,
        video_id: str,
        col_prefix: str = '_AG',
        plot: bool = False
) -> Tuple[float, float, float]:
    """
    Plot a heatmap of video ratings and calculate the normalized distance from the center.
    :param df:
    :param video_id:
    :param col_prefix:
    :param plot:
    :return:
    """
    ratings = df[f'{video_id}{col_prefix}'].values.astype(int) - 1
    heatmap = np.zeros((10, 10), int)
    rows, cols = divmod(ratings, 10)
    np.add.at(heatmap, (rows, cols), 1)

    total = heatmap.sum()
    if total == 0:
        return np.nan, np.nan, np.nan

    X, Y = np.meshgrid(np.arange(10), np.arange(10))
    center = 5
    cx = (heatmap * X).sum() / total
    cy = (heatmap * Y).sum() / total
    dx = cx - center
    dy = center - cy
    norm_x = dx / center
    norm_y = dy / center
    dist = np.hypot(norm_x, norm_y)

    if plot:
        plt.figure(figsize=(5, 5))
        ax = sns.heatmap(heatmap, annot=True, fmt="d", cmap="YlGnBu", cbar=False, xticklabels=False, yticklabels=False)
        ax.scatter(center, center, color='black', s=100, marker='x', label='Center (0,0)')
        ax.scatter(cx, cy, color='red', s=100, marker='o', label='Centroid')
        ax.plot([center, cx], [center, cy], color='black', linestyle='--', label='Distance')
        plt.title(f'Rating Heatmap for {video_id}')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f"Video rating heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()

    return norm_x, norm_y, dist


def count_factors(column, video_df):
    return Counter(item for sublist in video_df[column] if isinstance(sublist, list) for item in sublist)


def plot_factors(
        df: pd.DataFrame,
        video_ids: List[str],
        pf_col: str = c.PF,
        nf_col: str = c.NF,
        xlim: float = None
) -> None:
    """
    Plot positive and negative factors for given video IDs from a long DataFrame.
    :param pf_col: Column name for positive factors.
    :param nf_col: Column name for negative factors.
    :param df: DataFrame containing video data with positive and negative factors.
    :param video_ids: List of video IDs to plot factors for.
    :param xlim: Optional x-axis limit for the plot.
    """

    vids = list(video_ids) if isinstance(video_ids, (list, tuple, set)) else [video_ids]
    vids = [int(v) for v in vids]
    vid_series = pd.to_numeric(df[c.VIDEO_ID_COL], errors='coerce').astype('Int64')

    for vid in vids:
        sub = df[vid_series == vid].copy()

        def to_list(x):
            if isinstance(x, list): return x
            if pd.isna(x): return []
            return [i.strip() for i in str(x).split(';') if i.strip()]

        sub[pf_col] = sub[pf_col].apply(to_list)
        sub[nf_col] = sub[nf_col].apply(to_list)

        value_counts_pf = count_factors(pf_col, sub)
        value_counts_nf = count_factors(nf_col, sub)

        all_factors = sorted(set(value_counts_pf) | set(value_counts_nf))
        factor_counts = pd.DataFrame({
            'Positive': [value_counts_pf.get(f, 0) for f in all_factors],
            'Negative': [value_counts_nf.get(f, 0) for f in all_factors]
        }, index=all_factors)

        div = factor_counts.copy()
        div['Negative'] = -div['Negative']

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(div.index, div['Positive'], color="#3376b5", label='Positive')
        ax.barh(div.index, div['Negative'], color="#e2f3af", label='Negative')

        if xlim is not None:
            ax.set_xlim(-xlim, xlim)

        ax.axvline(0, lw=1)
        ax.set_xlabel("Count", fontsize=12)
        ax.set_title(f"Positive vs Negative Factors (Video {vid})", fontsize=14)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"Factors.png", dpi=150, bbox_inches='tight')
        plt.close()


def plot_factor_composition_diverging(
        df: pd.DataFrame,
        oe_order: List[str],
        oe_col: str = c.OE,
        pf_col: str = c.PF,
        nf_col: str = c.NF,
        sep: str = ";",
        percent: bool = False
) -> None:
    """
    Plot a diverging bar chart showing the composition of positive and negative factors by OE rating.
    :param df: DataFrame containing the positive and negative factor choice data.
    :param oe_order: order of overall experience rating categories.
    :param oe_col: overall experience rating column name.
    :param pf_col: positive factor column name.
    :param nf_col: negative factor column name.
    :param sep: separator for splitting factor strings into lists.
    :param percent: if True, show percentages instead of counts.
    """

    d = df.copy()
    d[oe_col] = pd.Categorical(d[oe_col], categories=oe_order, ordered=True)
    pf = d[pf_col].fillna("").str.get_dummies(sep=sep)
    nf = d[nf_col].fillna("").str.get_dummies(sep=sep)

    gpos = pf.groupby(d[oe_col], observed=False).sum().T.reindex(columns=oe_order, fill_value=0)
    gneg = nf.groupby(d[oe_col], observed=False).sum().T.reindex(columns=oe_order, fill_value=0)

    factors = gpos.index.union(gneg.index)
    gpos, gneg = gpos.reindex(factors, fill_value=0), gneg.reindex(factors, fill_value=0)
    order = (gpos.sum(1) + gneg.sum(1)).sort_values().index
    gpos, gneg = gpos.loc[order], gneg.loc[order]

    neg_cats = ['Very negative', 'Somewhat negative', 'Negative']
    neu_cat = ['Neutral']
    pos_cats = ['Somewhat positive', 'Positive', 'Very positive']
    pos_stack = neu_cat + pos_cats
    neg_stack = neu_cat + neg_cats
    colors = {
        **dict(zip(pos_stack, ['lightgrey'] + sns.color_palette("Greens", n_colors=len(pos_cats)))),
        **dict(zip(neg_stack, ['lightgrey'] + sns.color_palette("Reds", n_colors=len(neg_cats))))
    }

    if percent:
        denom = (gpos.sum(1) + gneg.sum(1)).replace(0, 1)
        gpos = gpos.div(denom, axis=0) * 100
        gneg = gneg.div(denom, axis=0) * 100
        xlab = "Percentage"
    else:
        xlab = "Number of Mentions"

    fig, ax = plt.subplots(figsize=(12, max(4, len(order) * 0.25)))
    gpos[pos_stack].plot(kind="barh", stacked=True, ax=ax,
                         color=[colors[c] for c in pos_stack], width=0.85, legend=False)
    (-gneg[neg_stack]).plot(kind="barh", stacked=True, ax=ax,
                            color=[colors[c] for c in neg_stack], width=0.85, legend=False)
    ax.axvline(0, color="black", lw=1)
    mx = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
    ax.set_xlim(-mx, mx)
    ax.text(0.5, 1.02, "mentioned as negative factors <> mentioned as positive factors",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=11)
    ax.set_xlabel(xlab);
    ax.set_ylabel("Factors")

    legend_items = neu_cat + pos_cats + neg_cats
    ax.legend([Patch(facecolor=colors[c]) for c in legend_items],
              legend_items, title="Overall Experience Rating", loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False)

    plt.tight_layout()
    plt.savefig(f"Factor overview.png", dpi=150, bbox_inches='tight')
    plt.close()


def summarize(
        df: pd.DataFrame,
        group_col: str,
        dv: str,
        order: list
) -> pd.DataFrame:
    """
    Summary stats with 95% CI for plotting.
    :param df: input DataFrame
    :param group_col: grouping variable column name
    :param dv: dependent variable column name
    :param order: list defining the order of groups
    :return: DataFrame with mean, sd, n, and 95% CI for each group
    """
    s = (df.groupby(group_col)[dv]
         .agg(mean='mean', sd='std', n='count')
         .reindex(order))
    s['ci95'] = 1.96 * s['sd'] / np.sqrt(s['n'])
    return s


def plot_count_vs_pos(
        df: pd.DataFrame,
        dv: str,
        count_col: str,
        count_order: list,
        count_tick_fmt="{v} B",
        pos_col=None,
        pos_order=None,
        pos_labels=None,
        color_count='black',
        color_pos='tab:blue',
        figsize=(4, 4),
        marker_count='o',
        marker_pos='s'
) -> tuple:
    """
    Line plot of mean DV ±95% CI by a count variable on bottom x-axis,
    with an optional second line & top axis for a 'position' variable.
    """
    # summaries
    s_cnt = summarize(df, count_col, dv, count_order)

    fig, ax = plt.subplots(figsize=figsize)
    # count line
    ax.plot(s_cnt.index, s_cnt['mean'], f'-{marker_count}', color=color_count, label=count_col)
    ax.errorbar(s_cnt.index, s_cnt['mean'], yerr=s_cnt['ci95'],
                fmt='none', ecolor=color_count, capsize=3, lw=1)
    ax.set_xlabel(count_col)
    ax.set_ylabel(f'{dv} (mean ± 95% CI)')
    ax.set_xticks(count_order)
    ax.set_xticklabels([count_tick_fmt.format(v=v) for v in count_order])

    # optional position line + top axis
    if pos_col is not None and pos_order is not None:
        s_pos = summarize(df, pos_col, dv, pos_order)

        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(pos_order)
        if pos_labels is None:
            ax2.set_xticklabels(pos_order)
        else:
            ax2.set_xticklabels([pos_labels[p] for p in pos_order])
        ax2.set_xlabel(pos_col)

        ax.plot(s_pos.index, s_pos['mean'], f'-{marker_pos}',
                color=color_pos, label=pos_col)
        ax.errorbar(s_pos.index, s_pos['mean'], yerr=s_pos['ci95'],
                    fmt='none', ecolor=color_pos, capsize=3, lw=1)

    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_individual_slopes(
        data: pd.DataFrame,
        grouping_var: str,
        output_path: str,
        x_col: str = 'spoiler_position_num',
        y_col: str = 'valence',
        scenario_col: str = 'scenario',
        participant_col: str = 'participant_id',
        palette_name: str = "Set2"
):
    """
    Plot individual regression slopes for each participant, colored by a grouping variable,
    :param data:
    :param grouping_var:
    :param output_path:
    :param x_col:
    :param y_col:
    :param scenario_col:
    :param participant_col:
    :param palette_name:
    :return:
    """

    unique_groups = data[grouping_var].unique()
    palette = dict(zip(unique_groups, sns.color_palette(palette_name, len(unique_groups))))

    scenarios = data[scenario_col].unique()
    fig, axes = plt.subplots(1, len(scenarios), figsize=(8 * len(scenarios), 7), sharey=True)
    fig.suptitle(f'Individual Participant Slopes by {grouping_var} and Scenario', fontsize=18, y=1.02)

    # Ensure axes is always a list for consistent indexing
    if len(scenarios) == 1:
        axes = [axes]

    # --- 3. Loop through each participant and plot their regression line ---
    for pid in data[participant_col].unique():
        participant_data = data[data[participant_col] == pid]
        if participant_data.empty:
            continue

        group_val = participant_data[grouping_var].iloc[0]
        color = palette[group_val]

        for i, scenario in enumerate(scenarios):
            scenario_data = participant_data[participant_data[scenario_col] == scenario]

            if not scenario_data.empty and len(scenario_data[x_col].unique()) > 1:
                sns.regplot(data=scenario_data, x=x_col, y=y_col,
                            ax=axes[i], color=color, scatter=False, ci=None, line_kws={'alpha': 0.6})

    # --- 4. Customize and add a single legend ---
    for i, scenario in enumerate(scenarios):
        axes[i].set_title(f'Scenario: {scenario}')
        axes[i].set_xlabel('Spoiler Position')
        axes[i].grid(True)
    axes[0].set_ylabel('Valence Rating')

    # Create custom legend handles
    legend_handles = [Line2D([0], [0], color=palette[g], lw=2, label=g) for g in unique_groups]
    axes[-1].legend(handles=legend_handles, title=grouping_var)

    # --- 5. Save the figure ---
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def plot_segmentation_overlay(
        image: Image,
        results: list,
        output_path: Path = None
) -> None:
    """
    Plot the segmentation overlay and save it to a file or show it.
    :param image: PIL Image object.
    :param results: List of segmentation results from the model.
    :param output_path: Optional. Path to save the output image. If None, shows the plot.
    :return: None
    """
    unique_labels = sorted({res['label'] for res in results})
    if not unique_labels:
        # If there's nothing to plot, just save the original image if a path is given
        if output_path:
            image.save(output_path)
        return

    num_labels = len(unique_labels)
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    integer_map = np.zeros(image.size[::-1], dtype=np.uint8)

    for res in results:
        mask = np.array(res['mask']) > 0
        if res['label'] in label_to_id:
            label_id = label_to_id[res['label']]
            integer_map[mask] = label_id

    cmap = plt.get_cmap('tab20', num_labels)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(image)
    ax.imshow(integer_map, cmap=cmap, alpha=0.7, vmin=0, vmax=num_labels - 1)
    ax.set_title("Segmented Overlay")
    ax.axis('off')

    legend_patches = [mpatches.Patch(color=cmap(i), label=label) for i, label in enumerate(unique_labels)]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    else:
        plt.show()


def plot_bland_altman(
    df: pd.DataFrame,
    measurement1: str,
    measurement2: str,
    label_col: str,
    save_path: Optional[Path] = None
) -> None:
    """
    Generates and saves or shows a Bland-Altman plot to compare two measurements.

    :param df: DataFrame containing the data.
    :param measurement1: Column name of the first measurement (e.g., 'valence').
    :param measurement2: Column name of the second measurement (e.g., 'valence_online').
    :param label_col: Column name for the point labels (e.g., 'video_id').
    :param save_path: Optional file path to save the plot. If None, the plot is shown.
    """
    # 1. Prepare the data by dropping NaNs and calculating average/difference
    df_agree = df.dropna(subset=[measurement1, measurement2]).copy()
    avg_col = 'average_score'
    diff_col = 'difference_score'
    df_agree[avg_col] = (df_agree[measurement1] + df_agree[measurement2]) / 2
    df_agree[diff_col] = df_agree[measurement1] - df_agree[measurement2]

    # 2. Calculate statistical limits
    mean_diff = df_agree[diff_col].mean()
    std_diff = df_agree[diff_col].std()
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    # 3. Create the plot
    plt.figure(figsize=(5, 5))
    plt.scatter(df_agree[avg_col], df_agree[diff_col], alpha=0.7)

    for index, row in df_agree.iterrows():
        label = str(row[label_col])
        plt.text(row[avg_col], row[diff_col], label, fontsize=8, ha='center', va='bottom')

    # Plot the reference lines
    plt.axhline(mean_diff, color='red', linestyle='--', label='Mean Difference (Bias)')
    plt.axhline(upper_loa, color='gray', linestyle='--', label='Upper Limit of Agreement')
    plt.axhline(lower_loa, color='gray', linestyle='--', label='Lower Limit of Agreement')
    title1 = measurement1.replace('_', ' ').title()
    title2 = measurement2.replace('_', ' ').title()
    plt.title(f'Bland-Altman Plot: {title1} vs. {title2}')
    plt.xlabel(f'Average of Scores')
    plt.ylabel(f'Difference in Scores ({measurement1} - {measurement2})')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Bland-Altman plot saved to {save_path}")
    else:
        plt.show()

