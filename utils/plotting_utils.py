from scipy.spatial import ConvexHull
from matplotlib.patches import Patch
import matplotlib.patches as mpatches
from collections import Counter
from PIL import Image
from typing import Any, Optional, Sequence, Tuple, Dict, List, Union
import seaborn as sns
import matplotlib as mpl
import re
from matplotlib.lines import Line2D
import constants as c
import scipy.stats as stats
from scipy.stats import ttest_ind
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.colors as mcolors
import statsmodels.api as sm
from statsmodels.formula.api import ols


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


def plot_affect_grid_subplots(
    df: pd.DataFrame,
    video_id: Optional[Any] = None,
    save_path: Optional[Path] = None,
    ncols: int = 5,
    nrows: int = 6,
    normalize: str = "count",
    robust_vmax_percentile: Optional[float] = None
) -> None:
    """
    Plot heatmaps of Affect Grid usage for multiple videos in a grid layout.
    """

    if video_id is None:
        vids = sorted(df[c.VIDEO_ID_COL].dropna().unique())
    elif isinstance(video_id, (list, tuple, set, pd.Series, np.ndarray)):
        vids = sorted(set(int(v) for v in video_id))
    else:
        vids = [int(video_id)]
    vids = vids[: ncols * nrows]

    grids = []
    for vid in vids:
        ag = pd.to_numeric(df.loc[df[c.VIDEO_ID_COL] == vid, c.AG], errors="coerce").dropna()
        ag = ag[(ag >= 1) & (ag <= 100)].astype(int).to_numpy()

        counts = np.bincount(ag - 1, minlength=100).reshape(10, 10).astype(float)
        if normalize == "percent" and counts.sum() > 0:
            counts = counts / counts.sum() * 100.0
        grids.append((vid, counts))

    all_vals = np.concatenate([g.ravel() for _, g in grids]) if grids else np.array([0.0])
    vmin, vmax = 0.0, float(np.nanmax(all_vals))
    if robust_vmax_percentile is not None:
        vmax = float(np.percentile(all_vals, robust_vmax_percentile))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.1, nrows * 3.1))
    axes = np.ravel(axes)
    cmap = plt.get_cmap("YlGnBu")

    for ax, (vid, grid) in zip(axes, grids):
        sns.heatmap(grid, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    cbar=False, square=True, xticklabels=False, yticklabels=False)
        ax.set_title(f"{vid}", fontsize=16, pad=1)

    for ax in axes[len(grids):]:
        ax.axis("off")

    sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.15, 0.06, 0.7, 0.02])
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Response Count" if normalize == "count" else "% of responses", fontsize=18)
    cbar.ax.tick_params(labelsize=14)
    fig.subplots_adjust(top=0.93, bottom=0.10, hspace=0.15, wspace=0.08)
    fig.suptitle("Affect Grid Heatmaps by Cycling Scenario ID", fontsize=18, y=0.975, fontweight="bold")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_bgmm_clusters_subplots(
    points_df: pd.DataFrame,
    video_id: Optional[Any] = None,
    save_path: Optional[Path] = None,
    ncols: int = 5,
    nrows: int = 6,
    # columns
    video_col: str = "video_id",
    x_col: str = "valence",
    y_col: str = "arousal",
    cluster_col: str = "cluster",
    # styling / behavior
    show_means: bool = True,
    means_df: Optional[pd.DataFrame] = None,   # optional: precomputed means for each video+cluster
    means_cols: tuple[str, str] = ("mean_valence", "mean_arousal"),
    alpha: float = 0.55,
    s: float = 12,
    mean_marker_size: float = 80,
    xlim: tuple[float, float] = (-1.0, 1.0),
    ylim: tuple[float, float] = (-1.0, 1.0),
    legend: bool = False,
) -> None:
    # ---- pick videos ----
    if video_id is None:
        vids = sorted(points_df[video_col].dropna().unique())
    elif isinstance(video_id, (list, tuple, set, pd.Series, np.ndarray)):
        vids = sorted(set(int(v) for v in video_id))
    else:
        vids = [int(video_id)]
    vids = vids[: ncols * nrows]

    # ---- set up figure ----
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.1, nrows * 3.1))
    axes = np.ravel(axes)

    # ---- global color mapping (stable across panels) ----
    # Collect all cluster labels across selected vids (excluding NaN)
    sub = points_df[points_df[video_col].isin(vids)]
    cl = pd.to_numeric(sub[cluster_col], errors="coerce").dropna().astype(int)
    # keep -1 if present; otherwise just 0..K-1
    uniq_clusters = sorted(cl.unique().tolist())

    # Use a qualitative colormap; tab20 is fine up to 20 clusters
    cmap = plt.get_cmap("tab20")
    def color_for(k: int):
        if k == -1:
            return (0.6, 0.6, 0.6, 1.0)  # grey for unassigned
        return cmap(k % cmap.N)

    # ---- plot each video ----
    for ax, vid in zip(axes, vids):
        vdf = points_df.loc[points_df[video_col] == vid, [x_col, y_col, cluster_col]].copy()
        vdf[x_col] = pd.to_numeric(vdf[x_col], errors="coerce")
        vdf[y_col] = pd.to_numeric(vdf[y_col], errors="coerce")
        vdf[cluster_col] = pd.to_numeric(vdf[cluster_col], errors="coerce")
        vdf = vdf.dropna()

        ax.set_title(f"{vid}", fontsize=16, pad=1)

        # draw by cluster to keep colors consistent
        for k in uniq_clusters:
            kdf = vdf[vdf[cluster_col].astype(int) == k]
            if kdf.empty:
                continue
            ax.scatter(
                kdf[x_col].to_numpy(),
                kdf[y_col].to_numpy(),
                s=s,
                alpha=alpha,
                c=[color_for(int(k))],
                linewidths=0,
            )

        # overlay means (either compute from points or use provided means_df)
        if show_means:
            if means_df is None:
                # compute point-means per cluster for this video
                if not vdf.empty:
                    g = vdf.groupby(vdf[cluster_col].astype(int))[[x_col, y_col]].mean()
                    for k, row in g.iterrows():
                        ax.scatter(
                            [row[x_col]], [row[y_col]],
                            s=mean_marker_size,
                            c=[color_for(int(k))],
                            marker="X",
                            edgecolors="black",
                            linewidths=0.7,
                            zorder=5
                        )
            else:
                mv = means_df[means_df[video_col] == vid]
                for _, r in mv.iterrows():
                    k = int(r[cluster_col])
                    ax.scatter(
                        [r[means_cols[0]]], [r[means_cols[1]]],
                        s=mean_marker_size,
                        c=[color_for(k)],
                        marker="X",
                        edgecolors="black",
                        linewidths=0.7,
                        zorder=5
                    )

        # axes formatting
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.axhline(0, linewidth=0.8, alpha=0.3)
        ax.axvline(0, linewidth=0.8, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")

    # turn off unused axes
    for ax in axes[len(vids):]:
        ax.axis("off")

    # optional legend (global, minimal)
    if legend and uniq_clusters:
        handles = []
        labels = []
        for k in uniq_clusters:
            handles.append(plt.Line2D([0], [0], marker='o', linestyle='',
                                     markerfacecolor=color_for(int(k)), markersize=8,
                                     markeredgewidth=0))
            labels.append(f"cluster {k}")
        fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 8),
                   frameon=False, bbox_to_anchor=(0.5, 0.03))

    fig.subplots_adjust(top=0.93, bottom=0.08, hspace=0.15, wspace=0.08)
    fig.suptitle("BGMM Valence–Arousal Clusters by Video", fontsize=18, y=0.975)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_affect_grid_usage_with_marginals(
    df: pd.DataFrame,
    ag_col: str = "AG",
    save_path: Optional[Path] = None,
    normalize: str = "percent",   # "count" or "percent"
    cmap: str = "YlGnBu"
) -> None:

    # 1. build 10×10 grid
    ag = pd.to_numeric(df[ag_col], errors="coerce").dropna()
    ag = ag[(ag >= 1) & (ag <= 100)].astype(int).to_numpy()
    grid = np.bincount(ag - 1, minlength=100).reshape(10, 10).astype(float)

    if normalize == "percent" and grid.sum() > 0:
        grid /= grid.sum()
        grid *= 100.0
        cbar_label = "% of responses"
    else:
        cbar_label = "Response Count"

    valence_dist = grid.sum(axis=0)
    arousal_dist = grid.sum(axis=1)

    # 2. derive per-bar colors from the colormap
    cm = plt.get_cmap(cmap)

    # 3. create figure with four axes (no shared axes)
    fig = plt.figure(figsize=(8, 8))

    ax_main = fig.add_axes([0.12, 0.12, 0.65, 0.65])
    ax_top = fig.add_axes([0.12, 0.78, 0.65, 0.12])
    ax_right = fig.add_axes([0.78, 0.12, 0.12, 0.65])
    ax_cbar = fig.add_axes([0.12, 0.06, 0.65, 0.03])

    #  4. main heatmap
    sns.heatmap(
        grid,
        ax=ax_main,
        cmap=cmap,
        cbar=False,
        xticklabels=False,
        yticklabels=False,
        annot=(normalize != "percent"),
        fmt=".0f",
        annot_kws={"fontsize": 12}
    )
    ax_main.set_xlim(0, 10)
    ax_main.set_ylim(10, 0)
    ax_main.set_aspect("equal", adjustable="box")
    ax_main.set_anchor("NW")
    ax_main.axhline(5, color="darkgrey", lw=1)
    ax_main.axvline(5, color="darkgrey", lw=1)

    #  5. draw once so the equal-aspect box is finalised
    fig.canvas.draw()

    # 6. read actual pixel bbox of the heatmap
    main_bbox = ax_main.get_window_extent(fig.canvas.get_renderer())
    fig_w, fig_h = fig.get_size_inches() * fig.dpi

    x0 = main_bbox.x0 / fig_w
    y0 = main_bbox.y0 / fig_h
    w = main_bbox.width / fig_w
    h = main_bbox.height / fig_h

    # 7. top histogram
    gap_top = 0.008
    hist_h = 0.07
    ax_top.set_position([x0, y0 + h + gap_top, w, hist_h])
    ax_top.set_xlim(0, 10)
    ax_top.set_ylim(0, valence_dist.max() * 1.15)
    ax_top.bar(
        np.arange(10), valence_dist,
        width=1.0, align="edge",
        color='lightgrey', edgecolor="white", linewidth=1.2, alpha=0.7
    )
    ax_top.axis("off")
    ax_top.text(0.5, 1.15, "Valence [-1, 1]", transform=ax_top.transAxes,
                ha="center", va="bottom", fontsize=18,  color="#333")

    # 8. right histogram
    gap_right = 0.008
    hist_w = 0.06
    ax_right.set_position([x0 + w + gap_right, y0, hist_w, h])
    ax_right.set_ylim(10, 0)
    ax_right.set_xlim(0, arousal_dist.max() * 1.15)
    ax_right.barh(
        np.arange(10), arousal_dist,
        height=1.0, align="edge",
        color='lightgrey', edgecolor="white", linewidth=1.2, alpha=0.7
    )
    ax_right.axis("off")
    ax_right.text(1.15, 0.5, "Arousal [1, -1]", transform=ax_right.transAxes,
                  ha="left", va="center", rotation=270, fontsize=16, color="#333")

    # 9. colorbar
    gap_cbar = 0.025
    cbar_h = 0.025
    ax_cbar.set_position([x0, y0 - gap_cbar - cbar_h, w, cbar_h])
    norm = mpl.colors.Normalize(vmin=float(grid.min()), vmax=float(grid.max()))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation="horizontal")
    cbar.set_label(cbar_label, fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    fig.suptitle("Overall Affect Grid Usage", fontsize=16, x=0.45, y=0.94, fontweight="bold")

    # 12. save / show
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_valence_arousal_by_oe(
    df: pd.DataFrame,
    oe_col: str,
    oe_order: Sequence[str],
    valence_col: str,
    arousal_col: str,
    save_path: Optional[Path] = None,
    kind: str = "violin",  # "violin" or "box"
    cmap: str = "YlGnBu",
) -> None:

    d = df[[oe_col, valence_col, arousal_col]].dropna().copy()
    d[oe_col] = pd.Categorical(d[oe_col], categories=list(oe_order), ordered=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    if kind not in {"violin", "box"}:
        raise ValueError("kind must be 'violin' or 'box'")

    common_kwargs = dict(
        data=d,
        x=oe_col,
        order=list(oe_order),
        hue=oe_col,
        palette=cmap,
        legend=False
    )

    # ---- Valence ----
    if kind == "violin":
        sns.violinplot(y=valence_col, ax=axes[0], inner="quartile", **common_kwargs)
    else:
        sns.boxplot(y=valence_col, ax=axes[0], **common_kwargs)

    axes[0].axhline(0, color="grey", lw=1, alpha=0.4)
    axes[0].set_title("Valence distribution by Overall Experience", fontsize=16)
    axes[0].set_ylabel("Valence")

    # ---- Arousal ----
    if kind == "violin":
        sns.violinplot(y=arousal_col, ax=axes[1], inner="quartile", **common_kwargs)
    else:
        sns.boxplot(y=arousal_col, ax=axes[1], **common_kwargs)

    axes[1].axhline(0, color="grey", lw=1, alpha=0.4)
    axes[1].set_title("Arousal distribution by Overall Experience", fontsize=16)
    axes[1].set_ylabel("Arousal")

    for ax in axes:
        ax.tick_params(axis='both', labelsize=12)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_metric_correlations(
        df: pd.DataFrame,
        cols_to_plot: List[str],
        output_dir: Union[str, Path],
        figsize: tuple = (16, 13)
):
    valid_cols = [c for c in cols_to_plot if c in df.columns]
    clip_df = df[valid_cols].copy()

    clean_cols = [c.replace('_', ' ').replace('mean distance', '').title().strip() for c in valid_cols]
    clip_df.columns = clean_cols

    corr = clip_df.corr(numeric_only=True)

    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    ax = sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={"shrink": 1.0, "ticks": [-1, -0.5, 0, 0.5, 1]}
    )

    plt.title("Correlation between Clip-Level Metrics", fontsize=15, pad=20)
    plt.xticks(rotation=90, ha="right")

    y_labels = ax.get_yticklabels()
    if len(y_labels):
        y_labels[0].set_visible(False)

    x_labels = ax.get_xticklabels()
    if len(x_labels):
        x_labels[-1].set_visible(False)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)

    output_path = Path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_diffusion_vs_polarization_map(
    video_metrics: pd.DataFrame,
    save_path: Optional[Path] = None,
    x_col: str = "dispersion_mean_distance",
    y_col: str = "anisotropy_index",
    figsize=(10, 6),
    cmap: str = "YlGnBu",
    single_color_pos: float = 0.65,
    marker_size: int = 140,
    edgecolor: str = "white",
    edge_lw: float = 1.0,
    show_labels: bool = True,
    label_fontsize: int = 9,
):
    """
    Diffusion vs Polarization map (RQ2), simplified:

    - x-axis: disagreement magnitude (mean distance to centroid)
    - y-axis: directional structure (anisotropy index)
    - single color for all points (sampled from cmap)
    - circles only
    - no OE legend, no GMM encoding, no reference lines
    """

    d = video_metrics.copy()
    d = d.dropna(subset=[c.VIDEO_ID_COL, x_col, y_col]).copy()

    # choose one color from the same palette
    cm = plt.get_cmap(cmap)
    point_color = cm(float(np.clip(single_color_pos, 0.0, 1.0)))

    fig, ax = plt.subplots(figsize=figsize)

    for _, r in d.iterrows():
        x = float(r[x_col])
        y = float(r[y_col])
        vid = str(int(r[c.VIDEO_ID_COL]))

        ax.scatter(
            x, y,
            s=marker_size,
            marker="o",
            color=point_color,
            edgecolor=edgecolor,
            linewidth=edge_lw,
            zorder=3
        )
        if show_labels:
            ax.text(x, y, vid, fontsize=label_fontsize, ha="center", va="center", zorder=4)

    ax.set_xlabel("Disagreement magnitude (mean distance to centroid)", fontsize=12)

    if y_col == "anisotropy_index":
        ax.set_ylabel("Directional structure (anisotropy index)", fontsize=12)
        ax.set_ylim(0, 1.02)
    else:
        ax.set_ylabel(y_col.replace("_", " "), fontsize=12)

    ax.set_title("Disagreement magnitude vs. directional structure (RQ2)", fontsize=14)
    ax.grid(True, alpha=0.25, zorder=0)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
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

"""
def plot_video_affect_means_with_vectors(
    video_metrics,
    save_path: Optional[Path] = None,
    oe_col: str = "oe_mean",
    cmap: str = "YlGnBu",
    arrow_scale: float = 0.35,
    marker_size: float = 100,
):
    req = ["video_id", "valence_mean", "arousal_mean",
           "valence_variance", "arousal_variance", "valence_arousal_covariance",
           oe_col]
    d = video_metrics.dropna(subset=req).copy()
    fig, ax = plt.subplots(figsize=(10, 10))

    # ------------------------------------------------------------------
    # DISCRETE COLORS BY OE CATEGORY
    # ------------------------------------------------------------------
    k = len(c.OE_ORDER)
    cm = plt.get_cmap(cmap)
    palette = cm(np.linspace(0.15, 0.9, k))
    oe_to_color = {lab: palette[i] for i, lab in enumerate(c.OE_ORDER)}
    d["_oe_cat"] = d["oe_mode"]

    # ------------------------------------------------------------------
    # PLOT POINTS + VECTORS
    # Vector length = anisotropy of covariance (directionality of disagreement)
    # ------------------------------------------------------------------
    for _, row in d.reset_index(drop=True).iterrows():
        x = float(row["valence_mean"])
        y = float(row["arousal_mean"])
        cat = row["_oe_cat"]
        color = oe_to_color[cat]

        cov = np.array([
            [float(row["valence_variance"]), float(row["valence_arousal_covariance"])],
            [float(row["valence_arousal_covariance"]), float(row["arousal_variance"])]
        ])

        # principal axis
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        v = vecs[:, 0]
        lambda1 = float(max(vals[0], 0.0))
        lambda2 = float(max(vals[1], 0.0))

        # anisotropy in [0,1): 0 = round cloud, 1 = highly elongated/polarized
        anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-9)
        anisotropy = float(np.clip(anisotropy, 0.0, 1.0))

        dx = float(v[0] * anisotropy * arrow_scale)
        dy = float(v[1] * anisotropy * arrow_scale)

        ax.scatter(
            x, y,
            s=marker_size,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3
        )

        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=2.0, alpha=0.9, zorder=2)
        ax.text(x, y, str(int(row["video_id"])), fontsize=10, ha="center", va="center", zorder=4)

    # ------------------------------------------------------------------
    # AFFECT GRID BACKDROP (10×10)
    # ------------------------------------------------------------------
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Valence [-1, 1]", fontsize=14)
    ax.set_ylabel("Arousal [-1, 1]", fontsize=14)

    ax.set_xticks([])
    ax.set_yticks([])

    step = 0.2
    grid_vals = np.arange(-1, 1 + 1e-9, step)
    for gv in grid_vals:
        ax.axhline(gv, color="lightgrey", lw=0.8, zorder=0)
        ax.axvline(gv, color="lightgrey", lw=0.8, zorder=0)

    ax.axhline(0, color="darkgrey", lw=1.2, zorder=1)
    ax.axvline(0, color="darkgrey", lw=1.2, zorder=1)
    ax.set_title("Cycling Scenarios in Valence–Arousal Space", fontsize=14, fontweight="bold", pad=14)

    # ------------------------------------------------------------------
    # DISCRETE LEGEND (OE categories)
    # ------------------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='',
               markerfacecolor=oe_to_color[lab],
               markeredgecolor='white', markeredgewidth=0.9,
               markersize=10, label=lab)
        for lab in c.OE_ORDER
    ]

    line_handle = Line2D([0], [0], color="grey", lw=2.0, label="Main variance direction")

    legend_handles.append(line_handle)
    ax.legend(
        handles=legend_handles,
        title="Overall Experience (OE)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=True,
        fontsize=14,
        title_fontsize=14
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
"""

def plot_video_affect_means_with_vectors(
        video_metrics,
        save_path: Optional[Path] = None,
        oe_col: str = "oe_mean",
        cmap: str = "YlGnBu",
        arrow_scale: float = 0.35,
        marker_size: float = 100,
):
    req = ["video_id", "valence_mean", "arousal_mean",
           "valence_variance", "arousal_variance", "valence_arousal_covariance",
           oe_col]
    d = video_metrics.dropna(subset=req).copy()
    fig, ax = plt.subplots(figsize=(10, 10))

    # ------------------------------------------------------------------
    # DISCRETE COLORS BY OE CATEGORY
    # ------------------------------------------------------------------
    k = len(c.OE_ORDER)
    cm = plt.get_cmap(cmap)
    palette = cm(np.linspace(0.15, 0.9, k))
    oe_to_color = {lab: palette[i] for i, lab in enumerate(c.OE_ORDER)}
    d["_oe_cat"] = d["oe_mode"]

    texts = []
    all_x = []
    all_y = []

    # ------------------------------------------------------------------
    # PLOT POINTS + VECTORS
    # ------------------------------------------------------------------
    for _, row in d.reset_index(drop=True).iterrows():
        x = float(row["valence_mean"])
        y = float(row["arousal_mean"])
        cat = row["_oe_cat"]
        color = oe_to_color[cat]

        cov = np.array([
            [float(row["valence_variance"]), float(row["valence_arousal_covariance"])],
            [float(row["valence_arousal_covariance"]), float(row["arousal_variance"])]
        ])

        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        v = vecs[:, 0]
        lambda1 = float(max(vals[0], 0.0))
        lambda2 = float(max(vals[1], 0.0))

        anisotropy = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-9)
        anisotropy = float(np.clip(anisotropy, 0.0, 1.0))

        dx = float(v[0] * anisotropy * arrow_scale)
        dy = float(v[1] * anisotropy * arrow_scale)

        ax.scatter(
            x, y,
            s=marker_size,
            color=color,
            edgecolor="white",
            linewidth=0.9,
            zorder=3
        )

        ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, lw=2.0, alpha=0.9, zorder=2)
        t = ax.text(x, y, str(int(row["video_id"])), fontsize=10,
                    ha="center", va="center", zorder=4, color="#333333")
        texts.append(t)
        all_x.append(x)
        all_y.append(y)

    # ------------------------------------------------------------------
    # AFFECT GRID BACKDROP (10×10) & LIMITS
    # ------------------------------------------------------------------
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Valence [-1, 1]", fontsize=14)
    ax.set_ylabel("Arousal [-1, 1]", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    step = 0.2
    grid_vals = np.arange(-1, 1 + 1e-9, step)
    for gv in grid_vals:
        ax.axhline(gv, color="lightgrey", lw=0.8, zorder=0)
        ax.axvline(gv, color="lightgrey", lw=0.8, zorder=0)

    ax.axhline(0, color="darkgrey", lw=1.2, zorder=1)
    ax.axvline(0, color="darkgrey", lw=1.2, zorder=1)
    ax.set_title("Cycling Scenarios in Valence–Arousal Space", fontsize=14, fontweight="bold", pad=14)

    # ------------------------------------------------------------------
    # DISCRETE LEGEND
    # ------------------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='',
               markerfacecolor=oe_to_color[lab],
               markeredgecolor='white', markeredgewidth=0.9,
               markersize=10, label=lab)
        for lab in c.OE_ORDER
    ]
    line_handle = Line2D([0], [0], color="grey", lw=2.0, label="Main variance direction")
    legend_handles.append(line_handle)

    ax.legend(
        handles=legend_handles,
        title="Overall Experience (OE)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        frameon=True,
        fontsize=14,
        title_fontsize=14
    )

    # ------------------------------------------------------------------
    # ADJUST TEXT (Aggressive Settings)
    # ------------------------------------------------------------------
    adjust_text(
        texts,
        x=all_x,
        y=all_y,
        arrowprops=dict(arrowstyle='-', color='gray', lw=1, alpha=0.5),
        force_text=10,
        force_points=20,
        expand_points=(1.5, 1.5)
    )

    plt.tight_layout(rect=[0, 0, 0.82, 1])
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _natural_key(x: Any) -> List[Any]:
    """
    Natural sort key: "vid2" < "vid10".
    Falls back gracefully for non-strings.
    """
    s = "" if pd.isna(x) else str(x)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def create_quadrant_distribution_plot(
            long_df: pd.DataFrame,
            video_id_column: str,
            output_path: Optional[Path] = None,
            figsize: Tuple[int, int] = (15, 8),
            show_percent_labels: bool = True,
            label_threshold_pct: float = 5.0
) -> None:

        sns.set_theme(style="whitegrid")
        colors = sns.color_palette("YlGnBu", n_colors=len(c.AFFECTIVE_STATES))

        df = long_df.copy()
        pivot = pd.crosstab(df[video_id_column], df[c.AFFECT_STATE])
        pivot = pivot.reindex(columns=c.AFFECTIVE_STATES, fill_value=0)

        row_sums = pivot.sum(axis=1).replace(0, np.nan)
        pivot_pct = (pivot.div(row_sums, axis=0) * 100).fillna(0)

        # --------------------------------------------------
        # Compute quadrant entropy (categorical ambiguity)
        # --------------------------------------------------
        eps = 1e-12
        entropy = - (pivot_pct / 100.0) * np.log((pivot_pct / 100.0) + eps)
        entropy = entropy.sum(axis=1)

        # --------------------------------------------------
        # Sort videos by entropy (low → high ambiguity)
        # --------------------------------------------------
        video_ids_sorted = entropy.sort_values(ascending=True).index.tolist()

        pivot = pivot.reindex(index=video_ids_sorted, fill_value=0)
        pivot_pct = pivot_pct.reindex(index=video_ids_sorted, fill_value=0)


        fig, ax = plt.subplots(figsize=figsize)
        n = len(video_ids_sorted)
        x = np.arange(n)
        bottom = np.zeros(n)

        for idx, state in enumerate(c.AFFECTIVE_STATES):
            values = pivot_pct[state].to_numpy()

            ax.bar(
                x,
                values,
                bottom=bottom,
                label=state,
                color=colors[idx]
            )
            ax.margins(x=0.01)

            if show_percent_labels:
                for j, (v, b) in enumerate(zip(values, bottom)):
                    if v >= label_threshold_pct:
                        ax.text(
                            float(x[j]),
                            float(b + v / 2),
                            f"{v:.0f}",
                            ha="center",
                            va="center",
                            fontsize=14,
                            color="white"
                        )

            bottom += values

        ax.set_xlabel("Cycling Scenario ID", fontsize=20, labelpad=14)
        ax.set_ylabel("Responses (%)", fontsize=20)
        ax.set_title("Affective Quadrant Distribution per Cycling Scenario", pad=16, fontsize=24, fontweight="bold")
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xticks(x)
        ax.set_xticklabels(video_ids_sorted, ha="center", fontsize=18)

        ax.legend(
            title="Affective States",
            ncol=len(c.AFFECTIVE_STATES),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.5),
            frameon=True,
            fontsize=18,
            title_fontsize=18
        )
        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=150, bbox_inches="tight")

        plt.close(fig)


def plot_disagreement_geometry_vs_cues(
        video_level_scores: pd.DataFrame,
        save_path: Optional[Path] = None,
        video_col: str = "video_id",
        experience_col: str = "valence_mean",
        cue_col: str = "pf_nf_label_entropy",
        figsize: Tuple[int, int] = (18, 6),
        cmap: str = "YlGnBu",
        annotate: bool = True,
        marker_size: int = 110,
        edgecolor: str = "white",
        edge_lw: float = 0.9,
) -> None:
    """
    RQ2 figure: Linking disagreement geometry to perceived environmental cues.
    """

    panels = [
        ("dispersion_mean_distance",
         "Dispersion (mean distance to centroid)",
         "Disagreement magnitude vs cue diversity"),

        ("anisotropy_index",
         "Anisotropy index (directional structure)",
         "Directional structure vs cue diversity"),

        ("affect_state_entropy",
         "Affect-state entropy (quadrant ambiguity)",
         "Categorical ambiguity vs cue diversity"),
    ]

    # --- minimal cleaning
    needed = [video_col, cue_col, experience_col] + [p[0] for p in panels]
    df = video_level_scores.copy()
    df = df[needed].dropna().copy()

    # --- color normalization: center at 0 for valence
    vmin = float(np.nanmin(df[experience_col]))
    vmax = float(np.nanmax(df[experience_col]))
    vmax_abs = max(abs(vmin), abs(vmax))
    norm = plt.Normalize(vmin=-vmax_abs, vmax=vmax_abs)
    cm = plt.get_cmap(cmap)

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True)

    # We need to store the mappable (sc) to create the colorbar later
    sc = None

    for ax, (y_col, y_label, title) in zip(axes, panels):
        sc = ax.scatter(
            df[cue_col],
            df[y_col],
            s=marker_size,
            c=df[experience_col],
            cmap=cm,
            norm=norm,
            edgecolor=edgecolor,
            linewidth=edge_lw,
            alpha=0.95,
            zorder=3,
        )

        if annotate:
            for _, r in df.iterrows():
                ax.annotate(
                    str(int(r[video_col])),
                    (float(r[cue_col]), float(r[y_col])),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=11,
                    alpha=0.85,
                    zorder=4,
                )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Cue entropy (PF + NF)", fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.grid(True, alpha=0.25, zorder=0)


    fig.tight_layout(rect=[0, 0, 0.90, 1])
    cax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Mean valence (scenario centroid)", fontsize=12)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_conditional_displacement_vectors(
    driver_df,
    save_path,
    limit: float = 0.35,
    cmap_name: str = "YlGnBu",
    grid_step: float = 0.1,
    panel_titles=("Positive cues", "Negative cues"),
    suptitle="Cue-conditioned displacement vectors",
    legend_title="Cue key",
):

    sns.set_style("white")
    cue_names = sorted({str(x).split(": ")[1] for x in driver_df["factor"]})
    cue_key = {name: (chr(97 + i) if i < 26 else f"z{i}") for i, name in enumerate(cue_names)}
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=0, vmax=float(driver_df["magnitude"].max()))
    grid_lines = np.arange(-1.0, 1.0 + 1e-9, grid_step)
    fig, axes = plt.subplots(1, 2, figsize=(16, 8.5), sharex=True, sharey=True)

    def _draw_backdrop(ax):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)

        for gl in grid_lines:
            ax.axhline(gl, color="#f0f0f0", lw=0.6, zorder=0)
            ax.axvline(gl, color="#f0f0f0", lw=0.6, zorder=0)

        # origin lines (0,0 = scenario centroid in displacement space)
        ax.axhline(0, color="#888888", lw=1.5, alpha=0.5, zorder=1)
        ax.axvline(0, color="#888888", lw=1.5, alpha=0.5, zorder=1)

    def _draw_vectors(ax, subset):
        for _, row in subset.iterrows():
            color = cmap(norm(row["magnitude"]))
            cue_name = str(row["factor"]).split(": ")[1]
            letter = cue_key[cue_name]

            dx = float(row["valence_pull"])
            dy = float(row["arousal_pull"])

            ax.arrow(
                0, 0, dx, dy,
                head_width=limit * 0.05,
                head_length=limit * 0.05,
                fc=color, ec=color,
                alpha=0.9,
                length_includes_head=True,
                lw=3.0,
                zorder=2,
            )

            ax.text(
                dx * 1.12, dy * 1.12, letter,
                fontsize=14,
                fontweight="bold",
                color="black",
                ha="center",
                va="center",
            )

    # ---- panels
    for i, cue_type in enumerate(["Positive", "Negative"]):
        ax = axes[i]
        subset = driver_df[driver_df["type"] == cue_type].sort_values("magnitude")

        _draw_backdrop(ax)
        _draw_vectors(ax, subset)

        ax.set_title(panel_titles[i], fontsize=20, pad=15)
        ax.set_xlabel(r"$\Delta$ Valence", fontsize=18)
        if i == 0:
            ax.set_ylabel(r"$\Delta$ Arousal", fontsize=18)

    plt.tight_layout(rect=[0, 0.1, 0.92, 0.95])
    pos = axes[1].get_position()
    cbar_ax = fig.add_axes([0.94, pos.y0, 0.015, pos.height])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax).set_label(
        r"Centroid displacement magnitude",
        fontsize=18,
        labelpad=15,
    )

    # ---- legend (same look)
    key_elements = [
        plt.Line2D([0], [0], color="w",
                   label=f"{letter}: {label}")
        for label, letter in cue_key.items()
    ]
    fig.legend(
        handles=key_elements,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.17),
        fontsize=18,
        title=legend_title,
        title_fontsize=20,
        frameon=True,
    )

    plt.suptitle(suptitle, fontsize=24, y=1.02, fontweight="bold")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()



























# -------------------------------------------------------------
# Demographic analysis functions
# -------------------------------------------------------------

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


def icc2_1_anova(df: pd.DataFrame, targets: str, raters: str, scores: str) -> float:
    d = df[[targets, raters, scores]].dropna()
    n = d[targets].nunique()  # Number of scenarios (targets)
    k = d[raters].nunique()  # Number of participants (raters)

    if n < 2 or k < 2: return np.nan

    # Fit two-way ANOVA model
    model = ols(f"{scores} ~ C({targets}) + C({raters})", data=d).fit()
    aov = sm.stats.anova_lm(model, typ=2)

    # Mean Squares
    MS_t = aov.loc[f"C({targets})", "sum_sq"] / aov.loc[f"C({targets})", "df"]
    MS_r = aov.loc[f"C({raters})", "sum_sq"] / aov.loc[f"C({raters})", "df"]
    MS_e = aov.loc["Residual", "sum_sq"] / aov.loc["Residual", "df"]

    # ICC(2,1) formula: Two-way random effects, absolute agreement, single rater
    num = MS_t - MS_e
    denom = MS_t + (k - 1) * MS_e + (k / n) * (MS_r - MS_e)

    return float(num / denom) if denom > 0 else np.nan


def generate_demographic_table(
        df: pd.DataFrame,
        demo_cols: List[str] = c.DEMOGRAPHIC_COLUMNS,
        pid_col: str = c.PARTICIPANT_ID,
        video_col: str = c.VIDEO_ID_COL,
        valence_col: str = c.VALENCE,
        arousal_col: str = c.AROUSAL,
        output_path: Optional[Path] = None,
        compute_icc: bool = True,
        min_n_participants: int = 25,
) -> pd.DataFrame:
    d0 = df.copy()

    # --- ICC Function ---
    def _maybe_icc(g: pd.DataFrame, score_col: str) -> float:
        if not compute_icc: return np.nan
        if g[pid_col].nunique() < min_n_participants: return np.nan
        if g[video_col].nunique() < 2: return np.nan
        return icc2_1_anova(g, targets=video_col, raters=pid_col, scores=score_col)

    def _posthoc_pairwise(data: pd.DataFrame, group_col: str, value_col: str, pid_col: str) -> pd.DataFrame:
        """
        For significant categories, perform pairwise comparisons between levels.
        """
        # Get participant means for each level
        level_means = {}
        for level, g in data.groupby(group_col):
            level_means[level] = g.groupby(pid_col)[value_col].mean().dropna()

        levels = list(level_means.keys())
        comparisons = []

        for i, level1 in enumerate(levels):
            for level2 in levels[i + 1:]:
                stat, p = ttest_ind(level_means[level1], level_means[level2], equal_var=False)
                mean_diff = level_means[level1].mean() - level_means[level2].mean()
                comparisons.append({
                    'Level_1': level1,
                    'Level_2': level2,
                    'Mean_diff': mean_diff,
                    'p_value': p
                })

        return pd.DataFrame(comparisons)

    # ---  Significance Test Helper ---
    def _calc_significance_corrected(data: pd.DataFrame, group_col: str, value_col: str) -> float:

        # calculate the MEAN rating per PARTICIPANT
        groups = [
            g.groupby(pid_col)[value_col].mean().dropna()
            for _, g in data.groupby(group_col)
        ]

        # Filter out groups with insufficient people
        groups = [g for g in groups if len(g) > 1]

        if len(groups) < 2:
            return np.nan
        elif len(groups) == 2:
            # Welch's t-test on PARTICIPANT MEANS
            stat, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
            return p
        else:
            # One-way ANOVA on PARTICIPANT MEANS
            stat, p = stats.f_oneway(*groups)
            return p

    def _fmt_p(p):
        if pd.isna(p): return ""
        if p < 0.001: return "***"
        if p < 0.01: return "**"
        if p < 0.05: return "*"
        return ""

    rows = []

    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 60)

    for col in demo_cols:
        d = d0[d0[col].notna()].copy()
        total_n = d[pid_col].nunique()

        p_val_valence = _calc_significance_corrected(d, col, valence_col)
        p_val_arousal = _calc_significance_corrected(d, col, arousal_col)

        print(f"\n{col}:")
        if pd.notna(p_val_valence):
            print(f"  Valence: p = {p_val_valence:.4f} {_fmt_p(p_val_valence)}")
            # If significant, do post-hoc
            if p_val_valence < 0.05:
                posthoc = _posthoc_pairwise(d, col, valence_col, pid_col)
                print("    Post-hoc pairwise comparisons:")
                for _, row in posthoc.iterrows():
                    print(
                        f"{row['Level_1']} vs {row['Level_2']}: p = {row['p_value']:.4f}, diff = {row['Mean_diff']:.3f}")

        if pd.notna(p_val_arousal):
            print(f"  Arousal: p = {p_val_arousal:.4f} {_fmt_p(p_val_arousal)}")
            # If significant, do post-hoc
            if p_val_arousal < 0.05:
                posthoc = _posthoc_pairwise(d, col, arousal_col, pid_col)
                print("    Post-hoc pairwise comparisons:")
                for _, row in posthoc.iterrows():
                    print(
                        f"{row['Level_1']} vs {row['Level_2']}: p = {row['p_value']:.4f}, diff = {row['Mean_diff']:.3f}")
        else:
            print(f"  Arousal: p = N/A (insufficient groups)")

        rows.append({
            "category": col,
            "level": "All",
            "counts": total_n,
            "%": 100.0,
            "valence (mean)": round(d[valence_col].mean(), 2),
            "valence (sd)": round(d[valence_col].std(), 2),
            "arousal (mean)": round(d[arousal_col].mean(), 2),
            "arousal (sd)": round(d[arousal_col].std(), 2),

            "ICC2_1_valence": round(_maybe_icc(d, valence_col), 3),
            "ICC2_1_arousal": round(_maybe_icc(d, arousal_col), 3),
            "Sig_Valence": _fmt_p(p_val_valence),
            "Sig_Arousal": _fmt_p(p_val_arousal)
        })

        for level, g in d.groupby(col):
            n = g[pid_col].nunique()
            rows.append({
                "category": col,
                "level": str(level),
                "counts": n,
                "%": round(n / total_n * 100, 1) if total_n else np.nan,
                "valence (mean)": round(g[valence_col].mean(), 2),
                "valence (sd)": round(g[valence_col].std(), 2),
                "arousal (mean)": round(g[arousal_col].mean(), 2),
                "arousal (sd)": round(g[arousal_col].std(), 2),
                "ICC2_1_valence": round(_maybe_icc(g, valence_col), 3),
                "ICC2_1_arousal": round(_maybe_icc(g, arousal_col), 3),
                "Sig_Valence": "",
                "Sig_Arousal": ""
            })

    print("\n" + "=" * 60)
    print()

    results = pd.DataFrame(rows)

    if output_path:
        results.to_csv(output_path, index=False)

    return results


def plot_subgroup_metrics_bootstrap(
        files,
        subgroup_cols,
        metrics,
        metric_labels=None,
        overall_path=None,
        n_boot=1000,
        seed=42,
        ci=(2.5, 97.5),
        min_videos=5,
        figsize=(14, 8),
        save_path=None,
):
    rng = np.random.default_rng(seed)

    if metric_labels is None:
        metric_labels = {m: m for m in metrics}

    # ---------------------------
    # Helper: Bootstrap CI & Significance
    # ---------------------------
    def boot_mean_ci_and_p(x, overall_mean):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < min_videos:
            return np.nan, np.nan, np.nan, np.nan

        mean = float(x.mean())
        boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, ci)

        # Hypothesis Test: Is the subgroup mean significantly different from overall_mean?
        # We shift the distribution to the null hypothesis (overall_mean)
        if np.isfinite(overall_mean):
            shifted_boots = boots - mean + overall_mean
            diff = abs(mean - overall_mean)
            p_val = np.mean(np.abs(shifted_boots - overall_mean) >= diff)
        else:
            p_val = np.nan

        return mean, float(lo), float(hi), p_val

    def boot_mean_ci(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < min_videos:
            return np.nan, np.nan, np.nan
        mean = float(x.mean())
        boots = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
        lo, hi = np.percentile(boots, ci)
        return mean, float(lo), float(hi)

    # ---------------------------
    # 1. Load Overall Reference Values (mean + CI)
    # ---------------------------
    overall_ci = {}
    if overall_path is not None:
        odf = pd.read_csv(Path(overall_path))
        for m in metrics:
            if m in odf.columns:
                overall_ci[m] = boot_mean_ci(odf[m].to_numpy(float))
            else:
                overall_ci[m] = (np.nan, np.nan, np.nan)
    else:
        for m in metrics:
            overall_ci[m] = (np.nan, np.nan, np.nan)

    # ---------------------------
    # 2. Process Data (Integrating p-value calculation)
    # ---------------------------
    rows = []
    for sg_name, path in files.items():
        df = pd.read_csv(Path(path))
        sg_col = subgroup_cols[sg_name]

        df = df.dropna(subset=[sg_col]).copy()
        if df.empty:
            continue

        for level, sub in df.groupby(sg_col):
            row = {"group": sg_name, "label": str(level), "level_name": level}
            for m in metrics:
                if m not in sub.columns:
                    row[f"{m}_mean"] = row[f"{m}_lo"] = row[f"{m}_hi"] = row[f"{m}_p"] = np.nan
                else:
                    # Get the reference mean for the hypothesis test
                    ref_mean = overall_ci[m][0]
                    mean, lo, hi, p = boot_mean_ci_and_p(sub[m].to_numpy(float), ref_mean)
                    row[f"{m}_mean"] = mean
                    row[f"{m}_lo"] = lo
                    row[f"{m}_hi"] = hi
                    row[f"{m}_p"] = p
            rows.append(row)

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise ValueError("No data to plot.")

    # ---------------------------
    # 3. Plotting Setup
    # ---------------------------
    group_order = list(files.keys())
    plot_df["group_order"] = pd.Categorical(plot_df["group"], categories=group_order, ordered=True)
    plot_df = plot_df.sort_values(["group_order", "label"], kind="stable").reset_index(drop=True)

    labels = plot_df["label"].tolist()
    y = np.arange(len(labels))[::-1]

    cmap = plt.get_cmap("YlGnBu")
    tones = np.linspace(0.25, 0.85, len(group_order))
    group_color = {g: cmap(t) for g, t in zip(group_order, tones)}
    row_colors = plot_df["group"].map(group_color).tolist()

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize, sharey=True, constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    # ---------------------------
    # 4. Plot Metrics
    # ---------------------------
    for ax, m in zip(axes, metrics):
        data_min = np.inf
        data_max = -np.inf

        overall_mean, overall_lo, overall_hi = overall_ci.get(m, (np.nan, np.nan, np.nan))

        if np.isfinite(overall_lo) and np.isfinite(overall_hi):
            ax.axvspan(overall_lo, overall_hi, color="0.8", alpha=0.25, zorder=0)
            ax.axvline(overall_lo, color="0.65", lw=1.2, ls=(0, (2, 2)), alpha=0.9, zorder=1)
            ax.axvline(overall_hi, color="0.65", lw=1.2, ls=(0, (2, 2)), alpha=0.9, zorder=1)
            data_min = min(data_min, overall_lo)
            data_max = max(data_max, overall_hi)

        if np.isfinite(overall_mean):
            ax.axvline(overall_mean, color="0.55", lw=1.4, ls=(0, (4, 3)), alpha=0.8, zorder=2)
            data_min = min(data_min, overall_mean)
            data_max = max(data_max, overall_mean)

        for i in range(len(plot_df)):
            mean = plot_df.loc[i, f"{m}_mean"]
            lo = plot_df.loc[i, f"{m}_lo"]
            hi = plot_df.loc[i, f"{m}_hi"]
            p_val = plot_df.loc[i, f"{m}_p"]

            if not (np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)):
                continue

            data_min = min(data_min, lo)
            data_max = max(data_max, hi)

            ax.errorbar(
                mean, y[i],
                xerr=[[mean - lo], [hi - mean]],
                fmt="o",
                color=row_colors[i],
                elinewidth=2.0,
                markersize=7,
                capsize=0,
                zorder=3
            )

            # SIGNIFICANCE ANNOTATION
            if np.isfinite(p_val) and p_val < 0.05:
                # Offset the asterisk slightly to the right of the upper CI bound
                ax.text(hi + 0.01, y[i], "*", color="black", va="center", ha="left", fontweight="bold", fontsize=16)

        # Axis limits logic (preserved)
        if np.isfinite(overall_mean) and data_min != np.inf:
            dist_left = overall_mean - data_min
            dist_right = data_max - overall_mean
            max_dist = max(dist_left, dist_right)
            pad = max_dist * 0.25 # Slightly increased padding for asterisks
            limit_dist = max_dist + pad
            ax.set_xlim(overall_mean - limit_dist, overall_mean + limit_dist)
        elif data_min != np.inf:
            pad = (data_max - data_min) * 0.15
            ax.set_xlim(data_min - pad, data_max + pad)

        ax.grid(True, axis="x", alpha=0.18)
        ax.set_title(metric_labels.get(m, m), fontsize=18, weight="bold")
        ax.set_xlabel("Mean (95% CI)", fontsize=16)
        ax.tick_params(axis="x", labelsize=14)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=14)

    # ---------------------------
    # 5. Legend (preserved)
    # ---------------------------
    legend_handles = [Line2D([0], [0], color=group_color[g], lw=4, label=g) for g in group_order]
    legend_handles.append(Line2D([0], [0], color="0.55", lw=1.4, ls=(0, (4, 3)), label="Overall mean"))
    legend_handles.append(Patch(facecolor="0.8", edgecolor="none", alpha=0.25, label="Overall 95% CI"))
    legend_handles.append(Line2D([0], [0], color="black", marker='*', linestyle='None', markersize=5, label="p < 0.05 vs. Overall"))

    fig.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.03),
        ncol=5,
        frameon=False,
        fontsize=16,
        title="Observer Characteristics",
        title_fontsize=18,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()



















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
    figsize = (10, 4)
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
        col_prefix: str = 'AG',
        plot: bool = False
) -> Tuple[float, float, float]:

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

