import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats

# --- Configuration ---
INPUT_FILE = Path(
    'C:/Users/agrisiute/OneDrive - ETH Zurich/Desktop/thesis/user_study/pilot/transition_technique_feedback.xlsx')
OUTPUT_DIR = Path('C:/Users/agrisiute/OneDrive - ETH Zurich/Desktop/thesis/user_study/pilot/')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

sns.set_context("paper", font_scale=1.0)
sns.set_style("white")

STYLE = {
    'palette': 'YlGnBu',
    'font_title': 12,
    'font_axis_label': 10,
    'font_tick': 9,
    'font_legend': 9,
    'dpi': 300,
}

TECHNIQUES = ['no transition', 'black', 'overlay', 'blur', 'frame interpolation']
TECHNIQUE_LABELS = ['No\ntransition', 'Black\nfade', 'Overlay\nfade', 'Blur', 'Frame\ninterp.']
SCENARIOS = ['Bikeable', 'Non-bikeable', 'Mixed']
SCENARIO_COL_KEYS = ['bikeable', 'non_bikeable', 'mixed']


def normalize_technique(s):
    if pd.isna(s): return None
    s = str(s).lower().strip().replace('tranisition', 'transition').replace('transitiom', 'transition').replace(
        'interpolaion', 'interpolation').replace('interpolaton', 'interpolation')
    return 'no transition' if s == 'without' else s


def parse_multiselect(cell):
    if pd.isna(cell): return []
    return [normalize_technique(x) for x in str(cell).split(',') if
            normalize_technique(x) and normalize_technique(x) != 'none']


def count_by_technique(series):
    counter = Counter()
    for cell in series: counter.update(parse_multiselect(cell))
    return [counter.get(t, 0) for t in TECHNIQUES]


def plot_unified_figure(df, save_path):
    """Creates a single publication-ready figure with 3 panels in one row."""
    cmap = plt.get_cmap(STYLE['palette'])
    scen_colors = {'Bikeable': cmap(0.30), 'Non-bikeable': cmap(0.55), 'Mixed': cmap(0.80)}
    chop_color, cont_color = cmap(0.30), cmap(0.80)

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.25)

    # Removed sharey so each plot gets its own Y-axis labels and ticks
    ax_pref = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_cont = fig.add_subplot(gs[0, 2])

    x = np.arange(len(TECHNIQUES))
    width = 0.27
    y_max = len(df) + 1  # Standardize Y-axis limit across all panels

    # --- Panel A & B: Preference and Distraction ---
    questions = {'A. Most preferred': '4', 'B. Least distracting': '5'}
    for ax, (qname, qprefix) in zip([ax_pref, ax_dist], questions.items()):
        for i, (scen, scen_key) in enumerate(zip(SCENARIOS, SCENARIO_COL_KEYS)):
            counts = count_by_technique(df[f'{qprefix}_{scen_key}'])

            # Removed the condition for label, so the legend applies to both plots
            bars = ax.bar(x + (i - 1) * width, counts, width, color=scen_colors[scen], edgecolor='white', linewidth=0.5,
                          label=scen)

            for bar, c in zip(bars, counts):
                if c > 0: ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(c), ha='center',
                                  va='bottom', fontsize=8)

        ax.set_title(qname, fontsize=STYLE['font_title'], loc='left', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(TECHNIQUE_LABELS, fontsize=STYLE['font_tick'])

        # Apply uniform Y-axis
        ax.set_ylim(0, y_max)
        ax.set_ylabel('Participants', fontsize=STYLE['font_axis_label'])

        ax.grid(True, axis='y', linestyle='--', alpha=0.25)
        sns.despine(ax=ax)

        # Ensure legend is on both plots
        ax.legend(title='Scenario', loc='upper right', frameon=False, fontsize=STYLE['font_legend'])

    # --- Panel C: Continuity ---
    chop, cont = Counter(), Counter()
    for cell in df['chopped']: chop.update(parse_multiselect(cell))
    for cell in df['continuous']: cont.update(parse_multiselect(cell))

    c_width = 0.35
    chop_vals = [chop.get(t, 0) for t in TECHNIQUES]
    cont_vals = [cont.get(t, 0) for t in TECHNIQUES]

    ax_cont.bar(x - c_width / 2, chop_vals, c_width, color=chop_color, edgecolor='white',
                label='Segmenting ("Chopped")')
    ax_cont.bar(x + c_width / 2, cont_vals, c_width, color=cont_color, edgecolor='white', label='Continuous')

    for xi, (c, ct) in enumerate(zip(chop_vals, cont_vals)):
        if c > 0: ax_cont.text(xi - c_width / 2, c + 0.2, str(c), ha='center', va='bottom', fontsize=8)
        if ct > 0: ax_cont.text(xi + c_width / 2, ct + 0.2, str(ct), ha='center', va='bottom', fontsize=8)

    ax_cont.set_title("C. Perceived Continuity", fontsize=STYLE['font_title'], loc='left', fontweight='bold')
    ax_cont.set_xticks(x)
    ax_cont.set_xticklabels(TECHNIQUE_LABELS, fontsize=STYLE['font_tick'])

    # Apply uniform Y-axis
    ax_cont.set_ylim(0, y_max)
    ax_cont.set_ylabel('Participants', fontsize=STYLE['font_axis_label'])

    ax_cont.legend(loc='upper right', frameon=False, fontsize=STYLE['font_legend'])
    ax_cont.grid(True, axis='y', linestyle='--', alpha=0.25)
    sns.despine(ax=ax_cont)

    plt.savefig(save_path, dpi=STYLE['dpi'], bbox_inches='tight')
    plt.close()
    log.info(f'Saved unified figure to {save_path}')


def main():
    df = pd.read_excel(INPUT_FILE)
    df_valid = df.dropna(subset=['gender', 'age']).copy()

    male = (df_valid['gender'].str.lower() == 'male').sum()
    female = (df_valid['gender'].str.lower() == 'female').sum()
    log.info(f'Valid participants: N = {len(df_valid)} ({male} male, {female} female)')
    log.info(f'Age: mean {df_valid["age"].mean():.1f}, sd {df_valid["age"].std():.1f}')

    # 1. Manipulation Check Analysis
    valid_responses = df_valid['different_scenarios'].dropna().astype(str).str.lower()
    noticed_diff = valid_responses[valid_responses.str.contains('yes|differen', regex=True)].count()
    log.info(f'Manipulation check: {noticed_diff}/{len(df_valid)} noticed environmental differences.')

    # 2. Negative Associations Analysis
    items = []
    for cell in df_valid['negative_association']: items.extend(parse_multiselect(cell))
    counts = Counter(items)
    log.info('\nNegative associations (participants who flagged technique as off-putting):')
    for t in TECHNIQUES: log.info(f'  {t:20s} {counts.get(t, 0)}')

    # 3. Plot Unified Figure
    plot_unified_figure(df_valid, OUTPUT_DIR / 'unified_pilot_results.png')

if __name__ == '__main__':
    main()