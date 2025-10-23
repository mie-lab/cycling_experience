import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import configparser
from pathlib import Path
import constants as c

# ==============================================================================
# PHASE 0: SETUP & CONFIGURATION (Using your provided paths)
# ==============================================================================
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read("config.ini")
lab_results_file = Path(config["filenames"]["lab_study_results_file"])
ipq_file = "C:/Users/agrisiute/OneDrive - ETH Zurich/Desktop/thesis/user_study/online survey/input_data/lab_results/IPQ_results(0-59).xlsx"


def plot_barcharts_by_group(data, group_by_col, title):

    question_cols = [f'Q{i}' for i in range(1, 15)]
    grouped_means = data.groupby(group_by_col)[question_cols].mean()

    print(f"\n--- Mean Scores for Each Question by {group_by_col} ---")
    print(grouped_means.round(2).T)

    groups = grouped_means.index
    num_groups = len(groups)

    fig, axes = plt.subplots(num_groups, 1, figsize=(10, 6 * num_groups), sharex=True)
    if num_groups == 1:
        axes = [axes]

    cmap = plt.cm.coolwarm
    max_abs_val = grouped_means.abs().max().max()
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs_val, vcenter=0, vmax=max_abs_val)

    for i, group_name in enumerate(groups):
        ax = axes[i]
        means = grouped_means.loc[group_name]

        ax.barh(
            y=means.index,
            width=means.values,
            color=cmap(norm(means.values)),
            edgecolor='black',
            linewidth=0.5
        )
        ax.axvline(0, color='grey', linestyle='--')
        ax.set_title(f'Group: {group_name}', fontsize=14)
        ax.invert_yaxis()
        ax.set_xlim(-3, 3)

    fig.suptitle(title, fontsize=18, y=1.02)
    plt.xlabel('Mean Score (-3 to +3)', fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    filename = f'barchart_panel_by_{group_by_col.lower()}.png'
    plt.savefig(filename)
    print(f"Saved bar chart panel to '{filename}'")


# ==========================================================================
# PHASE 1: LOAD & MERGE DATA
# ==========================================================================
try:
    lab_results_df = pd.read_excel(lab_results_file).set_index(c.PARTICIPANT_ID, drop=True)
    demographics_df = lab_results_df[c.DEMOGRAPHIC_COLUMNS].copy()
    demographics_df.reset_index(inplace=True)

    ipq_df = pd.read_excel(ipq_file)
    rename_dict = {ipq_df.columns[0]: c.PARTICIPANT_ID}
    rename_dict.update({col: f'Q{i + 1}' for i, col in enumerate(ipq_df.columns[1:])})
    ipq_df.rename(columns=rename_dict, inplace=True)

    merged_df = pd.merge(demographics_df, ipq_df, on=c.PARTICIPANT_ID)

    # ==========================================================================
    # PHASE 2: PLOT BAR CHART PANELS BY DEMOGRAPHIC GROUPS
    # ==========================================================================
    plot_barcharts_by_group(
        data=merged_df,
        group_by_col='Gender',
        title='Mean IPQ Scores for Each Question, Grouped by Gender'
    )
    plot_barcharts_by_group(
        data=merged_df,
        group_by_col='Age',
        title='Mean IPQ Scores for Each Question, Grouped by Gender'
    )

    plot_barcharts_by_group(
        data=merged_df,
        group_by_col='Cycling_confidence',
        title='Mean IPQ Scores for Each Question, Grouped by Cycling Confidence'
    )

    plot_barcharts_by_group(
        data=merged_df,
        group_by_col='Cycling_purpose',
        title='Mean IPQ Scores for Each Question, Grouped by Cycling purpose'
    )

    plot_barcharts_by_group(
        data=merged_df,
        group_by_col='Cycling_frequency',
        title='Mean IPQ Scores for Each Question, Grouped by Cycling frequency'
    )

    plot_barcharts_by_group(
        data=merged_df,
        group_by_col='Cycling_environment',
        title='Mean IPQ Scores for Each Question, Grouped by Cycling environment'
    )

except FileNotFoundError as e:
    print(f"\nERROR: A file was not found. Please check your config.ini and absolute paths.")
    print(f"File not found: {e.filename}")
except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")