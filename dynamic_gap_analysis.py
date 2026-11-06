import configparser
import logging
from pathlib import Path

import pandas as pd
from scipy import stats
from statsmodels.formula.api import ols

import constants as c
import utils.helper_utils
import utils.physio_utils
import utils.plot_utils
import utils.process_utils
import utils.sem_utils as sem

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

FEATURE_LABELS = {
    "valence": "Valence",
    "arousal": "Arousal",
    "SCL_Mean": "SCL",
    "SCR_Peaks_N": "SCR peaks",
    "SCR_Peaks_Amplitude_Mean": "SCR amplitude",
    "PPG_Rate_Mean": "Mean heart rate",
    "HRV_RMSSD": "HRV RMSSD",
    "bike_infra_type_numeric": "Bike infra. level",
    "pt_lane_osm": "Public transport lane",
    "intersection_count_osm": "Intersection count",
    "car_lanes_osm": "Motorised lane count",
    "side_parking_presence": "Side parking",
    "greenery_share_gis": "Greenery share (GIS)",
    "building_count_osm": "Building count",
    "tree_canopy_share": "Tree canopy share",
    "average_greenery_share": "Greenery share (Seg.)",
    "average_building_share": "Building share (Seg.)",
    "average_sky_share": "Sky share (Seg.)",
    "average_road_share": "Road share (Seg.)",
    "traffic_volume": "Traffic volume (AADT)",
    "heavy_traffic_volume": "Heavy traffic volume (AADT)",
    "motorized_traffic_speed_kmh": "Speed limit",
    "pois_count_osm": "POI count",
    "ped_and_cycl_count": "Ped. & cyclist count",
    "motor_vehicle_overtakes_count": "Overtake count",
    "light_vehicles_count": "Light vehicle count",
    "traffic_speed_mllm_kmh": "Traffic speed",
    "heavy_vehicles_count": "Heavy vehicle count",
}


def main():
    """The Dynamic Gap in Bikeability: sample, channel indices, path models and appendix checks."""
    # ==============================================================================
    # PHASE 0: SETUP & CONFIGURATION
    # ==============================================================================
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("config.ini")

    lab_results_file = Path(config["filenames"]["lab_study_results_file"])
    lab_setup_file = Path(config["filenames"]["lab_experiment_setup_file"])
    physio_file = Path(config["filenames"]["physiological_results_file"])
    ground_truth_file = Path(config["filenames"]["video_info_ground_truth"])

    output_dir = Path(config["paths"]["dynamic_gap_output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    PHYSIO_EXCLUDE = {0, 3, 12, 24}  # recording failure (0, 3, 24), medical reason (12)
    N_PLANNED = 600  # 60 participants x 10 clips
    AFFECT = [c.VALENCE, c.AROUSAL]
    PHYSIO_METRICS = ["SCL_Mean", "SCR_Peaks_N", "SCR_Peaks_Amplitude_Mean", "PPG_Rate_Mean", "HRV_RMSSD"]
    LOG_VARS = [
        "intersection_count_osm", "building_count_osm", "pois_count_osm", "heavy_traffic_volume",
        "ped_and_cycl_count", "motor_vehicle_overtakes_count",
        "light_vehicles_count", "heavy_vehicles_count",
        "SCR_Peaks_N", "SCR_Peaks_Amplitude_Mean", "HRV_RMSSD",
    ]

    # Channels: (features, feature that sets the sign of the index)
    STATIC = (
        ["bike_infra_type_numeric", "pt_lane_osm", "intersection_count_osm", "car_lanes_osm", "side_parking_presence"],
        "bike_infra_type_numeric",
    )
    VISUAL_GIS = (
        ["greenery_share_gis", "building_count_osm", "tree_canopy_share"],
        "greenery_share_gis",
    )
    VISUAL_SEG = (
        ["average_greenery_share", "average_building_share", "average_sky_share", "average_road_share"],
        "average_greenery_share",
    )
    DYNAMIC_GIS = (
        ["traffic_volume", "heavy_traffic_volume", "motorized_traffic_speed_kmh", "pois_count_osm"],
        "traffic_volume",
    )
    DYNAMIC_EVENTS = (
        ["ped_and_cycl_count", "light_vehicles_count", "motor_vehicle_overtakes_count",
         "heavy_vehicles_count", "traffic_speed_mllm_kmh"],
        "motor_vehicle_overtakes_count",
    )

    SCENARIOS = {
        "Planner": (STATIC, VISUAL_GIS, DYNAMIC_GIS),             # M1
        "Visual": (STATIC, VISUAL_SEG, DYNAMIC_GIS),              # M2
        "Temporal": (STATIC, VISUAL_GIS, DYNAMIC_EVENTS),         # M3
        "Visual-Temporal": (STATIC, VISUAL_SEG, DYNAMIC_EVENTS),  # M4
    }
    MAIN_SCENARIO = "Visual-Temporal"  # M5-M8

    FEATURES = []
    for channels in SCENARIOS.values():
        for cols, _ in channels:
            FEATURES += [f for f in cols if f not in FEATURES]

    # GIS proxy -> ground-level counterpart
    FEATURE_PAIRS = [
        ("greenery_share_gis", "average_greenery_share"),
        ("tree_canopy_share", "average_sky_share"),
        ("building_count_osm", "average_building_share"),
        ("car_lanes_osm", "average_road_share"),
        ("traffic_volume", "light_vehicles_count"),
        ("traffic_volume", "motor_vehicle_overtakes_count"),
        ("motorized_traffic_speed_kmh", "traffic_speed_mllm_kmh"),
        ("pois_count_osm", "ped_and_cycl_count"),
        ("heavy_traffic_volume", "heavy_vehicles_count"),
    ]

    # ==============================================================================
    # PHASE 1: CYCLING EXPERIENCE DATA
    # ==============================================================================
    log.info("Phase 1: Cycling experience data")

    lab_results_df = pd.read_excel(lab_results_file).set_index(c.PARTICIPANT_ID)
    lab_results_df = utils.process_utils.agg_by_chars(
        lab_results_df,
        cycling_frequency=True,
        cycling_confidence=True,
        cycling_purpose=True,
    )
    experiment_setup = pd.read_csv(lab_setup_file, header=None).set_index(0)

    demo_cols = ["Gender", "Age", "Cycling_frequency", "Cycling_confidence", "Cycling_purpose", "Cycling_environment"]
    demographics_df = lab_results_df[demo_cols].rename_axis(c.PARTICIPANT_ID).reset_index()

    scores = (
        lab_results_df
        .drop(columns=c.DEMOGRAPHIC_COLUMNS + [c.START, c.END], errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
    )
    task1 = utils.helper_utils.get_trial_dict(scores, experiment_setup, c.TRIAL_1, c.VIDEO_COUNTS)
    ratings = utils.process_utils.add_valence_arousal(utils.helper_utils.trial_dict_to_df(task1), ag_col="rating")

    # Physio segment k = presentation slot k: flag segments whose duration does not fit
    df_physio = pd.read_csv(physio_file)
    misaligned = utils.physio_utils.check_segment_alignment(df_physio, experiment_setup)
    if not misaligned.empty:
        log.warning(f"Misaligned physio segments:\n{misaligned.to_string(index=False)}")

    # Drop excluded participants and misaligned segments
    misaligned_ids = set(zip(misaligned[c.PARTICIPANT_ID], misaligned["slot"]))
    keep = [
        pid not in PHYSIO_EXCLUDE and (pid, slot) not in misaligned_ids
        for pid, slot in zip(df_physio[c.PARTICIPANT_ID], df_physio["segment_id"])
    ]
    df_physio = df_physio[keep].copy()

    # Physiology per clip
    physio = utils.physio_utils.get_physio_trial_df(df_physio, experiment_setup, c.TRIAL_1, PHYSIO_METRICS)
    physio[c.VIDEO_ID_COL] = physio["sequence_code"].str.extract(r"video_(\d+)", expand=False).astype(int)

    clips = pd.read_csv(ground_truth_file)

    df = (
        ratings
        .merge(physio[[c.PARTICIPANT_ID, c.VIDEO_ID_COL, "segment_id"] + PHYSIO_METRICS],
               on=[c.PARTICIPANT_ID, c.VIDEO_ID_COL], how="left")
        .merge(clips[[c.VIDEO_ID_COL] + FEATURES], on=c.VIDEO_ID_COL, how="left")
        .merge(demographics_df, on=c.PARTICIPANT_ID, how="left")
    )
    df = df[~df[c.PARTICIPANT_ID].isin(PHYSIO_EXCLUDE)].reset_index(drop=True)

    log.info(f"Data: {len(df)} rows, {df[c.PARTICIPANT_ID].nunique()} participants, "
             f"{df[c.VIDEO_ID_COL].nunique()} clips")

    # --- Sample and ratings ---
    n_participants = df[c.PARTICIPANT_ID].nunique()

    demographics = []
    for col in demo_cols:
        for option, group in df.groupby(col):
            demographics.append({
                "category": col,
                "option": option,
                "share_pct": 100 * group[c.PARTICIPANT_ID].nunique() / n_participants,
                **{f"{y}_mean": group[y].mean() for y in AFFECT},
                **{f"{y}_sd": group[y].std() for y in AFFECT},
            })
    demographics = pd.DataFrame(demographics).round(3)
    demographics.to_csv(output_dir / "sample_demographics.csv", index=False)
    log.info(f"Sample demographics:\n{demographics.to_string(index=False)}")

    rating_summary = []
    for y in AFFECT:
        icc1, icck = sem.rating_reliability(df, y)
        rating_summary.append({
            "dimension": y,
            "mean": df[y].mean(),
            "sd": df[y].std(),
            "q1": df[y].quantile(.25),
            "q3": df[y].quantile(.75),
            "skewness": stats.skew(df[y]),
            "icc1": icc1,
            "icck": icck,
        })
    rating_summary = pd.DataFrame(rating_summary).round(3)
    rating_summary.to_csv(output_dir / "rating_summary.csv", index=False)
    log.info(f"Rating summary:\n{rating_summary.to_string(index=False)}")

    raters = df.groupby(c.VIDEO_ID_COL)[c.PARTICIPANT_ID].nunique()
    clip_means = df.groupby(c.VIDEO_ID_COL)[AFFECT].mean()
    sample = {
        "n_participants": n_participants,
        "n_observations": len(df),
        "retention_pct": 100 * len(df) / N_PLANNED,
        "raters_per_clip_mean": raters.mean(),
        "raters_per_clip_sd": raters.std(),
        "raters_per_clip_min": raters.min(),
        "raters_per_clip_max": raters.max(),
        "va_spearman_obs_centred": stats.spearmanr(*sem.centre(df, AFFECT).T.values)[0],
        "va_pearson_clip_means": stats.pearsonr(clip_means[c.VALENCE], clip_means[c.AROUSAL])[0],
    }

    # Group differences: participant-clustered OLS, main groups only
    main_groups = {
        "Gender": ["Male", "Female"],
        "Cycling_environment": ["Urban area", "Rural area"],
        "Cycling_purpose": ["Commuting", "Recreational"],
    }
    for y in AFFECT:
        for col in demo_cols:
            subset = df[df[col].isin(main_groups[col])] if col in main_groups else df
            fit = ols(f"{y} ~ C({col})", subset).fit(
                cov_type="cluster", cov_kwds={"groups": subset[c.PARTICIPANT_ID]}
            )
            sample[f"group_p_{y}_{col}"] = fit.f_pvalue

    sample = pd.Series(sample).round(4)
    sample.to_csv(output_dir / "sample_statistics.csv", header=["value"])
    log.info(f"Sample statistics:\n{sample.to_string()}")

    utils.plot_utils.plot_clip_means_va(df, output_dir / "clip_mean_ratings.png")

    # ==============================================================================
    # PHASE 2: INFORMATION CHANNELS AND INDICES
    # ==============================================================================
    log.info("Phase 2: Information channels and indices")

    df = sem.prepare(df, LOG_VARS, AFFECT + PHYSIO_METRICS, FEATURES)

    # Drift over the session, as in the companion paper
    slot = sem.centre(df, ["segment_id"])["segment_id"]
    for m in PHYSIO_METRICS:
        df[m] = ols(f"{m} ~ slot", df.assign(slot=slot)).fit().resid

    model_data = {scenario: sem.channel_indices(df, channels) for scenario, channels in SCENARIOS.items()}

    # ==============================================================================
    # PHASE 3: PATH ANALYSIS
    # ==============================================================================
    log.info("Phase 3: Path analysis")

    # --- M1-M4: information scenarios ---
    # Physiology uses the same indices, on the rows with a complete recording
    r2_rows, path_estimates, physio_paths = [], [], []

    for model_id, (scenario, d) in zip(["M1", "M2", "M3", "M4"], model_data.items()):
        paths, r2, _ = sem.fit_path_model(sem.direct_model(AFFECT), d)
        path_estimates.append(paths.assign(model=model_id, scenario=scenario))
        if model_id == "M4":
            m4_paths, m4_r2 = paths, r2

        d_physio = d.dropna(subset=PHYSIO_METRICS)
        p_paths, p_r2, _ = sem.fit_path_model(sem.direct_model(PHYSIO_METRICS), d_physio)
        p_paths["p_holm"] = p_paths.groupby("rval")["p_cluster"].transform(sem.holm)
        physio_paths.append(p_paths.assign(model=model_id, scenario=scenario))

        r2_rows.append({"model": model_id, "scenario": scenario, **r2, **p_r2, "n_physio": len(d_physio)})

    explained = pd.DataFrame(r2_rows).round(4)
    explained.to_csv(output_dir / "explained_variance_by_scenario.csv", index=False)
    log.info(f"Explained variance per scenario:\n{explained.to_string(index=False)}")
    utils.plot_utils.plot_r2_scenarios(explained, output_dir / "explained_variance_by_scenario.png")

    physio_paths = pd.concat(physio_paths).round(4)
    physio_paths.to_csv(output_dir / "physiology_paths.csv", index=False)
    log.info(f"Physiology paths with p_Holm < .05:\n"
             f"{physio_paths[physio_paths['p_holm'] < .05].to_string(index=False)}")

    # --- M4 vs M5: static channel mediation comparison ---
    d = model_data[MAIN_SCENARIO]
    m5_paths, m5_r2, _ = sem.fit_path_model(sem.MEDIATION_MODEL, d)
    path_estimates.append(m5_paths.assign(model="M5", scenario=MAIN_SCENARIO))

    static = m4_paths[m4_paths["rval"] == "StaticIndex"].set_index("lval")
    static_channel = pd.DataFrame([
        {
            "outcome": y,
            "r2_m5": m5_r2[y],
            "r2_m4": m4_r2[y],
            "static_beta_m4": static.loc[y, "Est. Std"],
            "static_p_m4": static.loc[y, "p_cluster"],
        }
        for y in AFFECT
    ]).round(4)
    static_channel.to_csv(output_dir / "static_channel_direct_paths.csv", index=False)
    log.info(f"Static channel, M4 vs M5:\n{static_channel.to_string(index=False)}")

    # --- M6-M8: channel interactions ---
    interactions = []
    for model_id, term in sem.INTERACTIONS.items():
        paths, r2, fit = sem.fit_path_model(sem.direct_model(AFFECT, term), d)
        path_estimates.append(paths.assign(model=model_id, scenario=MAIN_SCENARIO))

        inter = paths[paths["rval"] == term].set_index("lval")
        interactions.append({
            "model": model_id, "term": term, **fit, **r2,
            **{f"beta_{y}": inter.loc[y, "Est. Std"] for y in AFFECT},
            **{f"p_{y}": inter.loc[y, "p_cluster"] for y in AFFECT},
        })

    interactions = pd.DataFrame(interactions)
    for y in AFFECT:
        interactions[f"p_holm_{y}"] = sem.holm(interactions[f"p_{y}"])
    interactions = interactions.round(4)
    interactions.to_csv(output_dir / "interaction_estimates.csv", index=False)
    log.info(f"Interaction estimates:\n{interactions.to_string(index=False)}")

    pd.concat(path_estimates).to_csv(output_dir / "path_estimates.csv", index=False)

    # ==============================================================================
    # APPENDIX A: DIAGNOSTICS (what the features and indices measure)
    # ==============================================================================
    log.info("Appendix: diagnostics")

    # Spearman correlations of all responses and features
    corr = df[AFFECT + PHYSIO_METRICS + FEATURES].corr(method="spearman")
    utils.plot_utils.plot_spearman_matrix(corr, FEATURE_LABELS, output_dir / "feature_correlation_matrix.png")

    # Each GIS proxy vs its ground-level counterpart, across clips (numbers for the feature figure)
    clip_df = df.groupby(c.VIDEO_ID_COL)[FEATURES + AFFECT].mean()
    feature_pairs = []
    for gis, ground in FEATURE_PAIRS:
        feature_pairs.append({
            "gis": gis,
            "ground": ground,
            "rho_pair": stats.spearmanr(clip_df[gis], clip_df[ground])[0],
            **{f"rho_{y}_gis": stats.spearmanr(clip_df[gis], clip_df[y])[0] for y in AFFECT},
            **{f"rho_{y}_ground": stats.spearmanr(clip_df[ground], clip_df[y])[0] for y in AFFECT},
        })

    feature_pairs = pd.DataFrame(feature_pairs).round(3)
    feature_pairs.to_csv(output_dir / "feature_pairs.csv", index=False)
    log.info(f"Feature pairs:\n{feature_pairs.to_string(index=False)}")

    # Each index: what it is made of and whether it relates to affect in the expected direction
    indices = {
        "Static": STATIC,
        "Visual (GIS)": VISUAL_GIS,
        "Visual (segmentation)": VISUAL_SEG,
        "Dynamic (GIS)": DYNAMIC_GIS,
        "Dynamic (events)": DYNAMIC_EVENTS,
    }
    index_summary = []
    for name, (cols, ref) in indices.items():
        score, loadings, variance = sem.pc1_index(df, cols, ref)
        score = sem.centre(df.assign(_s=score), ["_s"])["_s"]
        rho = {f"rho_{y}": stats.spearmanr(score, df[y])[0] for y in AFFECT}
        for f in cols:
            index_summary.append({
                "index": name, "variance_explained": variance, **rho, "feature": f, "loading": loadings[f],
            })

    index_summary = pd.DataFrame(index_summary).round(3)
    index_summary.to_csv(output_dir / "index_summary.csv", index=False)
    log.info(f"Index summary:\n{index_summary.to_string(index=False)}")

    # ==============================================================================
    # APPENDIX B: ROBUSTNESS (does the dynamic gap depend on the index size or the 42 clips?)
    # ==============================================================================
    log.info("Appendix: robustness checks")

    # Each index rebuilt from its three highest-loading features (equal size for all channels)
    top3 = []
    for scenario, channels in SCENARIOS.items():
        for index, (cols, ref) in zip(sem.INDICES, channels):
            _, load, _ = sem.pc1_index(df, cols, ref)
            top = load.abs().nlargest(3).index.tolist()

            score, _, _ = sem.pc1_index(df, top, ref if ref in top else top[0])
            score = sem.centre(df.assign(_s=score), ["_s"])["_s"]

            top3.append({
                "scenario": scenario,
                "index": index,
                "features": " | ".join(top),
                **{y: stats.spearmanr(score, df[y])[0] for y in AFFECT},
            })

    top3 = pd.DataFrame(top3)
    top3.round(3).to_csv(output_dir / "top3_index_correlations.csv", index=False)
    log.info(f"Top-3 index correlations:\n{top3.round(3).to_string(index=False)}")
    utils.plot_utils.plot_top3_bars(top3, output_dir / "top3_index_correlations.png")

    # Scenario comparison on held-out clips: each clip predicted from the other clips
    comparisons = [
        ("Planner", "Visual"),
        ("Planner", "Temporal"),
        ("Planner", "Visual-Temporal"),
        ("Temporal", "Visual-Temporal"),
    ]
    clip_ids = df[c.VIDEO_ID_COL].to_numpy()

    scenario_gap = []
    for a, b in comparisons:
        for y in AFFECT:
            result = sem.heldout_comparison(
                model_data[a][sem.INDICES].to_numpy(),
                model_data[b][sem.INDICES].to_numpy(),
                df[y].to_numpy(),
                clip_ids,
            )
            scenario_gap.append({"scenario_a": a, "scenario_b": b, "outcome": y, **result})

    scenario_gap = pd.DataFrame(scenario_gap).round(4)
    scenario_gap.to_csv(output_dir / "heldout_scenario_gap.csv", index=False)
    log.info(f"Held-out scenario gap:\n{scenario_gap.to_string(index=False)}")

    log.info(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
