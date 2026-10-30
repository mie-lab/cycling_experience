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
    "intersection_count": "Intersection count",
    "car_lanes_total_count": "Motorised lane count",
    "greenery_share_gis": "Greenery share (GIS)",
    "building_count": "Building count",
    "tree_canopy_share": "Tree canopy share",
    "average_greenery_share": "Greenery share (Seg.)",
    "average_building_share": "Building share (Seg.)",
    "average_sky_share": "Sky share (Seg.)",
    "average_road_share": "Road share (Seg.)",
    "traffic_volume": "Traffic volume (AADT)",
    "motorized_traffic_speed_kmh": "Speed limit",
    "pois_count": "POI count",
    "pt_route_gis": "Public transport route",
    "ped_and_cycl_count": "Ped. & cyclist count",
    "motor_vehicle_overtakes_count": "Overtake count",
    "unique_motor_vehicles_count": "Vehicle count",
    "heavy_vehicles_count": "Heavy vehicle count",
}


def main():
    """The Dynamic Gap in Bikeability: Sections 4.1-4.4 and Appendix B."""
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
        "intersection_count", "building_count", "pois_count",
        "ped_and_cycl_count", "motor_vehicle_overtakes_count", "unique_motor_vehicles_count",
        "heavy_vehicles_count",
        "SCR_Peaks_N", "SCR_Peaks_Amplitude_Mean", "HRV_RMSSD",
    ]

    # Channels: (features, feature that sets the sign of the index)
    STATIC = (
        ["bike_infra_type_numeric", "intersection_count", "car_lanes_total_count"],
        "bike_infra_type_numeric",
    )
    VISUAL_GIS = (
        ["greenery_share_gis", "building_count", "tree_canopy_share"],
        "greenery_share_gis",
    )
    VISUAL_SEG = (
        ["average_greenery_share", "average_building_share", "average_sky_share", "average_road_share"],
        "average_greenery_share",
    )
    DYNAMIC_GIS = (
        ["traffic_volume", "motorized_traffic_speed_kmh", "pois_count", "pt_route_gis"],
        "traffic_volume",
    )
    DYNAMIC_EVENTS = (
        ["ped_and_cycl_count", "motor_vehicle_overtakes_count", "unique_motor_vehicles_count",
         "heavy_vehicles_count"],
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
        ("building_count", "average_building_share"),
        ("car_lanes_total_count", "average_road_share"),
        ("traffic_volume", "unique_motor_vehicles_count"),
        ("motorized_traffic_speed_kmh", "motor_vehicle_overtakes_count"),
        ("pois_count", "ped_and_cycl_count"),
        ("pt_route_gis", "heavy_vehicles_count"),
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

    demo_cols = ["Gender", "Age", "Cycling_frequency", "Cycling_confidence", "Cycling_purpose",
                 "Cycling_environment"]
    demographics_df = lab_results_df[demo_cols].rename_axis(c.PARTICIPANT_ID).reset_index()

    scores = (
        lab_results_df
        .drop(columns=c.DEMOGRAPHIC_COLUMNS + [c.START, c.END], errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
    )
    task1 = utils.helper_utils.get_trial_dict(scores, experiment_setup, c.TRIAL_1, c.VIDEO_COUNTS)
    ratings = utils.process_utils.add_valence_arousal(
        utils.helper_utils.trial_dict_to_df(task1), ag_col="rating"
    )

    # Physiology per clip, without misaligned segments
    df_physio = pd.read_csv(physio_file)
    misaligned = utils.physio_utils.check_segment_alignment(df_physio, experiment_setup)
    misaligned_ids = set(zip(misaligned[c.PARTICIPANT_ID], misaligned["slot"]))
    keep = [
        (pid, slot) not in misaligned_ids
        for pid, slot in zip(df_physio[c.PARTICIPANT_ID], df_physio["segment_id"])
    ]
    log.info(f"Physio: dropped misaligned segments {sorted(misaligned_ids)}")

    physio = utils.physio_utils.get_physio_trial_df(
        df_physio[keep], experiment_setup, c.TRIAL_1, PHYSIO_METRICS
    )
    physio[c.VIDEO_ID_COL] = (
        physio["sequence_code"].str.extract(r"video_(\d+)", expand=False).astype(int)
    )

    clips = pd.read_csv(ground_truth_file)
    clips["pt_route_gis"] = (clips["bus_line_gis"] | clips["tram_line_gis"]).astype(int)

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

    # --- Section 4.1: sample and affective responses (raw ratings) ---
    n_participants = df[c.PARTICIPANT_ID].nunique()

    table4 = []
    for col in demo_cols:
        for option, group in df.groupby(col):
            table4.append({
                "category": col,
                "option": option,
                "share_pct": 100 * group[c.PARTICIPANT_ID].nunique() / n_participants,
                **{f"{y}_mean": group[y].mean() for y in AFFECT},
                **{f"{y}_sd": group[y].std() for y in AFFECT},
            })
    table4 = pd.DataFrame(table4).round(3)
    table4.to_csv(output_dir / "table4_demographics.csv", index=False)
    log.info(f"Table 4:\n{table4.to_string(index=False)}")

    table5 = []
    for y in AFFECT:
        icc1, icck = sem.rating_reliability(df, y)
        table5.append({
            "dimension": y,
            "mean": df[y].mean(),
            "sd": df[y].std(),
            "q1": df[y].quantile(.25),
            "q3": df[y].quantile(.75),
            "skewness": stats.skew(df[y]),
            "icc1": icc1,
            "icck": icck,
        })
    table5 = pd.DataFrame(table5).round(3)
    table5.to_csv(output_dir / "table5_ratings.csv", index=False)
    log.info(f"Table 5:\n{table5.to_string(index=False)}")

    raters = df.groupby(c.VIDEO_ID_COL)[c.PARTICIPANT_ID].nunique()
    clip_means = df.groupby(c.VIDEO_ID_COL)[AFFECT].mean()
    text = {
        "n_participants": n_participants,
        "n_observations": len(df),
        "retention_pct": 100 * len(df) / N_PLANNED,
        "raters_per_clip_mean": raters.mean(),
        "raters_per_clip_sd": raters.std(),
        "raters_per_clip_min": raters.min(),
        "raters_per_clip_max": raters.max(),
        "va_spearman_obs_raw": stats.spearmanr(df[c.VALENCE], df[c.AROUSAL])[0],
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
            text[f"group_p_{y}_{col}"] = fit.f_pvalue

    text = pd.Series(text).round(4)
    text.to_csv(output_dir / "text_statistics_4_1.csv", header=["value"])
    log.info(f"Text statistics (Section 4.1):\n{text.to_string()}")

    utils.plot_utils.plot_clip_means_va(df, output_dir / "figure3_video_means.png")

    # ==============================================================================
    # PHASE 2: INFORMATION CHANNELS AND INDICES
    # ==============================================================================
    log.info("Phase 2: Information channels and indices")

    df = sem.prepare(df, LOG_VARS, AFFECT + PHYSIO_METRICS, FEATURES)

    # Drift over the session, as in the companion paper
    slot = sem.centre(df, ["segment_id"])["segment_id"]
    for m in PHYSIO_METRICS:
        df[m] = ols(f"{m} ~ slot", df.assign(slot=slot)).fit().resid

    model_data, loadings = {}, []
    for scenario, channels in SCENARIOS.items():
        model_data[scenario], scenario_loadings = sem.channel_indices(df, channels, scenario)
        loadings += scenario_loadings

    loadings = pd.DataFrame(loadings).round(4)
    loadings.to_csv(output_dir / "index_loadings.csv", index=False)
    log.info(f"Index loadings:\n{loadings.to_string(index=False)}")

    # ==============================================================================
    # PHASE 3: PATH ANALYSIS
    # ==============================================================================
    log.info("Phase 3: Path analysis")

    # --- M1-M4: information scenarios (Section 4.2, Figure 4, Table C1) ---
    # Physiology is modelled in parallel with affect, on the rows with complete physiology
    df_complete = df.dropna(subset=PHYSIO_METRICS)
    r2_rows, path_estimates, physio_paths, resid_corr = [], [], [], []

    for model_id, (scenario, channels) in zip(["M1", "M2", "M3", "M4"], SCENARIOS.items()):
        paths, r2, _, _ = sem.fit_path_model(sem.direct_model(AFFECT), model_data[scenario])
        path_estimates.append(paths.assign(model=model_id, scenario=scenario))

        d_physio, _ = sem.channel_indices(df_complete, channels, scenario)
        p_paths, p_r2, p_resid, _ = sem.fit_path_model(sem.direct_model(AFFECT + PHYSIO_METRICS), d_physio)

        r2_rows.append({
            "model": model_id,
            "scenario": scenario,
            **r2,
            **{m: p_r2[m] for m in PHYSIO_METRICS},
            "n_physio": len(d_physio),
        })

        p_paths = p_paths[p_paths["lval"].isin(PHYSIO_METRICS)].copy()
        p_paths["p_holm"] = p_paths.groupby("rval")["p_cluster"].transform(sem.holm)
        physio_paths.append(p_paths.assign(model=model_id, scenario=scenario))

        for m in PHYSIO_METRICS:
            resid_corr.append({
                "model": model_id,
                "scenario": scenario,
                "metric": m,
                **{y: p_resid[m].corr(p_resid[y]) for y in AFFECT},
            })

    r2_table = pd.DataFrame(r2_rows).round(4)
    r2_table.to_csv(output_dir / "r2_scenarios_tableC1.csv", index=False)
    log.info(f"R2 per scenario (Figure 4, Table C1):\n{r2_table.to_string(index=False)}")
    utils.plot_utils.plot_r2_scenarios(r2_table, output_dir / "figure4_r2_scenarios.png")

    pd.concat(physio_paths).to_csv(output_dir / "physio_paths.csv", index=False)

    resid_corr = pd.DataFrame(resid_corr).round(4)
    resid_corr.to_csv(output_dir / "physio_residual_correlations.csv", index=False)
    log.info(f"Max |residual corr| physiology-affect: {resid_corr[AFFECT].abs().max().max():.3f}")

    # --- M4 vs M5: static channel (Section 4.3, Table 6) ---
    d = model_data[MAIN_SCENARIO]
    m4_paths, m4_r2, _, m4_fit = sem.fit_path_model(sem.direct_model(AFFECT), d)
    m5_paths, m5_r2, _, m5_fit = sem.fit_path_model(sem.MEDIATION_MODEL, d)
    path_estimates.append(m5_paths.assign(model="M5", scenario=MAIN_SCENARIO))

    static = m4_paths[m4_paths["rval"] == "StaticIndex"].set_index("lval")
    test = sem.joint_static_test(d, AFFECT)

    table6 = pd.DataFrame([
        {
            "model": "M4", **m4_fit, **m4_r2,
            **{f"static_beta_{y}": static.loc[y, "Est. Std"] for y in AFFECT},
            **{f"static_p_{y}": static.loc[y, "p_cluster"] for y in AFFECT},
        },
        {
            "model": "M5", **m5_fit, **{y: m5_r2[y] for y in AFFECT},
            **{f"clustered_test_{k}": v for k, v in test.items()},
        },
    ]).round(4)
    table6.to_csv(output_dir / "table6_static_channel.csv", index=False)
    log.info(f"Table 6:\n{table6.to_string(index=False)}")

    # --- M6-M8: channel interactions (Section 4.4, Table 7) ---
    table7 = []
    for model_id, term in sem.INTERACTIONS.items():
        paths, r2, _, fit = sem.fit_path_model(sem.direct_model(AFFECT, term), d)
        path_estimates.append(paths.assign(model=model_id, scenario=MAIN_SCENARIO))

        inter = paths[paths["rval"] == term].set_index("lval")
        table7.append({
            "model": model_id, "term": term, **fit, **r2,
            **{f"beta_{y}": inter.loc[y, "Est. Std"] for y in AFFECT},
            **{f"p_{y}": inter.loc[y, "p_cluster"] for y in AFFECT},
        })

    table7 = pd.DataFrame(table7)
    for y in AFFECT:
        table7[f"p_holm_{y}"] = sem.holm(table7[f"p_{y}"])
    table7 = table7.round(4)
    table7.to_csv(output_dir / "table7_interactions.csv", index=False)
    log.info(f"Table 7:\n{table7.to_string(index=False)}")

    pd.concat(path_estimates).to_csv(output_dir / "path_estimates.csv", index=False)

    # ==============================================================================
    # APPENDIX B: INDEX DIAGNOSTICS
    # ==============================================================================
    log.info("Appendix B: Index diagnostics")

    # Figure B1: Spearman correlations of all responses and features
    corr = df[AFFECT + PHYSIO_METRICS + FEATURES].corr(method="spearman")
    utils.plot_utils.plot_spearman_matrix(corr, FEATURE_LABELS, output_dir / "figureB1_correlation_matrix.png")

    # Figure B2: channel indices vs affect
    channel_corr = []
    for scenario in SCENARIOS:
        d = model_data[scenario]
        for index in sem.INDICES:
            channel_corr.append({
                "scenario": scenario,
                "index": index,
                **{y: stats.spearmanr(d[index], d[y])[0] for y in AFFECT},
            })

    channel_corr = pd.DataFrame(channel_corr)
    channel_corr.round(3).to_csv(output_dir / "figureB2_channel_correlations.csv", index=False)
    log.info(f"Channel correlations (Figure B2):\n{channel_corr.round(3).to_string(index=False)}")
    utils.plot_utils.plot_channel_correlations(
        channel_corr, sem.INDICES, list(SCENARIOS), output_dir / "figureB2_channel_correlations.png"
    )

    # Figure B3: each index rebuilt from its three highest-loading features
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
    top3.round(3).to_csv(output_dir / "figureB3_top3.csv", index=False)
    log.info(f"Top-3 check (Figure B3):\n{top3.round(3).to_string(index=False)}")
    utils.plot_utils.plot_top3_bars(top3, output_dir / "figureB3_top3.png")

    # Feature pairs: each GIS proxy vs its ground-level counterpart, across clips
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
    feature_pairs.to_csv(output_dir / "appendixB_feature_pairs.csv", index=False)
    log.info(f"Feature pairs:\n{feature_pairs.to_string(index=False)}")

    # Held-out check without PCA: GIS vs ground-level version of a channel (raw features)
    d_raw = df.copy()
    d_raw[FEATURES] = sem.centre(d_raw, FEATURES)
    channel_versions = {
        "visual": (VISUAL_GIS[0], VISUAL_SEG[0], STATIC[0] + DYNAMIC_EVENTS[0]),
        "dynamic": (DYNAMIC_GIS[0], DYNAMIC_EVENTS[0], STATIC[0] + VISUAL_SEG[0]),
    }

    validity = []
    for channel, (gis, ground, base) in channel_versions.items():
        for y in AFFECT:
            for row in sem.heldout_gain(d_raw, y, base, gis, ground):
                validity.append({"channel": channel, "outcome": y, **row})

    validity = pd.DataFrame(validity).round(4)
    validity.to_csv(output_dir / "appendixB_incremental_validity.csv", index=False)
    log.info(f"Held-out validation:\n{validity.to_string(index=False)}")

    # Held-out scenario gap: each clip predicted from the other clips
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
    scenario_gap.to_csv(output_dir / "appendixB_heldout_scenario_gap.csv", index=False)
    log.info(f"Held-out scenario gap:\n{scenario_gap.to_string(index=False)}")

    log.info(f"Done. Outputs in {output_dir}")


if __name__ == "__main__":
    main()
