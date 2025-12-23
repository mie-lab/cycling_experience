import os
from pathlib import Path
from typing import Any
import re
import ast
import numpy as np
from shapely import LineString
import pandas as pd
import geopandas as gpd
from typing import List, Union
import constants as c
import logging


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


def filter_aggregate_results(
        df: pd.DataFrame,
        consent: bool = True,
        duration: bool = True,
        location: bool = False,
        gender: bool = False,
        age: bool = False,
        cycling_environment: bool = False,
        cycling_frequency: bool = False,
        cycling_confidence: bool = False,
        by_country: bool = False
) -> pd.DataFrame:
    """
    Implements filtering and aggregation logic to clean the survey response data
    :param consent: bool = True, filter out entries without consent.
    :param duration: bool = True, filter out entries with duration less than 15 minutes.
    :param location: bool = True, filter out entries with missing location data.
    :param by_country: bool = False, if True, apply location filter by country.
    :param age: bool = True, aggregate age groups 46-55, 56-65, +65 into '46+ years'.
    :param gender: bool = True, only keep 'Male' and 'Female' entries.
    :param cycling_frequency: bool = False, aggregate into 'Infrequent', 'Occasional', 'Regular'.
    :param cycling_environment: bool = False, filter out 'Other' responses.
    :param cycling_confidence: bool = False, aggregate into 'Confident' and 'Not confident'.
    :param df: DataFrame containing the data to filter.
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
        mask &= (end - start) >= pd.Timedelta(minutes=15)

    if location:
        mask &= df[c.COUNTRY] != '_'
        if by_country:
            mask &= df[c.COUNTRY].isin(['Switzerland', 'CH'])

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
            "1-3 times/month": "Occasional",
            "1-2 days/week": "Occasional",
            "3-4 days/week": "Regular",
            "5-6 days/week": "Regular",
            "Every day": "Regular"
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

    return df.loc[mask]


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

    return df


def calculate_video_level_scores(
        long_df: pd.DataFrame,
        rating_col: str = 'rating',
        grid_size: int = 10
) -> pd.DataFrame:
    """
    Calculate video-level centroids for valence and arousal based on ratings.
    :param long_df: DataFrame containing long-format data with video IDs and ratings.
    :param rating_col: Column name containing the ratings (default is 'rating').
    :param grid_size: Size of the rating grid (default is 10).
    :return: DataFrame with video IDs and their corresponding valence and arousal centroids.
    """
    # TODO: Check if simple merge of the means for valence and arousal is sufficient

    rows = []

    for vid in sorted(long_df[c.VIDEO_ID_COL].unique()):
        ratings = pd.to_numeric(long_df.loc[long_df[c.VIDEO_ID_COL] == vid, rating_col], errors='coerce').dropna()
        ratings = ratings[(ratings >= 1) & (ratings <= grid_size ** 2)].astype(int)
        heatmap = np.zeros((grid_size, grid_size), int)
        coords = ratings - 1
        row_coords, col_coords = divmod(coords, grid_size)
        np.add.at(heatmap, (row_coords, col_coords), 1)

        total = heatmap.sum()
        if total == 0:
            rows.append((vid, np.nan, np.nan))
            continue

        X, Y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
        center = 5

        cx = (heatmap * X).sum() / total
        cy = (heatmap * Y).sum() / total

        norm_x = (cx - center) / center
        norm_y = (center - cy) / center
        rows.append((vid, norm_x, norm_y))

    aggregated_df = pd.DataFrame(rows, columns=[c.VIDEO_ID_COL, c.VALENCE, c.AROUSAL])

    return aggregated_df


def add_midpoint_centered_column(
        df: pd.DataFrame,
        col_name: str,
        category_order: list,
) -> pd.DataFrame:
    """
    Add a new column to the DataFrame with values centered around the midpoint of the category order.
    :param df: DataFrame to modify.
    :param col_name: Column name containing the categorical data.
    :param category_order: List of categories in the desired order.
    :return: DataFrame with the new centered column.
    """

    # Dynamically calculate the midpoint from the category list
    midpoint = len(category_order) // 2
    codes = pd.Categorical(df[col_name], categories=category_order, ordered=True).codes
    new_col_name = f"{col_name}_centered"
    df[new_col_name] = codes - midpoint

    return df


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


def factor_counts(
        df: pd.DataFrame,
        col: str,
        prefix: str,
        group_by: str = c.VIDEO_ID_COL
) -> pd.DataFrame:
    """
    Create a DataFrame with counts of unique labels in a specified column, grouped by video_id.
    :param df: DataFrame containing the data.
    :param col: Column name containing the labels to count.
    :param prefix: Prefix to add to the label columns in the output DataFrame.
    :param group_by: Column name to group by (default is 'video_id').
    :return: DataFrame with video_id and counts of unique labels.
    """
    t = (df[[group_by, col]]
         .assign(label=lambda d: d[col].fillna('').str.split(';'))
         .explode('label'))
    t = t.loc[t['label'] != '']
    if t.empty:
        return pd.DataFrame({group_by: []})
    return (t.groupby([group_by, 'label']).size()
            .unstack(fill_value=0)
            .add_prefix(f'{prefix}: ')
            .reset_index())


def aggregate_video_level_scores(
        df: pd.DataFrame,
        group_by: str = c.VIDEO_ID_COL,
) -> pd.DataFrame:
    """
    Aggregate video-level scores from a long DataFrame containing valence and arousal scores.
    :param df: DataFrame containing the scores with columns 'video_id', 'valence', and 'arousal'.
    :param group_by: Column name to group by (default is 'video_id').
    :return: DataFrame with aggregated scores and counts of PF and NF factors.
    """
    video_level_scores = calculate_video_level_scores(df, rating_col=c.AG)

    pf_counts = factor_counts(df, c.PF, c.PF, group_by)
    nf_counts = factor_counts(df, c.NF, c.NF, group_by)

    video_level_scores = (video_level_scores
                          .merge(pf_counts, on=group_by, how='left')
                          .merge(nf_counts, on=group_by, how='left'))

    count_cols = video_level_scores.filter(regex=r'^(PF|NF): ').columns
    video_level_scores[count_cols] = video_level_scores[count_cols].fillna(0)

    return video_level_scores


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


def assign_affective_states(survey_results_df):
    """
    Assign affective states based on valence and arousal scores.
    :param survey_results_df: DataFrame containing 'valence' and 'arousal' columns.
    :return: DataFrame with an additional 'affective_state' column.
    """
    conditions = [
        (survey_results_df['valence'] > 0) & (survey_results_df['arousal'] > 0),
        (survey_results_df['valence'] > 0) & (survey_results_df['arousal'] <= 0),
        (survey_results_df['valence'] <= 0) & (survey_results_df['arousal'] > 0),
        (survey_results_df['valence'] <= 0) & (survey_results_df['arousal'] <= 0)
    ]

    categories = ['Activation', 'Contentment', 'Tension', 'Deactivation']
    survey_results_df['affective_state'] = np.select(conditions, categories, default='Unknown')

    for cat in categories:
        survey_results_df[f'is_{cat}'] = (survey_results_df['affective_state'] == cat).astype(int)

    return survey_results_df