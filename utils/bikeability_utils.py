import numpy as np
import pandas as pd
import geopandas as gpd
import utils.process_utils as P

BUFFER = 15  # m, as in the video ground truth
INDICES = ['lts', 'bsl', 'bci', 'blos', 'weikl', 'netascore', 'danish_blos']
RULES = ['average', 'sequential']

# Assumed widths (m), not measured
LANE_WIDTH = 3.0
CAR_FREE_WIDTH = 4.6  # widest BSL class
BIKE_LANE_WIDTH = 1.5
PATH_WIDTH = 2.2

SURFACE = pd.DataFrame({
    'hcm': {'asphalt': 5, 'asphalt_colored': 5,
            'asphalt_cracks_present': 4, 'gravel': 3, 'tiles': 2},
    'weikl': {'asphalt': 5, 'asphalt_colored': 5,
              'asphalt_cracks_present': 4, 'gravel': 2, 'tiles': 3},
    'netascore': {'asphalt': 1, 'asphalt_colored': 1,
                  'asphalt_cracks_present': 1, 'gravel': 0.75, 'tiles': 0.75},
})
NETASCORE_INFRA = {'shared_path': 0.9, 'advisory': 0.75, 'no_bike_infra': 0}


def cut(values, bins, **kwargs):
    """Bin index (0, 1, ...) of each value."""
    return pd.cut(values, [-np.inf, *bins, np.inf], labels=False, **kwargs)


def add_context(clips, cfg):
    """Truck volume, road noise, land-use zone, cycle route and NO2 per clip."""
    buf = P.calculate_buffer(clips, BUFFER)
    merge = P.merge_spatial_attribute

    trucks = ['AADT_truck_veh', 'AADT_articulated_truck_veh']
    traffic = gpd.read_file(cfg['traffic_volume_file'])
    clips = merge(buf, clips, traffic, trucks)
    clips['heavy_vehicles'] = clips[trucks].sum(axis=1)

    noise = gpd.read_file(cfg['noise_file'], layer=cfg['noise_layer'])
    clips = merge(buf, clips, noise, 'lret', 'noise_db')

    zones = gpd.read_file(cfg['land_use_file'])
    clips = merge(buf, clips, zones, 'typ', 'zone')

    routes = gpd.read_file(cfg['bike_network_file'],
                           layer=cfg['bike_network_layer'])
    clips = merge(buf, clips, routes, 'velo', 'designated_route')

    air = gpd.read_file(cfg['air_quality_file']).query('jahr == 2021')
    nearest = gpd.sjoin_nearest(clips[['geometry']], air[['no2', 'geometry']])
    clips['no2'] = nearest['no2'].groupby(level=0).mean()
    return clips


def prepare_inputs(clips):
    """Shared inputs. Shared paths and car-free streets have no motor traffic."""
    infra = clips['bike_infra_type']
    d = pd.DataFrame(index=clips.index)
    d['separated'] = infra.eq('shared_path')
    d['lane'] = infra.eq('advisory')
    d['facility'] = d['separated'] | d['lane']
    d['on_road'] = clips['car_lanes_total_count'].gt(0) & ~d['separated']

    on_road = d['on_road']
    d['speed'] = clips['motorized_traffic_speed_kmh'].where(on_road, 0)
    d['aadt'] = clips['traffic_volume'].where(on_road, 0)
    d['lanes'] = clips['car_lanes_total_count'].where(on_road, 0)
    one_way = clips['one_way'].astype(bool)
    lanes_dir = np.where(one_way, d['lanes'], d['lanes'] / 2)
    d['lanes_dir'] = lanes_dir.clip(min=1)
    d['parking'] = clips['side_parking_presence'].astype(bool) & on_road
    d['curb_width'] = np.where(on_road, LANE_WIDTH, CAR_FREE_WIDTH)
    d['bike_width'] = np.where(d['facility'], BIKE_LANE_WIDTH, 0.0)

    # Peak hour = 10% of AADT, split by direction
    split = np.where(one_way, 1, 0.5)
    d['hour_dir'] = d['aadt'] * 0.1 * split
    d['curb_volume'] = d['hour_dir'] / d['lanes_dir']
    trucks = clips['heavy_vehicles'].fillna(0).where(on_road, 0)
    d['curb_trucks'] = trucks * 0.1 * split / d['lanes_dir']
    share = trucks / d['aadt'].replace(0, np.nan)
    d['heavy_share'] = share.fillna(0).clip(upper=0.5)
    d['residential'] = clips['zone'].fillna('').str.startswith('W')

    # Missing road noise: median on roads, quiet (< 60 dB) off roads
    road_median = clips.loc[on_road, 'noise_db'].median()
    fill = pd.Series(np.where(on_road, road_median, 55), index=clips.index)
    d['noise'] = clips['noise_db'].fillna(fill)

    for method in SURFACE:
        d[f'pavement_{method}'] = clips['surface_type'].map(SURFACE[method])
    return d


def calc_lts(d):
    """Level of Traffic Stress (Furth et al. 2016, Tables 2-3), worst factor."""
    mph = cut(d['speed'], [40, 50, 60]).to_numpy()  # <= 25, 30, 35, 40+ mph
    lanes = np.ceil(d['lanes_dir']).to_numpy()
    lanes_lts = np.where(lanes <= 1, 1, 3)

    # Bike lane with parking: reach 1.5 + 2 m is LTS 3 (LTS 2 at <= 25 mph)
    reach = np.where(mph == 0, 2, 3)
    lane_park = np.maximum.reduce([lanes_lts, reach, mph + 1])
    # Bike lane without parking: reach 1.5 m is LTS 2
    speed_free = np.select([mph <= 1, mph == 2], [1, 3], 4)
    lane_free = np.maximum.reduce([lanes_lts, np.full(len(d), 2), speed_free])

    # Mixed traffic: no centreline if ADT <= 3000 and one lane per direction
    shared = (d['aadt'].to_numpy() <= 3000) & (lanes <= 1)
    mixed = np.select(
        [shared, lanes <= 1, lanes == 2],
        [np.choose(mph, [1, 2, 4, 4]),
         np.choose(mph, [2, 3, 4, 4]),
         np.choose(mph, [3, 4, 4, 4])],
        4,
    )
    on_lane = np.where(d['parking'], lane_park, lane_free)
    lts = np.where(d['lane'], on_lane, mixed)
    return pd.Series(np.where(d['on_road'], lts, 1), index=d.index)


def calc_bsl(d):
    """Bicycle Stress Level (Sorton & Walsh 1994): mean of three points."""
    width_m = d['curb_width'] - np.where(d['parking'], 2.4, 0)
    volume = cut(d['curb_volume'], [50, 150, 250, 350]) + 1
    width = 5 - cut(width_m, [3.7, 4.0, 4.3, 4.6], right=False)
    speed = cut(d['speed'], [40, 50, 60, 65]) + 1
    return (volume + width + speed) / 3


def calc_bci(d):
    """Bicycle Compatibility Index (Harkey et al. 1998, metric)."""
    f_truck = cut(d['curb_trucks'], [10, 20, 30, 60, 120], right=False) / 10
    return (3.67
            - 0.966 * d['facility']
            - 0.410 * d['bike_width']
            - 0.498 * d['curb_width']
            + 0.002 * d['curb_volume']
            + 0.0004 * (d['hour_dir'] - d['curb_volume'])
            + 0.022 * d['speed']
            + 0.506 * d['parking']
            - 0.264 * d['residential']
            + f_truck)


def calc_blos(d):
    """Bicycle Level of Service (HCM link model)."""
    v15 = (d['aadt'] * 0.1 * 0.565 / 4).clip(lower=4 * d['lanes_dir'])
    speed_mph = (d['speed'] * 0.621371).clip(lower=21)
    speed_factor = 1.1199 * np.log(speed_mph - 20) + 0.8103
    width_ft = (d['curb_width'] + d['bike_width']) * 3.28084
    return (0.507 * np.log(v15 / d['lanes_dir'])
            + 0.199 * speed_factor * (1 + 10.38 * d['heavy_share']) ** 2
            + 7.066 * (1 / d['pavement_hcm']) ** 2
            - 0.005 * width_ft ** 2
            + 0.760)


def calc_weikl(d, clips):
    """
    Weikl & Mayer (2023), segment indicators (Table 7, Eqs. 4, 5, 8, 9), 1-5.
    Weights rescaled to the available indicators (no width, lighting,
    braking or bicycle parking).
    """
    speed = 5 - cut((d['speed'] - 16.9).abs(), [5, 15, 25, 35])
    speed = speed.where(d['on_road'], 5)
    mixed = 4 - cut(d['aadt'], [500, 2500, 5000], right=False)
    guidance = np.select([~d['on_road'], d['lane']], [5, 3], mixed)
    stationary = np.where(d['parking'], 2, 5)  # parked cars 0.75-1.5 m away
    collision = np.minimum(guidance, stationary)
    conflict = 5  # linear sections, no intersections

    rise = clips['slope'].clip(lower=0) * clips['length']
    severity = rise ** 2 / clips['length']
    slope = 5 - cut(severity, [0.033, 0.075, 0.2, 0.4], right=False)
    green = cut(clips['greenery_share_gis'], [0, 25, 50, 75]) + 1
    noise = 5 - cut(d['noise'], [60, 65, 70, 75])
    air = 5 - cut(clips['no2'], [20, 35, 50, 100])

    safety = (0.11 * speed + 0.23 * collision + 0.26 * conflict) / 0.60
    comfort = (0.19 * slope + 0.26 * d['pavement_weikl']) / 0.45
    attract = 0.35 * green + 0.3 * noise + 0.35 * air
    return (0.3 * safety + 0.19 * comfort + 0.13 * attract) / 0.62


def calc_netascore(d, clips):
    """NetAScore default bike profile (Werner et al. 2024), 0-1."""
    s = d['speed']
    infra = clips['bike_infra_type'].map(NETASCORE_INFRA)
    # No motor traffic, service (<= 20), residential (30), secondary, primary
    road = np.select(
        [~d['on_road'], s <= 20, s <= 30, d['lanes'] < 4],
        [1, 0.85, 0.8, 0.2], 0,
    )
    route = np.where(clips['designated_route'] == 1, 0.8, 0)
    speed = np.select(
        [s >= 100, s >= 80, s >= 70, s >= 60, s >= 50, s >= 30, s > 0],
        [0, 0.2, 0.3, 0.4, 0.6, 0.85, 0.9], 1,
    )
    parking = np.where(d['parking'], 0, 1)

    grade = clips['slope'] * 100
    grade_class = np.sign(grade) * cut(grade.abs(), [1.5, 3, 6, 12], right=False)
    gradient = grade_class.map({4: 0, 3: 0.25, 2: 0.4, 1: 0.5, 0: 0.9,
                                -1: 1, -2: 0.95, -3: 0.35, -4: 0})
    pavement = d['pavement_netascore']
    steep_loose = (pavement < 1) & (grade_class.abs() >= 3)
    w = np.where(steep_loose, 1.6, 0.1)

    total = (0.2 * infra + 0.3 * road + 0.1 * route + 0.1 * speed
             + 0.1 * parking + w * pavement + w * gradient)
    return total / (0.8 + 2 * w)


def calc_danish_blos(d, clips):
    """Danish bicycle LOS (Jensen 2007): expected satisfaction, 1 to 6."""
    km = clips['length'] / 1000
    mot = d['aadt'] * 0.1
    speed = d['speed']
    ped = clips['unique_pedestrians_count'] / km * 20  # per hour at 20 km/h
    park = (clips['side_parking_count'] / km / 10).where(d['on_road'], 0)
    path = np.where(d['separated'], PATH_WIDTH, 0)
    ulan = np.where(d['lane'], BIKE_LANE_WIDTH, 0)
    sidewalk = (~d['separated']).astype(int)
    area = np.where(d['residential'], 0.0557, -0.0334)

    eta = (area
           - 0.0005585 * mot
           - 0.0958 * speed
           + 0.000421 * speed ** 2
           - 0.000002913 * mot * speed
           - 0.001623 * ped
           + 0.0000008309 * ped ** 2
           - 0.09416 * park
           + 1.7782 * path
           + 1.3938 * ulan
           + 0.2413 * d['curb_width']
           + 1.2694 * sidewalk
           + 0.6821 * (d['lanes'] >= 4))
    alphas = np.array([-1.3652, 0.3741, 1.5512, 2.4805, 3.8449])
    cum = 1 / (1 + np.exp(-(alphas[None, :] + eta.to_numpy()[:, None])))
    n = len(d)
    probs = np.diff(np.column_stack([np.zeros(n), cum, np.ones(n)]), axis=1)
    return pd.Series(probs @ np.arange(6, 0, -1), index=d.index)


def calc_indices(clips):
    """All indices per clip, higher = more bikeable (stress scales reversed)."""
    d = prepare_inputs(clips)
    return pd.DataFrame({
        'video_id': clips['video_id'],
        'lts': 5 - calc_lts(d),
        'bsl': 6 - calc_bsl(d),
        'bci': 6 - calc_bci(d),
        'blos': 6 - calc_blos(d),
        'weikl': calc_weikl(d, clips),
        'netascore': calc_netascore(d, clips),
        'danish_blos': calc_danish_blos(d, clips),
    })


def route_scores(routes, indices, weights):
    """Route value of every index: simple average and position-weighted."""
    lookup = indices.set_index('video_id')
    keep = ['participant_id', 'block', 'sequence', 'off_type_position']
    rows = []
    for index in INDICES:
        s = np.column_stack([
            routes[f'pos{k}_video_id'].map(lookup[index]) for k in (1, 2, 3)
        ])
        rules = {'average': s.mean(axis=1), 'sequential': s @ weights}
        for rule, score in rules.items():
            rows.append(routes[keep].assign(index=index, rule=rule, score=score))
    return pd.concat(rows, ignore_index=True)


def sequence_table(scores, indices):
    """Mean route value per sequence and rule, vs baseline route (% of range)."""
    key = ['index', 'block', 'off_type_position', 'sequence', 'rule']
    out = scores.groupby(key)['score'].mean().rename('mean').reset_index()

    group = ['index', 'block', 'rule']
    baseline = out[out['off_type_position'] == 0].set_index(group)['mean']
    base = pd.MultiIndex.from_frame(out[group]).map(baseline)
    span = indices[INDICES].max() - indices[INDICES].min()
    out['vs_baseline_pct'] = (out['mean'] - base) / out['index'].map(span) * 100
    return out


def order_effect(seq):
    """Off-type segment last minus first under Sequential, net of averaging."""
    ends = seq[seq['off_type_position'].isin([1, 3])]
    wide = ends.pivot_table(index=['index', 'block', 'rule'],
                            columns='off_type_position', values='vs_baseline_pct')
    effect = (wide[3] - wide[1]).unstack('rule')
    return (effect['sequential'] - effect['average']).unstack('block')
