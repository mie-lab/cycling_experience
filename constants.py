from typing import Literal

PROJ_CRS = 2056
BUFFER = 15

# Column names
VALENCE = 'valence'
AROUSAL = 'arousal'
PARTICIPANT_ID = 'participant_id'
VIDEO_ID_COL = 'video_id'
OE = 'OE'
AG = 'AG'
PF = 'PF'
NF = 'NF'
F = 'F'
START = 'start'
END = 'end'
CONSENT = 'Consent'
COUNTRY = 'Country'
GENDER = 'Gender'
AGE = 'Age'
CYCL_FREQ = 'Cycling_frequency'
CYCL_CONF = 'Cycling_confidence'
CYCL_PURP = 'Cycling_purpose'
CYCL_ENV = 'Cycling_environment'

# Levels and their orderings
GENDER_ORDER = [
    'Female',
    'Male'
    ]

# aggregated categories to ensure sufficient sample size in each group
AGE_ORDER = [
    "18 - 25 years",
    "26 - 35 years",
    "36 - 45 years",
    "46+ years"  # combines '46 - 55 years', '56 - 65 years', '+65 years'
    ]

# aggregated categories to ensure sufficient sample size in each group
CYCL_FREQ_ORDER = [
    'Infrequent',  # combines "Never", "Less than once a month",
    'Occasional',  # combines "1-3 times/month", "1-2 days/week"
    'Regular',  # combines "3-4 days/week", "5-6 days/week", "Every day"
    ]

CYCL_PURP_ORDER = [
    'Commuting (e.g., work, school)',
    'Recreational / leisure',
    'Exercise / fitness',
    'All purposes',
    'I do not cycle'
    ]

CYCL_ENV_ORDER = [
    'Urban area',
    'Rural area',
    'Both',
    'I do not cycle'
    ]

# aggregated categories to ensure sufficient sample size in each group
CYCL_CONF_ORDER = [
    'Confident',  # combines 'Very confident' and 'Somewhat confident'
    'Not confident'  # combines 'Slightly not confident', 'Not confident at all'
    ]

FAM_ORDER = [
    'Not at all familiar',
    'Somewhat unfamiliar',
    'Equally familiar and unfamiliar',
    'Somewhat familiar',
    'Extremely familiar'
    ]

OE_ORDER = [
    'Very negative',
    'Negative',
    'Somewhat negative',
    'Neutral',
    'Somewhat positive',
    'Positive',
    'Very positive'
    ]

AFFECTIVE_STATES = ['Tension', 'Activation', 'Contentment', 'Deactivation']

DEPENDENT_VARIABLES = (VALENCE, AROUSAL)

# We focus on valence as the primary target variable
TARGET_COL = VALENCE

DEMOGRAPHIC_COLUMNS = [GENDER, AGE, CYCL_FREQ, CYCL_PURP, CYCL_CONF, CYCL_ENV]

CATEGORY_ORDERS = {
    GENDER: GENDER_ORDER,
    AGE: AGE_ORDER,
    CYCL_FREQ: CYCL_FREQ_ORDER,
    CYCL_PURP: CYCL_PURP_ORDER,
    CYCL_ENV: CYCL_ENV_ORDER,
    CYCL_CONF: CYCL_CONF_ORDER,
    F: FAM_ORDER,
    OE: OE_ORDER}

LABEL_COLS = [
    'PF: Car overtakes',
    'PF: Cycling surface quality',
    'PF: Pedestrians on a cycling way',
    'PF: Pedestrians on a separate sidewalk',
    'PF: Slope', 'PF: Speed of traffic',
    'PF: Surrounding buildings',
    'PF: Surrounding greenery',
    'PF: Traffic volume',
    'PF: Type of bike infrastructure',
    'PF: Weather conditions',
    'NF: Car overtakes',
    'NF: Cycling surface quality',
    'NF: Pedestrians on a cycling way',
    'NF: Pedestrians on a separate sidewalk',
    'NF: Slope', 'NF: Speed of traffic',
    'NF: Surrounding buildings',
    'NF: Surrounding greenery',
    'NF: Traffic volume',
    'NF: Type of bike infrastructure',
    'NF: Weather conditions'
]

DATA_COLS = [
    "bike_infra_type_numeric",
    "motorized_traffic_speed_kmh",
    "traffic_volume",
    "average_greenery_share",
    "motor_vehicle_overtakes_presence"
]
NUMERIC_FIELDS = [
    "car_lanes_total_count",
    "bike_infra_width_estimate_meters",
    "unique_motor_vehicles_count",
    "motorized_traffic_speed_kmh",
    "motor_vehicle_overtakes_count",
    "unique_cyclists_count",
    "unique_pedestrians_count",
    'average_greenery_share'
]
CATEGORICAL_BOOLEAN_FIELDS = [
    "surface_material",
    "one_way",
    "bike_infra_type",
    "bike_infra_presence",
    "side_parking_presence",
    "tram_lane_presence",
    "motor_vehicle_overtakes_presence"
]

SURFACE_MATERIAL = Literal[
    "asphalt", "concrete", "gravel", "tiles", "cobblestone", "dirt", "mixed"
]
BIKE_LANE_TYPE = Literal[
    "shared_path", "advisory", "no_bike_infra"
]

# Trial-specific constants
VIDEO_COUNTS = {
    'DJI': 10,
    '2': 8,
    '3': 8,
    '4': 8
}

TRIAL_1 = 'DJI'

TRIAL_2_PARAMS = {
    'trial_label': '2',
    'video_mapping': {
        '2_1': ['B', 'B'],
        '2_2': ['B', 'NB'],
        '2_3': ['NB', 'B'],
        '2_4': ['NB', 'NB']
    },
    'outlier_position': {
        0: 'no NB',
        1: 'NB last',
        2: 'NB first',
        3: 'NB-NB'
    }
}

TRIAL_3_PARAMS = {
    'trial_label': '3',
    'video_mapping': {
        '3_1': ['B', 'B', 'B'],
        '3_2': ['B', 'B', 'NB'],
        '3_3': ['B', 'NB', 'B'],
        '3_4': ['NB', 'B', 'B']
    },
    'outlier_position': {
        0: 'no NB',
        1: 'NB last',
        2: 'NB first',
        3: 'NB-NB'
    }
}

TRIAL_4_PARAMS = {
    'trial_label': '4',
    'video_mapping': {
        '4_1': ['NB', 'NB', 'NB'],
        '4_2': ['NB', 'NB', 'B'],
        '4_3': ['NB', 'B', 'NB'],
        '4_4': ['B', 'NB', 'NB']
    },
    'outlier_position': {
        0: 'no B',
        1: 'B last',
        2: 'B middle',
        3: 'B first'
    }
}

TRIAL_2_PLOT_ORDER = [
    'B → B',
    'NB → B',
    'B → NB',
    'NB → NB'
]

TRIAL_3_PLOT_ORDER = [
    'B → B → B',  # No outlier
    'NB → B → B',  # Outlier 'NB' at position 1
    'B → NB → B',  # Outlier 'NB' at position 2
    'B → B → NB'  # Outlier 'NB' at position 3
]

TRIAL_4_PLOT_ORDER = [
    'NB → NB → NB',  # No outlier
    'B → NB → NB',  # Outlier 'B' at position 1
    'NB → B → NB',  # Outlier 'B' at position 2
    'NB → NB → B'  # Outlier 'B' at position 3
]

