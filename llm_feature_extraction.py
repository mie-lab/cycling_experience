import json
import logging
from datetime import time
from pathlib import Path
import re
import pandas as pd
import configparser
import constants as c
from utils.segmentation_utils import get_differences, calculate_aggregate_metrics
from pydantic import BaseModel
from google import genai

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


class VideoFeatures(BaseModel):
    surface_material: c.SURFACE_MATERIAL
    car_lanes_total_count: int
    one_way: bool
    bike_lane_type: c.BIKE_LANE_TYPE
    bike_lane_presence: bool
    bike_lane_width_estimate_meters: float
    side_parking_presence: bool
    tram_lane_presence: bool
    bus_lane_presence: bool
    unique_motor_vehicles_count: int
    motorized_traffic_speed_kmh: float
    car_overtakes_count: int
    car_overtakes_presence: bool
    unique_cyclists_count: int
    unique_pedestrians_count: int
    average_greenery_share: float
    average_building_share: float
    average_sky_share: float
    average_road_share: float


def get_llm_extracted_features(config: configparser.ConfigParser) -> dict:
    try:
        api_key = Path(config["models"]["gemini_api_key"]).read_text().strip()
        video_candidates_path = Path(config["paths"]["video_candidates_path"])
        prompt_file = Path(config["filenames"]["prompt_file"])
        client = genai.Client(api_key=api_key)
        genai_config = genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=0,
            ),
            seed=1233,
            candidate_count=1,
            top_k=1,
            top_p=0.,
            temperature=0.,
            response_mime_type='application/json',
            response_schema=VideoFeatures,
        )
    except KeyError as e:
        log.error(f"Missing configuration key: {e}. Please check your config.ini file.")
        return {}

    try:
        prompt_text = prompt_file.read_text()
        log.info(f"Successfully loaded prompt from {prompt_file.name}")
    except FileNotFoundError:
        log.error(f"Prompt file not found at: {prompt_file}")
        return {}

    video_files = list(video_candidates_path.glob('*.mp4')) + list(video_candidates_path.glob('*.mov'))
    log.info(f"Found {len(video_files)} videos to process in {video_candidates_path}.")
    all_features = {}

    for video_file in video_files:
        log.info(f"--- Processing video: {video_file.name} ---")
        try:
            log.info("Reading video file to memory...")
            with open(video_file, 'rb') as f:
                video_file_data = f.read()

            log.info("Sending request to Gemini 2.5 Flash API...")
            video_part = genai.types.Part(
                inline_data=genai.types.Blob(data=video_file_data, mime_type='video/mp4')
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[video_part, genai.types.Part(text=prompt_text)],
                config=genai_config,
            )

            all_features[video_file.name] = response.parsed.model_dump()
            log.info(f"Successfully extracted features for {video_file.name}.")
        except Exception as e:
            log.error(f"An unexpected error occurred while processing {video_file.name}: {e}")

            # Add a small delay to respect API rate limits
            time.sleep(1)

    # TODO replace with a csv export
    #with open(Path(config["filenames"]["video_info_llm_extraction"]), "w") as f:
    #    json.dump(all_features, f, indent=2)

    # --- CSV EXPORT LOGIC ---
    if not all_features:
        log.warning("No features extracted. Skipping CSV export.")
        return {}

    log.info("Converting extracted features to DataFrame...")

    # 1. Create DataFrame (orient='index' uses filenames as rows)
    df_llm = pd.DataFrame.from_dict(all_features, orient='index')
    df_llm = df_llm.reset_index().rename(columns={'index': 'source_filename'})

    try:
        df_llm['video_name'] = df_llm['source_filename'].str.extract(r'(video_\d+)')
        cols = ['video_name', 'source_filename'] + [c for c in df_llm.columns if
                                                    c not in ['video_name', 'source_filename']]
        df_llm = df_llm[cols]
    except Exception as e:
        log.warning(f"Could not extract clean video_name (e.g., 'video_54') from filenames: {e}")

    output_path = Path(config["filenames"]["video_info_llm_extraction"]).with_suffix(".csv")

    df_llm.to_csv(output_path, index=False)
    log.info(f"Successfully exported LLM features to CSV at: {output_path}")

    return all_features


def run_ground_truth_comparison(config: configparser.ConfigParser) -> None:
    """
    Compares LLM-extracted features against the ground truth CSV file.
    """
    log.info("--- Comparing LLM Predictions with Ground Truth ---")
    llm_extractions_path = Path(config["filenames"]["video_info_llm_extraction"])
    ground_truth_path = Path(config['filenames']['video_info_ground_truth'])

    try:
        with open(llm_extractions_path, 'r') as f:
            predictions_dict = json.load(f)

        predictions_dict = {
            int(re.search(r'_video_(\d+)', key).group(1)): value
            for key, value in predictions_dict.items()
            if re.search(r'_video_(\d+)', key)
        }
        log.info(f"Successfully transformed {len(predictions_dict)} prediction keys.")

        ground_truth_df = pd.read_csv(ground_truth_path)
        ground_truth_dict = ground_truth_df.set_index(c.VIDEO_ID_COL).to_dict(orient='index')

        detailed_results, common_videos, numerical_fields, _ = get_differences(predictions_dict, ground_truth_dict)

        if detailed_results:
            summary_metrics = calculate_aggregate_metrics(detailed_results, common_videos, numerical_fields)
            summary_df = pd.DataFrame.from_dict(summary_metrics, orient='index')
            print("\n--- Aggregate Comparison Metrics ---")
            print(summary_df)
            print("------------------------------------\n")
        else:
            log.warning("No comparison could be made as no common videos were found.")

    except FileNotFoundError as e:
        log.error(f"Could not find a required file. Check paths in config.ini. Details: {e}")
    except json.JSONDecodeError as e:
        log.error(f"Could not parse a JSON file. It might be malformed. Details: {e}")
    except KeyError as e:
        log.error(f"A required column is missing from the CSV, likely 'video_id'. Details: {e}")


def main():
    """
    Main function to load configuration and run the comparison.
    """
    log.info("Loading configuration from config.ini...")
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("config.ini")

    # Define the path for the LLM output file from your config
    llm_output_file = Path(config["filenames"]["video_info_llm_extraction"])

    if llm_output_file.exists():
        log.info(f"LLM features file found at '{llm_output_file}'. Skipping extraction.")
    else:
        log.info(f"LLM features file not found. Running feature extraction...")
        get_llm_extracted_features(config)

    run_ground_truth_comparison(config)


if __name__ == "__main__":
    main()