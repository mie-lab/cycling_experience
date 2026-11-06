"""
Zero-shot MLLM feature extraction from the cycling videos (Gemini 2.5 Flash).

The prompt (input_data/video_data/prompt.txt) and the response schema below request the
same fields under the same names. The output is a JSON file keyed by video filename,
read by build_ground_truth.py and mllm_extraction_figures.py (validation, Table A1),
plus a CSV copy for inspection.

Run with:  python llm_feature_extraction.py
Resumable: the JSON is saved after every video and a run stops when the daily API quota
is reached, so the next run continues with the videos still missing. A video counts as
done only when its entry has every field of the schema.
"""
import configparser
import json
import logging
import time
from pathlib import Path

import pandas as pd
from google import genai
from pydantic import BaseModel

import constants as c

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)

MODEL = "gemini-2.5-flash"
VIDEO_SUFFIXES = {".mp4", ".mov"}


class VideoFeatures(BaseModel):
    """Response schema; field names and order match prompt.txt."""
    surface_material: c.SURFACE_MATERIAL
    car_lanes_total_count: int
    one_way: bool
    bike_infra_type: c.BIKE_LANE_TYPE
    bike_infra_presence: bool
    bike_infra_width_estimate_meters: float | None
    side_parking_presence: bool
    tram_lane_presence: bool
    bus_lane_presence: bool
    unique_motor_vehicles_count: int
    buses_passing_count: int
    trams_passing_count: int
    trucks_passing_count: int
    motorized_traffic_speed_kmh: float
    motor_vehicle_overtakes_count: int
    motor_vehicle_overtakes_presence: bool
    unique_cyclists_count: int
    unique_pedestrians_count: int
    average_greenery_share: float
    average_building_share: float
    average_sky_share: float
    average_road_share: float


def save_features(features, path):
    """Write the JSON in one step, so an interrupted run never leaves a broken file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)
    tmp.replace(path)


def is_quota_error(error):
    """Daily or per-minute request limit reached (HTTP 429)."""
    return getattr(error, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(error)


def extract_features(client, prompt, video_files, features, output_file, seconds_between=1.0):
    """
    One deterministic request per video, added to `features` and saved after each video.
    Stops at the API quota; other failures are logged and the video is retried next run.
    """
    generation = genai.types.GenerateContentConfig(
        thinking_config=genai.types.ThinkingConfig(include_thoughts=False, thinking_budget=1024),  # helps counting
        seed=1233,
        candidate_count=1,
        top_k=1,
        top_p=0.0,
        temperature=0.0,
        response_mime_type="application/json",
        response_schema=VideoFeatures,
    )

    for i, video in enumerate(video_files, 1):
        log.info(f"[{i}/{len(video_files)}] {video.name}")
        try:
            part = genai.types.Part(inline_data=genai.types.Blob(data=video.read_bytes(), mime_type="video/mp4"))
            response = client.models.generate_content(
                model=MODEL, contents=[part, genai.types.Part(text=prompt)], config=generation
            )
            features[video.name] = response.parsed.model_dump()
            save_features(features, output_file)
        except Exception as e:
            if is_quota_error(e):
                log.warning("API quota reached. Run again later to continue.")
                break
            log.error(f"Failed on {video.name}: {e}")
        time.sleep(seconds_between)  # API rate limit
    return features


def main():
    """Extract the MLLM features for all candidate videos and save them."""
    # ==============================================================================
    # PHASE 0: SETUP & CONFIGURATION
    # ==============================================================================
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("config.ini")

    output_file = Path(config["filenames"]["video_info_llm_extraction"])
    prompt_file = Path(config["filenames"]["prompt_file"])
    video_dir = Path(config["paths"]["video_candidates_path"])

    videos = sorted(p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)

    # Resume: keep complete entries from earlier runs, redo the rest
    features = {}
    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            features = json.load(f)
    fields = set(VideoFeatures.model_fields)
    todo = [v for v in videos if not fields <= set(features.get(v.name, {}))]
    log.info(f"{len(videos) - len(todo)} of {len(videos)} videos done, {len(todo)} to go")
    if not todo:
        return

    # ==============================================================================
    # PHASE 1: EXTRACTION
    # ==============================================================================
    client = genai.Client(api_key=Path(config["models"]["gemini_api_key"]).read_text().strip())
    prompt = prompt_file.read_text(encoding="utf-8")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features = extract_features(client, prompt, todo, features, output_file)

    # ==============================================================================
    # PHASE 2: CSV COPY
    # ==============================================================================
    table = pd.DataFrame.from_dict(features, orient="index").rename_axis("source_filename").reset_index()
    table.insert(0, c.VIDEO_ID_COL, table["source_filename"].str.extract(r"_video_(\d+)", expand=False).astype(int))
    table.to_csv(output_file.with_suffix(".csv"), index=False)

    left = sum(not fields <= set(features.get(v.name, {})) for v in videos)
    log.info(f"Saved {output_file.name} (+ .csv): {len(videos) - left} of {len(videos)} videos done, {left} left")


if __name__ == "__main__":
    main()
