# build_ground_truth.py
import pandas as pd
import configparser
from pathlib import Path
import logging
import constants as c
import utils.processing_utils
from utils.segmentation_utils import extract_frames_from_videos, run_semantic_segmentation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    """
    Generates a complete ground truth dataset for all videos.
    This is the single source of truth for video features.
    """
    # ==============================================================================
    # 0. SETUP
    # ==============================================================================
    log.info("Loading configuration...")
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read("config.ini")

    video_geom_dir = Path(config["paths"]["video_geom_dir"])
    video_dir = Path(config["paths"]["video_candidates_path"])
    segmentation_results_file = Path(config["filenames"]["segmentation_results_file"])
    output_file = Path(config['filenames']['video_info_ground_truth'])

    # ==============================================================================
    # 1. GENERATE GEOSPATIAL FEATURES
    # ==============================================================================
    log.info("Phase 1: Aggregating and enriching geospatial data for all videos.")
    gpx_paths = [video_geom_dir / f for f in video_geom_dir.iterdir() if f.suffix.lower() in ['.gpkg', '.gpx']]

    video_geom = utils.processing_utils.aggregate_video_level_geometry(gpx_paths)
    video_geom = utils.processing_utils.enrich_with_spatial_data(video_geom, config)

    # ==============================================================================
    # 2. GENERATE SEMANTIC FEATURES
    # ==============================================================================
    if not any(d.is_dir() for d in video_dir.iterdir()):
        log.info("No frame folders found. Extracting frames from videos...")
        extract_frames_from_videos(video_dir, interval_seconds=1)
    else:
        log.info("Frame folders already exist. Skipping frame extraction.")

    # The pipeline is only run if the final results file is missing.
    if not segmentation_results_file.exists():
        log.info(f"Segmentation results file not found. Running the pipeline...")
        run_semantic_segmentation(config)
    else:
        log.info(f"Segmentation results file found. Skipping segmentation.")

    # Now load the results, which are guaranteed to exist at this point.
    log.info("Loading and merging semantic segmentation features.")
    segmentation_df = pd.read_csv(segmentation_results_file)
    segmentation_df[c.VIDEO_ID_COL] = segmentation_df[c.VIDEO_ID_COL].str.extract(r'_video_(\d+)').astype(int)
    video_geom = pd.merge(video_geom, segmentation_df, on=c.VIDEO_ID_COL, how='left')

    # ==============================================================================
    # 3. SAVE THE MASTER GROUND TRUTH FILE
    # ==============================================================================
    log.info(f"Saving the complete ground truth data to {output_file}")
    video_geom.drop(columns=['geometry']).to_csv(output_file, index=False)
    log.info("Ground truth generation complete!")


if __name__ == "__main__":
    main()