import os
import logging
from collections import defaultdict
from pathlib import Path
from transformers import pipeline
import cv2
import numpy as np
import pandas as pd
from moviepy.editor import VideoFileClip
from PIL import Image

import constants
from utils.plotting_utils import plot_segmentation_overlay

#--- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def extract_frames_from_videos(
        source_folder: Path,
        interval_seconds=1
) -> None:
    """
    Extract frames from all video files in the specified folder at regular intervals.
    :param source_folder: Path to the folder containing video files.
    :param interval_seconds: Interval in seconds at which to extract frames.
    :return: None
    """
    if not os.path.isdir(source_folder):
        log.error(f"Source folder not found at '{source_folder}'")
        return

    log.info(f"Scanning for video files in '{source_folder}'...")
    supported_formats = ('.mp4', '.mov', '.avi', '.mkv')

    for filename in os.listdir(source_folder):
        file_path = os.path.join(source_folder, filename)
        if filename.lower().endswith(supported_formats) and os.path.isfile(file_path):
            input_video_path = file_path

            log.info(f"\n--- Processing video: {filename} ---")
            video_name_without_ext = os.path.splitext(filename)[0]
            video_specific_output_folder = os.path.join(source_folder, video_name_without_ext)

            if os.path.exists(video_specific_output_folder):
                log.info(f"Output folder '{video_specific_output_folder}' already exists. Skipping.")
                continue  # Move to the next file

            if not os.path.exists(video_specific_output_folder):
                os.makedirs(video_specific_output_folder)
                log.info(f"Created sub-directory: {video_specific_output_folder}")

            saved_frame_count = 0
            clip = None
            try:
                clip = VideoFileClip(input_video_path)
                video_duration_sec = clip.duration
                log.info(f"Video Info: {video_duration_sec:.2f} seconds long.")

                for current_time_sec in np.arange(0, video_duration_sec, interval_seconds):
                    frame_rgb = clip.get_frame(current_time_sec)
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    output_filename = os.path.join(video_specific_output_folder,
                                                   f"frame_at_{current_time_sec:.2f}_sec.jpg")
                    cv2.imwrite(output_filename, frame_bgr)
                    saved_frame_count += 1
                log.info(f"Successfully saved {saved_frame_count} frames to '{video_specific_output_folder}'.")
            except Exception as e:
                log.error(f"An error occurred while processing {filename}: {e}")
            finally:
                if clip:
                    clip.close()
    log.info("\n--- Batch frame extraction complete. ---")


def calculate_label_ratio(
        image: Image,
        results: list,
        labels: list
) -> float:
    """
    Calculate the ratio of pixels belonging to a set of target labels.
    :param image: PIL Image object.
    :param results: List of segmentation results from the model.
    :param labels: List of labels to be measured (e.g., ['vegetation', 'terrain']).
    :return: Percentage of pixels in the image covered by the target labels.
    """
    label_pixel_count = 0
    total_pixels = image.width * image.height
    for result in results:
        if result['label'] in labels:
            mask = result['mask']
            label_pixel_count += np.count_nonzero(np.array(mask))
    return (label_pixel_count / total_pixels) * 100 if total_pixels > 0 else 0


def run_semantic_segmentation(
        config: dict
) -> pd.DataFrame:
    """
    Wrapper function to run semantic segmentation for multiple label groups.
    :param config: Configuration parser object with paths and settings.
    """
    log.info("--- STAGE 2: Performing Semantic Segmentation ---")
    video_root_folder = Path(config["paths"]["video_candidates_path"])
    model_name = config["models"]["seg_former_model"]
    output_csv_path = Path(config["filenames"]["segmentation_results_file"])
    processed_videos = set()

    # Check for existing results (this logic remains the same)
    if output_csv_path.exists():
        try:
            df_existing = pd.read_csv(output_csv_path)
            if 'video_id' in df_existing.columns:
                processed_videos = set(df_existing['video_id'].astype(str))
                log.info(f"Loaded {len(processed_videos)} previously processed video IDs.")
        except Exception as e:
            log.error(f"Could not read existing results file. Will start fresh. Error: {e}")

    # Model loading
    log.info(f"Loading segmentation model: {model_name}...")
    try:
        semantic_segmentation = pipeline("image-segmentation", model=model_name)
    except Exception as e:
        log.error(f"Failed to load model. Error: {e}")
        return

    video_level_results = []
    folder_list = sorted([d for d in os.listdir(video_root_folder) if os.path.isdir(video_root_folder / d)])

    # The key is the column name, and the value is the list of Cityscapes labels.
    semantic_label_groups = {
        'greenery': ['vegetation', 'terrain'],
        'sky': ['sky'],
        'building': ['building'],
        'road': ['road']
    }
    # TODO: implement instance tracking and counting instead of manual labelling and counting.
    instance_label_groups = {}

    log.info(f"Target label groups to analyze: {list(semantic_label_groups.keys())}")
    for folder_name in folder_list:
        if folder_name in processed_videos:
            log.info(f"Result for '{folder_name}' found in CSV. Skipping.")
            continue

        folder_path = video_root_folder / folder_name
        log.info(f"Processing frames in: {folder_name}")

        # Initialize a dictionary to hold lists of ratios for each frame ---
        frame_ratios_by_group = {key: [] for key in semantic_label_groups.keys()}

        frame_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.jpg')])
        for i, frame_filename in enumerate(frame_files):
            cur_image_path = folder_path / frame_filename
            cur_image = Image.open(cur_image_path)
            cur_results = semantic_segmentation(cur_image)

            for group_name, labels in semantic_label_groups.items():
                ratio = calculate_label_ratio(cur_image, cur_results, labels)
                frame_ratios_by_group[group_name].append(ratio)

            # Sanity test save
            if i == 0:
                base_name = cur_image_path.stem
                output_filename = f"{base_name}_segmented.png"
                output_path = Path(cur_image_path.with_name(output_filename))
                log.info(f"Saving segmentation overlay for first frame of {folder_name}...")
                plot_segmentation_overlay(cur_image, cur_results, output_path=output_path)

        # Calculate average for each group and store in one dictionary ---
        video_result = {'video_id': folder_name}
        all_ratios_valid = True
        log_message = f"--> Averages for {folder_name}: "

        for group_name, ratios in frame_ratios_by_group.items():
            if not ratios:
                all_ratios_valid = False
                break
            average_ratio = np.mean(ratios)
            video_result[f'average_{group_name}_share'] = average_ratio
            log_message += f"{group_name}: {average_ratio:.2f}% | "

        if all_ratios_valid:
            video_level_results.append(video_result)
            log.info(log_message.strip(" | "))

    df_new = pd.DataFrame(video_level_results)

    if 'df_existing' in locals() and not df_existing.empty:
        final_df = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=['video_id'], keep='last')
        log.info(f"Updated/appended {len(df_new)} results to the existing {len(df_existing)} results.")
    else:
        final_df = df_new
        log.info(f"Creating a new results file with {len(df_new)} entries.")

    final_df.to_csv(output_csv_path, index=False)
    log.info(f"Segmentation results saved to {output_csv_path}")

    return final_df


def get_differences(
        predictions: dict,
        ground_truth: dict
) -> tuple:
    """
    Calculate differences between prediction and ground truth dictionaries.
    :param predictions: Dictionary of predictions.
    :param ground_truth: Dictionary of ground truth values.
    :return: Tuple containing detailed results, list of common videos, numerical fields, and categorical/boolean fields.
    """
    gt_keys, pred_keys = set(ground_truth.keys()), set(predictions.keys())
    common_videos = sorted(list(gt_keys.intersection(pred_keys)))
    if not common_videos:
        log.warning("No common video files found between the two JSON files.")
        return {}, [], [], []
    log.info(f"Found {len(common_videos)} common videos to compare.")
    detailed_results = {}

    numerical_fields = constants.NUMERIC_FIELDS
    categorical_boolean_fields = constants.CATEGORICAL_BOOLEAN_FIELDS

    for video_key in common_videos:
        gt_video, pred_video = ground_truth[video_key], predictions[video_key]
        video_comparison = {}
        for field, gt_value in gt_video.items():
            pred_value = pred_video.get(field, 'N/A')
            comparison_details = {"ground_truth": gt_value, "prediction": pred_value}
            if gt_value is None or pred_value is None:
                comparison_details["match"] = (gt_value == pred_value)
            elif field in numerical_fields:
                try:
                    error = abs(float(gt_value) - float(pred_value))
                    comparison_details["absolute_error"] = round(error, 2)
                except (ValueError, TypeError):
                    comparison_details["absolute_error"] = "N/A (type mismatch)"
            elif field in categorical_boolean_fields:
                comparison_details["match"] = (gt_value == pred_value)
            video_comparison[field] = comparison_details
        detailed_results[video_key] = video_comparison
    return detailed_results, common_videos, numerical_fields, categorical_boolean_fields


def calculate_aggregate_metrics(
        detailed_results: dict,
        common_videos: list,
        numerical_fields: list
) -> dict:
    """
    Calculate aggregate metrics from detailed comparison results.
    :param detailed_results: Dictionary of detailed comparison results.
    :param common_videos: List of common video keys.
    :param numerical_fields: List of fields considered numerical.
    :return: Dictionary of aggregate metrics.
    """
    field_metrics = defaultdict(list)
    for video_key in common_videos:
        for field, comparison in detailed_results[video_key].items():
            if "absolute_error" in comparison and isinstance(comparison["absolute_error"], (int, float)):
                field_metrics[field].append(comparison["absolute_error"])
            elif "match" in comparison:
                field_metrics[field].append(1 if comparison["match"] else 0)
    aggregate_summary = {}
    for field, values in field_metrics.items():
        if not values: continue
        if field in numerical_fields:
            mae = sum(values) / len(values)
            aggregate_summary[field] = {"mean_absolute_error": round(mae, 2)}
        else:
            accuracy = (sum(values) / len(values)) * 100
            aggregate_summary[field] = {"accuracy_percent": round(accuracy, 2)}
    return aggregate_summary
