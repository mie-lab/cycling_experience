# Analysis of Cyclists' Affective Responses to Environmental Factors

This project investigates how objective environmental factors—derived from both geospatial data and video content—influence a cyclist's subjective emotional experience (affective response). The analysis pipeline integrates data from a large-scale online survey, a computer vision feature extraction module, a predictive modeling component, and a controlled lab study to build a comprehensive understanding of what makes a cycling route feel safe, pleasant, or stressful.

## 📊 Project Workflow

The project is structured as a multi-stage pipeline where the output of one stage serves as the input for the next:

1.  **Online Survey Analysis**: Raw survey data is processed to quantify cyclists' affective responses (valence and arousal) for a set of videos. This provides the "ground truth" for perceived bikeability.
2.  **Objective Feature Engineering**:
    * **Geospatial Analysis**: Route data (GPX traces) is enriched with contextual layers like traffic volume, green space proximity, infrastructure type, and points of interest (POIs).
    * **Video Content Analysis**: A computer vision model performs semantic segmentation on video frames to automatically calculate features like the percentage of greenery.
3.  **Predictive Modeling & Candidate Selection**: A machine learning model is trained to predict the perceived valence of a route based on its objective features. This model is then used to score and select new "candidate" videos for the lab study.
4.  **Lab Study Analysis**: The selected videos are used in a controlled lab experiment to validate the online survey findings and test specific hypotheses about how the sequence of events in a ride (e.g., a sudden negative event on a pleasant route) impacts overall experience.

## 📂 Project Structure

## Project Structure

```
├── data/                  
│   ├── context_data/
│   ├── video_traces/
│   └── ...
├── output_data/           
├── utils/                 
│   ├── clustering_utils.py
│   ├── helper_functions.py
│   ├── lmm_utils.py
│   ├── plotting_utils.py
│   └── processing_utils.py
├── config.ini                  # IMPORTANT: Configuration file for all paths and models
├── constants.py                # Stores constant variables (column names, categories, etc.)
├── feature_extraction.py  
├── online_survey_analysis.py 
├── candidate_video_prediction.py 
├── lab_study_analysis.py  
└── requirements.txt            # Python package dependencies
```

---

## 📜 Scripts and Modules

This project contains four main analysis scripts supported by a central configuration file and several utility modules.

### `feature_extraction.py`

This script runs a computer vision pipeline to extract objective features directly from video files. It uses a semantic segmentation model to quantify the presence of environmental elements.

**Key Functions:**
* **Frame Extraction**: Automatically extracts frames from videos at a set interval.
* **Semantic Segmentation**: Uses a pre-trained `nvidia/segformer` model to identify features like vegetation, buildings, roads, etc., in each frame.
* **Feature Calculation**: Calculates the "greenery ratio" (percentage of pixels identified as vegetation/terrain) for each video.
* **Data Integration**: Merges the calculated greenery ratio into the ground truth data file (`video_info_ground_truth.json`).
* **Validation**: Compares features extracted by an LLM against the ground truth data to measure LLM accuracy.

### `online_survey_analysis.py`

This script processes and analyzes the data from the initial online survey to establish baseline affective scores for each video.

**Key Functions:**
* **Data Preprocessing**: Loads, cleans, and transforms raw survey responses into a structured format.
* **Affective Score Calculation**: Calculates **valence** (pleasantness/unpleasantness) and **arousal** (intensity of feeling) for each video based on participant ratings.
* **Statistical Analysis**: Employs Linear Mixed-Effects Models (LMMs) to analyze how affective responses differ across demographic groups.

### `candidate_video_prediction.py`

This script combines the subjective survey scores with objective geospatial data to train a model that can predict the "bikeability" of new, unseen routes.

**Key Functions:**
* **Geospatial Enrichment**: Aggregates GPX traces for each video and enriches them with contextual data layers (e.g., bike networks, traffic volume, green spaces).
* **Clustering**: Uses k-means clustering to group videos based on their objective environmental features.
* **Predictive Modeling**: Trains a k-Nearest Neighbors (k-NN) model to predict a video's valence score from its geospatial features.
* **Candidate Selection**: Applies the trained model to a set of new videos to predict their valence, identifying the best candidates for the lab study.

### `lab_study_analysis.py`

This script analyzes the data from the controlled lab experiment to test specific hypotheses about cognitive biases in sequence evaluation (e.g., peak-end rule).

**Key Functions:**
* **Data Integration**: Loads and merges data from the lab experiment, online survey (for validation), and the predictive model.
* **Validation Analysis**: Compares the valence ratings from the lab, the online survey, and the model's predictions using Bland-Altman plots and correlation metrics.
* **Hypothesis Testing**: Uses LMMs to analyze how the position of a "spoiler" event (a non-bikeable segment in a pleasant ride, or vice versa) affects the overall rating of the sequence.

---

## 🚀 Getting Started

Follow these steps to set up the project environment and run the analysis pipeline.

### Prerequisites

You will need Python 3.8+ and the packages listed in `requirements.txt`. 

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repository.git](https://github.com/your-username/your-repository.git)
    cd your-repository
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **‼️ Configure Data Paths:**
    This is the most important step. Open the **`config.ini`** file and **update all file and directory paths** to match the locations on your local machine. The scripts will not run without the correct paths.

    ```ini
    [paths]
    # Example: update this to your local path
    video_candidates_path = /Users/yourname/Documents/cycling_study/data/video_candidates
    input_dir = /Users/yourname/Documents/cycling_study/data/input_data/video_traces
    # ... and so on for all other paths ...
    ```

### Running the Analysis Pipeline

The scripts are designed to be run in a specific order. Open your terminal, activate the virtual environment, and run the following commands.

1.  **Run the Feature Extraction Pipeline:**
    The `feature_extraction.py` script is modular. You can run the full pipeline or individual stages using flags.

    ```bash
    # Run the entire pipeline: extract frames, segment, merge, and compare
    python llm_feature_extraction.py --all

    # Or run individual stages:
    python llm_feature_extraction.py --extract  # Stage 1: Extracts frames from videos
    python llm_feature_extraction.py --segment  # Stage 2: Performs semantic segmentation
    python llm_feature_extraction.py --merge    # Stage 3: Merges greenery data into ground truth
    python llm_feature_extraction.py --compare  # Stage 4: Compares LLM predictions to ground truth
    ```

2.  **Run the Online Survey Analysis:**
    ```bash
    python online_survey_analysis.py
    ```

3.  **Run the Candidate Video Prediction:**
    ```bash
    python candidate_video_prediction.py
    ```

4.  **Run the Lab Study Analysis:**
    ```bash
    python lab_study_analysis.py
    ```

All outputs, including CSV files and plots, will be saved to the `output_dir` specified in your `config.ini`.