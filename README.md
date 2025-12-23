# Analysis of Cyclists' Affective Responses to Environmental Factors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

## 📖 Project Overview
This project analyzes how environmental factors impact a cyclist's emotional experience (affective response). The workflow uses results from a large online survey to inform a controlled lab study, with a key focus on comparing manually created vs. LLM-generated video descriptions for downstream analyses purposes.

The core pipeline is as follows:

* **Predictive Modeling & Candidate Selection**: Analyzes cyclist ratings from a large survey to train a model that selects affectively diverse videos for the lab study.
* **Video Content Description**: Uses two parallel methods—manual "ground truth" labeling and automated LLM-based feature extraction—to describe video events and environmental features.
* **Physiological Data Processing**: Extracts heart rate (PPG) and skin conductance (EDA) metrics, calculating baseline-corrected "Deltas" to measure physiological reactivity.
* **Lab Study & SEM Analysis**: Performs general analysis of lab ratings and employs Structural Equation Modeling (SEM) to analyze how subjective and physiological responses are influenced by static (e.g., scenery) and dynamic (e.g., traffic) elements.
* **Presence & Immersion**: Analyzes the Igroup Presence Questionnaire (IPQ) to evaluate participant immersion during the lab study across different demographics.

## 📂 Project Directory Structure
```
cycling_experience/
├── input_data/ 
│ ├── context_data/                 # Stores geospatial data layers (e.g., bike networks, traffic volume) 
│ ├── video_traces/                 # Contains GPX traces for cycling routes 
│ ├── video_candidates/             # Raw video files for analysis 
│ ├── online_results/               # Raw survey data from the online study
│ └── lab_results/                  # Raw data from the lab study
│
├── output_data/                    # Stores all outputs (e.g., processed data, predictions, plots) 
│ └── video_data/                   # Processed video data and extracted features 
│ ...
│
├── utils/                          # Utility scripts for data processing, plotting, etc. 
│ ├── clustering_utils.py 
│ ├── helper_functions.py 
│ ├── lmm_utils.py 
│ ├── plotting_utils.py 
│ └── processing_utils.py 
│
├── build_ground_truth.py           # Script for processing manual labels and geospatial features into the ground truth dataset
├── llm_feature_extraction.py       # Script for extracting features from videos using LLMs
├── online_survey_analysis.py       # Script for analyzing online survey data 
├── candidate_video_prediction.py   # Script for predictive modeling and candidate video selection 
├── lab_study_analysis.py           # Script for analyzing lab study data
├── static_dynamic_analysis)SEM.py  # Script for analyzing static vs. dynamic video features using Structural Equation Modelling. This pipeline corresponds to the following publication: Understanding Subjective Cycling Experience, with Static, Dynamic and Physiological Cues. 
│
├── config.ini                      # Configuration file for paths and models
├── constants.py                    # Stores constant variables (e.g., column names, categories) 
└── requirements.txt                # Python package dependencies
```

## 🛠️ Setup and Installation

1.  **Clone the repository**:
    ```bash
    git clone [https://github.com/mie-lab/cycling_experience.git](https://github.com/mie-lab/cycling_experience.git)
    cd cycling_experience
    ```

2.  **Create and Activate a Virtual Environment**:
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 📥 Data Download and Setup

1.  **Download Data**:
    * Place datasets (geospatial data, video files, survey results) in their respective folders under `input_data/`.
    * Ensure the folder structure matches the directory description provided above.
2.  **Update Paths in `config.ini`**:
    * Update all file and directory paths to match your local machine.
    * Provide a `gemini_api_key` if you intend to run `llm_feature_extraction.py`.

## 🚀 Running the Analysis Pipeline

1.  **Generate the Ground Truth**:
    ```bash
    python build_ground_truth.py
    ```
    * **Description**: Aggregates geospatial data (traffic, greenery, bike networks) and runs semantic segmentation on video frames.
    * **Output**: 30 frames per video, `segmentation_results.csv`, and the master `video_ground_truth.csv`.

2.  **Run the Online Survey Analysis**:
    ```bash
    python online_survey_analysis.py
    ```
    * **Description**: Processes online ratings to assess valence/arousal, and generates demographic summaries to establish 'bikeable' or 'non-bikeable labels'.
    * **Output**: Processed survey data and affect-grid visualizations in `output_data/`.

3.  **Predict Candidate Videos**:
    ```bash
    python candidate_video_prediction.py
    ```
    * **Description**: Uses KNN clustering and RMSE optimization to predict valence for candidate videos based on geospatial and semantic features.
    * **Output**: `candidate_predictions.csv` with predicted valence scores.

4.  **Analyze Lab Study Data**:
    ```bash
    python lab_study_analysis.py
    ```
    * **Description**: Performs block-level analysis (Validation, Equal, Positive, and Negative scenarios) and tests positional effects of "spoilers" using Linear Mixed Models (LMMs).
    * **Output**: Scenario-specific visualizations and statistical model comparisons.

5.  **Extract Features Using LLMs**:
    ```bash
    python llm_feature_extraction.py
    ```
    * **Description**: Sends video files to the Gemini 2.5 Flash API to extract environmental features (lane counts, surface material, motorized traffic speed) via a Pydantic-validated prompt.
    * **Output**: `video_llm_info.csv` containing automated video features.

6.  **Process Physiological Data**:
    ```bash
    python physiological_data_analysis.py
    ```
    * **Description**: Processes physiological signals to extract cleaned EDA (SCL, SCR) and PPG (HR, HRV) signals.
    * **Output**: `physiological_results.csv` with event-related and tonic metrics.

7.  **Run SEM & Causal Analysis**:
    ```bash
    python static_dynamic_analysis_SEM.py
    ```
    * **Description**: Fits Structural Equation Models (SEM) and runs LiNGAM causal discovery to evaluate how infrastructure, visual elements, and dynamic events drive subjective and physiological affect.
    * **Output**: Path diagrams, model fit statistics (`SEM_model_comparison.csv`), and coefficient matrices.

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.