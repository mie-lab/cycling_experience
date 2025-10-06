# Analysis of Cyclists' Affective Responses to Environmental Factors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

## 📖 Project Overview
This project analyzes how environmental factors impact a cyclist's emotional experience (affective response). The workflow uses results from a large online survey to inform a controlled lab study, with a key focus on comparing manually created vs. LLM-generated video descriptions for downstream analyses purposes.

The core pipeline is as follows:

* **Predictive Modeling & Candidate Selection**: The `online_survey_analysis.py` script analyzes cyclist ratings from a large survey. This data is then used by `candidate_video_prediction.py` to train a model that selects specific, affectively diverse videos for the controlled lab study.

* **Video Content Description (Manual vs. Automated)**: The project uses two parallel methods to create descriptive features for the videos.
    * **Ground Truth**: `build_ground_truth.py` processes **manually created labels** to generate a structured "ground truth" dataset of video events and environmental features.
    * **LLM-based**: As an alternative to manual labeling, `llm_feature_extraction.py` uses a Large Language Model (LLM) to **automatically generate video descriptions** without prior knowledge of the ground truth.

* **Lab Study Analysis**: The `lab_study_analysis.py` script performs a general analysis of the lab study ratings. The `static_dynamic_analysis.py` script then uses the feature datasets (either the ground truth or the LLM-generated ones) to analyze *how* lab participants' ratings were influenced by combinations of **static** (e.g., scenery) and **dynamic** (e.g., traffic events) elements in the videos.

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
├── static_dynamic_analysis.py      # Script for analyzing static vs. dynamic video features
│
├── config.ini                      # Configuration file for paths and models
├── constants.py                    # Stores constant variables (e.g., column names, categories) 
└── requirements.txt                # Python package dependencies
```

## 🛠️ Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mie-lab/cycling_experience.git
   cd cycling_experience
   ```

2. **Create and Activate a Virtual Environment**:
   ```bash
   # For Unix/macOS
   python3 -m venv venv
   source venv/bin/activate

   # For Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
## 📥 Data Download and Setup

1. **Download Data**:
   - Download the required datasets (e.g., geospatial data, video files, survey results) and store them in their respective folders under the `input_data/` directory. Please request access to the data if needed.
   - Ensure the folder structure matches the one described above.
   - If you also downloaded the output data and stored it in `output_data/video_data`, some of the scripts can be skipped.

2. **Update Paths in `config.ini`**:
   - Open the `config.ini` file and update all file and directory paths to match the locations on your local machine. Make sure to specify the `root` directory of the project and provide `gemini_api_key` if the `llm_feature_extraction` is used.

## 🚀 Running the Analysis Pipeline

1. **Generate the Ground Truth**:  
   ```bash
   python build_ground_truth.py
   ```
   - Output: 30 frames per each video extracted and stored in `input_data/video_candidates/` in a separate folder for each video.
   - Output: `sementation_results.csv` file with semantic segmentation results (30 frames/video) in `output_data/video_data/`.
   - Output: `video_ground_truth.csv` file with video information in `output_data/video_data/`.
   - Note: This step can be skipped if you already have the output data.
2. **Run the Online Survey Analysis**:  
   ```bash
   python online_survey_analysis.py
   ```
    - Output: processed survey data and visualizations in `output_data/`.
   
3. **Predict Candidate Videos**:  
   ```bash
    python candidate_video_prediction.py
   ```
    - Output: `candidate_predictions.csv` file with video valence predictions in `output_data/video_data/`.
    - Note: This step can be skipped if you already have the output data.
4. **Analyze Lab Study Data**:  
   ```bash
   python lab_study_analysis.py
   ```
    - Output: processed lab study data and visualizations in `output_data/`.
   
5. **Extract Features from Videos Using LLMs**:  
   ```bash
   python llm_feature_extraction.py
   ```
   - Output: `video_llm_info.json` file with LLM-extracted video features in `output_data/video_data/`.
   - Note: This step can be skipped if you already have the output data.
6. **Static vs Dynamic Analysis**:  
   ```bash
   python static_dynamic_analysis.py
   ```
   - in-progress
   

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.