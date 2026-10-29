# Cycling Experience: Affective Responses to Street Environments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

Code for our studies on how street environments shape cyclists' emotional experience,
measured as valence (pleasant–unpleasant) and arousal (calm–stressed).

The project has two data sources:

- **Online survey:** participants rated short first-person cycling videos from Zurich.
  These ratings show which street features make a ride feel more or less pleasant,
  and they were used to select the videos for the lab study.
- **Lab study:** participants watched single videos and sequences of videos while their skin
  conductance (EDA) and heart rate (PPG) were recorded, and rated each one.

Each video is described by its street features: manual labels (e.g. traffic events),
GIS data (e.g. traffic volume, bike infrastructure), image segmentation (e.g. greenery)
and features extracted with a multimodal LLM (Gemini).

The analyses ask (1) what drives cyclists' ratings in the online survey, (2) how the order
of street segments along a route shapes how the whole route is remembered, and (3) how much
static features (the street layout) and dynamic features (traffic and other events)
explain cyclists' ratings and physiological responses.

## Installation

```bash
git clone https://github.com/mie-lab/cycling_experience.git
cd cycling_experience
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Data

1. Put the data in `input_data/` (GIS layers, GPX traces, videos, survey and lab results).
2. Set the paths in `config.ini` to your machine.
3. To run the LLM feature extraction, add a Gemini API key to `config.ini`.

All outputs are written to `output_data/`. Run every script from the repository root.

## Usage

**1. Prepare the video and physiological data** (needed by all analyses)

```bash
python build_ground_truth.py            # video features: manual labels, GIS, segmentation
python llm_feature_extraction.py        # video features extracted with Gemini
python physiological_data_analysis.py   # EDA and heart-rate measures
```

**2. Online survey analysis**

```bash
python online_survey_analysis.py        # survey ratings and subgroup analysis
python candidate_video_prediction.py    # selects the videos for the lab study
```

**3. Sequence effects analysis** (how segment order shapes route ratings)

```bash
python sequence_effects_analysis.py     # main analysis (lab Tasks 1–3)
python bikeability_indices_example.py   # appendix: bikeability indices
```

**4. Static vs dynamic features analysis** (path models of ratings and physiology)

```bash
python dynamic_gap_analysis.py          # main analysis
python mllm_extraction_figures.py       # appendix Table A1: LLM vs manual features
```

Analysis decisions for this analysis are logged in `REVISION_LOG.md`.

## License

MIT. See [LICENSE](LICENSE).
