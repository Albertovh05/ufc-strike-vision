# UFC Strike Tracker

Automated per-round strike counting from UFC broadcast video, validated against
[UFCStats](http://ufcstats.com) ground truth.

## What it does

UFC broadcasts carry official strike statistics (significant strikes, total strikes,
accuracy, targets) that are published on UFCStats after every event. This project
builds a computer-vision pipeline that reproduces those counts directly from video,
round by round, fighter by fighter — without relying on the in-broadcast HUD overlay.

The pipeline has four stages:

| Stage | Module | Description |
|---|---|---|
| Scrape | `src/scraper.py` | Downloads per-fight strike totals from UFCStats as JSON |
| Preprocess | `src/preprocess.py` | Extracts and normalises video frames at 25 fps |
| Detect | `src/detect.py` | Detects individual strike events in the frame sequence |
| Re-ID | `src/reid.py` | Associates each detected strike with a fighter identity |

Predictions are compared against UFCStats ground truth using the eval harness in
`src/eval.py`. Five fights are held out in `data/holdout/` and never touched during
development; all reported numbers come from this set.

## Project layout

```
ufc-tracker/
  data/
    raw/          # UFCStats JSON files (one per fight)
    processed/    # Extracted video frames
    holdout/      # 5 fights reserved for final eval
  src/
    scraper.py    # UFCStats scraper
    preprocess.py # Frame extraction and normalisation
    detect.py     # Strike detection model
    reid.py       # Fighter re-identification
    eval.py       # Evaluation harness
  notebooks/
    explore.ipynb # EDA on scraped data
  tests/
    test_eval.py  # Unit tests for the eval harness
```

## Setup

Requires Python 3.11.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

**Scrape UFCStats** (writes JSON to `data/raw/`):

```bash
python -m src.scraper
```

**Collect YouTube full fights + UFCStats counts**:

```bash
python -m src.collect_youtube_fights --limit 5
```

This writes `data/raw/youtube_fights_manifest.json` plus one UFCStats JSON file
per matched fight. It records YouTube URLs by default. To also store authorized
local video files under ignored `data/videos/`, add `--download-videos`.

**Run tests:**

```bash
pytest tests/
```

**Evaluate predictions against holdout fights:**

```python
from src.eval import StrikePrediction, run_holdout_eval

predictions = {
    "fight_id_here": [
        StrikePrediction("fight_id_here", round_number=1, fighter_index=0,
                         sig_strikes=18, total_strikes=27),
        # ...
    ]
}
summary = run_holdout_eval(predictions)
print(summary)
```

## Metrics

Primary metric is **Mean Absolute Error (MAE)** on significant-strike count per
round per fighter, measured on the held-out five fights. A secondary MAE is reported
for total strikes.

## Data policy

Raw scraped data is excluded from version control (see `.gitignore`). Do not commit
video files or fighter images. Holdout fight IDs should not be used during model
development or hyperparameter tuning.
