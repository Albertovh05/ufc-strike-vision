# UFC Strike Vision

Computer-vision tooling for turning UFC broadcast video into a structured
timeline of strikes. The goal is to detect individual strike attempts, classify
their type, attribute them to the correct fighter, and measure the time between
events so coaches, managers, and analysts can study repeatable fight patterns.

## What it does

UFCStats provides useful round-level totals after each event, but it does not
provide a per-strike timeline, strike-type labels, lead/rear limb context, stance,
or the time gaps between actions. This project builds the pipeline needed to
learn those events directly from video.

The intended output is one event per detected strike or takedown:

```json
{
  "timestamp_ms": 12430,
  "fighter": "A",
  "strike_type": "jab",
  "stance": "orthodox",
  "hand_leg": "lead",
  "combat_state": "distance",
  "confidence": 0.87
}
```

Once those events exist, round-level counts become a derived output rather than
the primary objective. The richer timeline can support coach/manager workflows
such as combination detection, pace analysis, stance-switch patterns, counter
timing, and opponent tendency datasets.

Minimum strike taxonomy:

| Category | Types |
|---|---|
| Punches | jab, cross, lead hook, rear hook, lead uppercut, rear uppercut, overhand |
| Kicks | lead leg kick, rear leg kick, body kick, head kick |
| Other | knee, elbow, takedown |

The pipeline is built in stages:

| Stage | Module | Description |
|---|---|---|
| Collect | `src/collect_youtube_fights.py` | Finds full-fight videos and matches them to UFCStats fight pages |
| Scrape | `src/scraper.py` | Downloads UFCStats round-level strike counts for weak validation |
| Preprocess | `src/preprocess.py` | Extracts normalized video frames for model input |
| Detect | `src/detect.py` | Detects fighters, tracks people, and proposes candidate strike moments |
| Re-ID | `src/reid.py` | Associates tracked people with Fighter A or Fighter B |
| Classify | planned | Classifies candidate clips into the strike taxonomy above |
| Timeline | planned | Emits timestamped strike events and time-between-strike features |

UFCStats counts are used as a weak validation target: after the model detects and
classifies events, its derived per-round counts should be close to official
round-level totals. Strike-type accuracy requires a separately labeled clip
dataset because UFCStats does not provide per-strike labels or timestamps.

## Project layout

```
ufc-tracker/
  data/
    raw/          # UFCStats JSON files (one per fight)
    videos/       # Local full-fight videos; ignored by git
    processed/    # Extracted video frames
    holdout/      # 5 fights reserved for final eval
  src/
    collect_youtube_fights.py # YouTube/UFCStats matching manifest builder
    scraper.py    # UFCStats scraper
    preprocess.py # Frame extraction and normalisation
    detect.py     # Detection/tracking/candidate strike primitives
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

**Collect YouTube full-fight metadata + UFCStats counts**:

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

**Evaluate derived counts against holdout fights:**

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

The final system needs several validation views:

| Metric | Purpose |
|---|---|
| Strike-type accuracy | Measures whether candidate clips are classified correctly, using individually labeled strike clips |
| Count error vs UFCStats | Checks whether timeline-derived per-round counts stay close to official UFCStats totals |
| Timestamp quality | Measures whether event times are close enough for sequence and time-gap analysis |
| Fighter attribution accuracy | Measures whether each event is assigned to the correct fighter |

The existing `src/eval.py` harness currently covers count error against UFCStats.
The strike-type and timestamp metrics depend on building a labeled clip dataset.

## Data policy

Raw scraped data is excluded from version control (see `.gitignore`). Do not commit
video files, fighter images, labeled clips, or derived datasets that may contain
sensitive or licensed material. Holdout fight IDs should not be used during model
development or hyperparameter tuning.
