# UFC Tracker — Session Handoff

## Project Goal

Build an automated computer-vision pipeline that ingests UFC broadcast video and
produces a structured, millisecond-accurate timeline of every strike and takedown,
attributed to the correct fighter and classified by specific strike type.

**Minimum required strike taxonomy**

| Category | Types |
|---|---|
| Punches | jab, cross, lead hook, rear hook, lead uppercut, rear uppercut, overhand |
| Kicks | lead leg kick, rear leg kick, body kick, head kick |
| Other | knee, elbow, takedown |

Each event must carry:
- Fighter identity (A or B)
- Stance at time of throw (orthodox / southpaw — may switch mid-fight)
- Hand/leg relative to stance (lead vs rear)
- Combat state: distance / clinch / ground (matching UFCStats taxonomy)
- Millisecond timestamp
- Per-event confidence score

**Validation targets (holdout set)**
- (a) Per-round total event counts within ±15 % of official UFCStats round-level
  statistics
- (b) Strike-type classification accuracy ≥ 75 % on individually labeled strike clips

Output schema stability and a consistent taxonomy are first-class requirements —
the timeline feeds downstream pattern-detection work.

---

## Current state of the code

### What exists and works

```
src/scraper.py       ✓  Complete — UFCStats per-round stats scraper
src/eval.py          ✓  Complete — evaluation harness (MAE vs ground truth)
src/preprocess.py    ✓  Skeleton — frame extraction from video (untested end-to-end)
src/detect.py        ✗  Empty stub
src/reid.py          ✗  Empty stub
tests/test_scraper.py  ✓  53 tests, all passing
tests/test_eval.py     ✓  5 tests, all passing
data/raw/            empty (no fights scraped yet)
data/processed/      empty
data/holdout/        empty — 5 fights must be reserved before any modelling begins
```

### Scraper (`src/scraper.py`)

Scrapes UFCStats event pages → per-round JSON.

**CLI:**
```bash
python src/scraper.py --url http://ufcstats.com/event-details/<id> --output data/raw/
```

**Output schema per fight (`{la}_{lb}_{YYYYMMDD}.json`):**
```json
{
  "fight_id": "str",
  "date": "YYYY-MM-DD",
  "fighter_a": "Full Name",
  "fighter_b": "Full Name",
  "rounds": [
    {
      "round": 1,
      "fighter_a": {
        "knockdowns": 0,
        "sig_strikes":        { "landed": 25, "attempted": 51 },
        "sig_strikes_pct":    "49%",
        "total_strikes":      { "landed": 35, "attempted": 65 },
        "takedowns":          { "landed": 0,  "attempted": 1  },
        "takedown_pct":       "0%",
        "submission_attempts": 0,
        "reversals":          0,
        "ctrl_time":          "0:00",
        "sig_strikes_head":   { "landed": 18, "attempted": 35 },
        "sig_strikes_body":   { "landed": 5,  "attempted": 10 },
        "sig_strikes_leg":    { "landed": 2,  "attempted": 6  },
        "sig_strikes_distance": { "landed": 15, "attempted": 30 },
        "sig_strikes_clinch": { "landed": 5,  "attempted": 10 },
        "sig_strikes_ground": { "landed": 5,  "attempted": 11 }
      },
      "fighter_b": { "...same fields..." }
    }
  ]
}
```

**Key HTML assumption (must verify against live site):**  
UFCStats per-round tables use one `<tr>` per round where each `<td>` holds two
`<p>` children — `p[0]` = fighter A (red corner), `p[1]` = fighter B (blue corner).
Per-round sections are distinguished from per-fight aggregate sections by the
`_rnd` vs `_tot` suffix on the `b-fight-details__collapse-link` class.
The selector in `_find_rows()` (`p[class*='collapse-link_rnd']`) is the one line
that may need adjustment on first real run.

### Eval harness (`src/eval.py`)

Loads ground truth from `data/holdout/`, merges against `StrikePrediction` objects,
computes per-row MAE for sig strikes and total strikes. Entry point:

```python
from src.eval import StrikePrediction, run_holdout_eval
summary_df = run_holdout_eval({"fight_id": [StrikePrediction(...), ...]})
```

### Preprocessor (`src/preprocess.py`)

`extract_frames(video_path, fps_target=25)` → `list[np.ndarray]`.  
`save_frames(frames, out_dir)` writes `frame_NNNNNN.jpg` files.  
No normalization pipeline beyond resize to 1280 px wide. Not tested end-to-end
(needs real video files).

---

## Files being actively developed

- `src/scraper.py` — done for this session; next touch-point is verifying selectors
  against a real event URL
- `src/detect.py` — next major work item (empty)
- `src/reid.py` — blocked on detect.py (empty)

---

## Everything tried that failed / known blockers

| Issue | Detail |
|---|---|
| Live UFCStats fetch during development | `ECONNREFUSED` on both the fight-detail and event-detail fetch attempts. Scrapers were written against documented HTML structure from the community, not verified live. First real scrape may expose wrong selectors. |
| No video data yet | The entire CV pipeline (detect.py, reid.py, stance detection, timestamp extraction) cannot be built or tested until video files are available. preprocess.py is a skeleton that assumes OpenCV can open the file but has never run on real footage. |
| Holdout set not populated | `data/holdout/` is empty. The eval harness references it but cannot run until 5 fights are scraped and moved there. These 5 fights must be chosen BEFORE any modelling to avoid leakage. |
| Strike-type labels don't exist | UFCStats provides only totals (sig strikes, head/body/leg, distance/clinch/ground) — it has no per-strike-type breakdown. The ≥75 % classification accuracy target requires a separate labeled dataset of individual strike clips that does not yet exist. |
| Stance detection not started | Orthodox vs southpaw per-frame classification is a prerequisite for "lead vs rear" labeling. No approach has been chosen or prototyped. |
| Millisecond timestamps not started | The current preprocess.py extracts frames at a fixed FPS sample rate. Recovering exact frame timestamps from variable-framerate broadcast video requires reading `cv2.CAP_PROP_POS_MSEC` per frame — this is noted but not implemented. |

---

## Next step to take

**Immediate (before any modelling):**

1. **Verify the scraper against a real event URL.**  
   Run:
   ```bash
   python src/scraper.py --url http://ufcstats.com/event-details/<any-recent-event-id> \
       --output data/raw/ --delay 1.0
   ```
   Check that `data/raw/` fills with valid JSON. If the files are empty or the
   round counts are wrong, inspect the live HTML and fix the selectors in
   `_find_rows()` in `src/scraper.py`.

2. **Scrape ~50 events, then designate holdout fights.**  
   Pick 5 fights that span weight class and fight duration variation. Move their
   JSON files to `data/holdout/` and never touch them again until final eval.

**After data is confirmed:**

3. **Implement millisecond timestamps in `preprocess.py`.**  
   Change `extract_frames` to capture `cap.get(cv2.CAP_PROP_POS_MSEC)` alongside
   each frame and return `list[tuple[float, np.ndarray]]` (timestamp_ms, frame).

4. **Build `src/detect.py` — the core CV pipeline.**  
   Recommended sequence:
   - Pose estimation per frame (MediaPipe Pose or ViTPose) to get keypoint
     trajectories for both fighters
   - Wrist/foot velocity spikes as candidate strike events
   - Temporal windowing (±N frames) around each candidate for clip-level
     classification
   - Classify combat state (distance / clinch / ground) as a prerequisite to
     stride the detection differently per state

5. **Build `src/reid.py` — fighter identity assignment.**  
   Must assign bounding boxes / pose tracks to fighter A vs B consistently across
   the full fight, even when fighters cross or the camera cuts. A re-ID model
   (e.g. appearance embedding + Hungarian matching) is the standard approach.

6. **Stance detection (orthodox / southpaw).**  
   Per-frame classifier on the relative foot positions from the pose skeleton.
   Stance can switch; use a short temporal smoothing window to avoid jitter.

7. **Strike-type classifier.**  
   Once pose keypoints + stance + combat state are available, train a clip-level
   classifier (3D-CNN or transformer over the ±N-frame window) against a labeled
   dataset. This dataset does not yet exist — creating it (manually labeling a
   few hundred clips or finding an existing MMA action-recognition dataset) is
   the gating dependency for the ≥75 % accuracy target.
