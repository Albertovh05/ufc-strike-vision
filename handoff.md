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
- (a) Per-round total event counts within ±15 % of official UFCStats round-level statistics
- (b) Strike-type classification accuracy ≥ 75 % on individually labeled strike clips

Output schema stability and a consistent taxonomy are first-class requirements —
the timeline feeds downstream pattern-detection work.

---

## Is this achievable?

Yes. Strike recognition from broadcast video is a solved research problem at the
component level. The full pipeline stacks several well-understood building blocks:

```
Video frames
  → Person detection + tracking  (YOLOv8 + ByteTrack)
  → Pose estimation per frame     (MediaPipe Pose or ViTPose)
  → Keypoint trajectory analysis  (wrist/foot velocity spikes → candidate events)
  → Fighter re-identification      (appearance embeddings + Hungarian matching)
  → Clip-level strike classifier   (3D-CNN or small Video Transformer)
  → Output timeline JSON
```

**What's well-solved (off-the-shelf):**
- Pose estimation — MediaPipe Pose and ViTPose both give reliable wrist/elbow/
  shoulder/foot keypoints at broadcast resolution
- Short-clip action classification — 3D-CNNs (X3D, SlowFast) and Video Transformers
  (TimeSformer, VideoMAE) handle ±15-frame windows well
- Person tracking — YOLOv8 + ByteTrack handles multi-person broadcast scenes

**What's hard:**
- Clinch/ground occlusion — fighters overlap; pose keypoints get confused
- Camera cuts — re-ID must survive hard cuts and instant angle changes
- Strike speed — a jab takes 20–80 ms; at 25 fps you get 1–3 frames of commitment
  phase, so temporal windowing around the peak matters a lot
- Fighter re-ID persistence across a full 25-minute fight

**The gating blocker — labeled training data does not exist:**

UFCStats provides round-level totals only (e.g. "47 sig strikes in round 1").
It has no per-strike-type breakdown and no frame-level timestamps. Before a
strike-type classifier can be trained, someone must label individual strike clips
with their type. Options:

| Approach | Effort | Notes |
|---|---|---|
| Manual labeling | High — weeks of work | Full control; label exactly what you need |
| Existing datasets | Low if a match exists | HMDB51 / FineGym / Kinetics have some combat clips; none are UFC-specific |
| Weak supervision | Medium | Use UFCStats round totals as a count signal; build a separate clip labeling tool |

Until labeled clips exist the count-validation target (±15 % of UFCStats totals)
is achievable but the ≥75 % type-classification target is not.

---

## Current state of the code

```
src/scraper.py         ✓  Complete — UFCStats per-round stats scraper
src/eval.py            ✓  Complete — evaluation harness (MAE vs ground truth)
src/preprocess.py      ✓  Skeleton — frame extraction from video (untested end-to-end)
src/detect.py          ✗  Empty stub
src/reid.py            ✗  Empty stub
tests/test_scraper.py  ✓  53 tests, all passing
tests/test_eval.py     ✓  5 tests, all passing
data/raw/              ~50 scraped fights (JSON)
data/processed/        empty
data/holdout/          empty — 5 fights must be reserved before any modelling begins
```

---

## Scraper (`src/scraper.py`)

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
        "sig_strikes":           { "landed": 25, "attempted": 51 },
        "sig_strikes_pct":       "49%",
        "total_strikes":         { "landed": 35, "attempted": 65 },
        "takedowns":             { "landed": 0,  "attempted": 1  },
        "takedown_pct":          "0%",
        "submission_attempts":   0,
        "reversals":             0,
        "ctrl_time":             "0:00",
        "sig_strikes_head":      { "landed": 18, "attempted": 35 },
        "sig_strikes_body":      { "landed": 5,  "attempted": 10 },
        "sig_strikes_leg":       { "landed": 2,  "attempted": 6  },
        "sig_strikes_distance":  { "landed": 15, "attempted": 30 },
        "sig_strikes_clinch":    { "landed": 5,  "attempted": 10 },
        "sig_strikes_ground":    { "landed": 5,  "attempted": 11 }
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

---

## Eval harness (`src/eval.py`)

Loads ground truth from `data/holdout/`, merges against `StrikePrediction` objects,
computes per-row MAE for sig strikes and total strikes. Entry point:

```python
from src.eval import StrikePrediction, run_holdout_eval
summary_df = run_holdout_eval({"fight_id": [StrikePrediction(...), ...]})
```

---

## Preprocessor (`src/preprocess.py`)

`extract_frames(video_path, fps_target=25)` → `list[np.ndarray]`.
`save_frames(frames, out_dir)` writes `frame_NNNNNN.jpg` files.
No normalization pipeline beyond resize to 1280 px wide. Not tested end-to-end
(needs real video files).

Millisecond timestamps are not yet captured. To fix: change `extract_frames` to
also read `cap.get(cv2.CAP_PROP_POS_MSEC)` alongside each frame and return
`list[tuple[float, np.ndarray]]` (timestamp_ms, frame).

---

## Known blockers

| Issue | Detail |
|---|---|
| No video data | The entire CV pipeline (detect.py, reid.py, stance detection, timestamp extraction) cannot be built or tested until video files are available. |
| Holdout set not populated | `data/holdout/` is empty. The eval harness references it but cannot run until 5 fights are scraped and moved there. These 5 fights must be chosen BEFORE any modelling to avoid leakage. |
| Strike-type labels don't exist | UFCStats provides only round totals — no per-strike-type breakdown. The ≥75 % classification accuracy target requires a separate labeled dataset of individual strike clips. |
| Stance detection not started | Orthodox vs southpaw per-frame classification is a prerequisite for "lead vs rear" labeling. No approach has been chosen or prototyped. Likely implemented as a foot-position rule on pose keypoints with a short temporal smoothing window. |
| Millisecond timestamps not captured | `preprocess.py` extracts frames at a fixed FPS sample rate but does not read `cv2.CAP_PROP_POS_MSEC`. Needed for the output timeline. |
| Live UFCStats fetch | `ECONNREFUSED` on both fight-detail and event-detail fetch attempts during development. Scrapers were written against documented HTML structure from the community, not verified live. First real scrape may expose wrong selectors. |

---

## Full build sequence

### Phase 0 — Data foundation (before any modelling)

1. **Verify the scraper against a real event URL.**
   ```bash
   python src/scraper.py \
       --url http://ufcstats.com/event-details/<any-recent-event-id> \
       --output data/raw/ --delay 1.0
   ```
   Check that `data/raw/` fills with valid JSON. If files are empty or round counts
   are wrong, inspect the live HTML and fix the selectors in `_find_rows()`.

2. **Scrape ~50 events, then designate holdout fights.**
   Pick 5 fights that span weight class and fight duration variation. Move their
   JSON files to `data/holdout/` and never touch them again until final eval.

3. **Acquire video files** for at least the non-holdout fights. Source and format
   are TBD — the pipeline assumes files that OpenCV can open (mp4/mkv etc.).

### Phase 1 — Preprocessing

4. **Add millisecond timestamps to `preprocess.py`.**
   Change `extract_frames` to capture `cap.get(cv2.CAP_PROP_POS_MSEC)` alongside
   each frame and return `list[tuple[float, np.ndarray]]`.

5. **Run preprocessing end-to-end** on a real video file and verify
   `data/processed/<fight_id>/frame_NNNNNN.jpg` files look correct.

### Phase 2 — Person detection and tracking

6. **Implement `src/detect.py` — Phase 1: bounding box detection.**
   Use YOLOv8 (people class) + ByteTrack to get stable per-person bounding box
   tracks across frames. Output: `list[dict]` with frame index, track ID, bbox.

7. **Implement `src/reid.py` — fighter identity assignment.**
   Assign track IDs to Fighter A vs Fighter B consistently across the full fight,
   including across camera cuts. Standard approach: appearance embedding
   (e.g. OSNet or a lightweight ResNet) + Hungarian matching against corner colors
   or jersey color anchors established in the first few frames.

### Phase 3 — Pose estimation and candidate detection

8. **Add pose estimation to `src/detect.py`.**
   Run MediaPipe Pose or ViTPose on each fighter's cropped bounding box region
   to get 33-keypoint skeletons. Store alongside frame metadata.

9. **Implement wrist/foot velocity spike detection.**
   Compute per-keypoint velocity across frames. Threshold spikes on wrist and
   foot keypoints as candidate strike events. Apply a ±N frame temporal window
   (start with N=8, ~320 ms at 25 fps) around each spike to create candidate clips.

10. **Implement combat-state classifier** (distance / clinch / ground).
    Input: relative bounding box positions + keypoint layout. This gates the
    downstream classifier — distance strikes and clinch strikes look different.

### Phase 4 — Stance detection

11. **Implement per-frame stance classifier** (orthodox / southpaw).
    Input: foot keypoints from pose skeleton. Rule-based: if left foot is forward,
    stance is orthodox (for a right-handed fighter). Apply a short temporal
    smoothing window (e.g. 5-frame majority vote) to avoid jitter. Stance can
    switch mid-fight (switch-hitter); the classifier must handle this.

### Phase 5 — Strike-type classification

12. **Build a labeled clip dataset.**
    This is the gating dependency for the ≥75 % accuracy target. Options:
    - Manual labeling tool: a simple script that plays a 30-frame clip and
      asks for a label from the taxonomy
    - Existing dataset augmentation (HMDB51 "punch" / "kick" classes as a
      bootstrap; fine-tune on UFC clips)
    - Target: ~300–500 labeled clips minimum before training

13. **Train clip-level strike-type classifier.**
    Input: ±N frames around a candidate event, cropped to the active fighter,
    with pose keypoints overlaid or concatenated as a channel.
    Architecture options (in order of complexity):
    - **X3D-S** or **SlowFast** — strong baseline for short-clip action recognition
    - **VideoMAE fine-tune** — best accuracy if you have enough labeled data
    - **Keypoint-only MLP/LSTM** — faster to train, lower ceiling, good for
      prototyping before committing to a full video model
    Labels: use the taxonomy from the Project Goal section above.

### Phase 6 — Integration and output

14. **Integrate all components into a single pipeline script.**
    `python src/run_pipeline.py --video <path> --fight-id <id>` should output a
    timeline JSON to `data/output/<fight_id>.json`.

    Output schema per event:
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

15. **Run holdout evaluation.**
    Use `src/eval.py` to compare predicted round totals against UFCStats ground
    truth. Target: MAE within ±15 % of official totals across all 5 holdout fights.

---

## Recommended model choices (current best practice)

| Task | Recommended | Alternative |
|---|---|---|
| Person detection | YOLOv8n (fast) | RT-DETR |
| Person tracking | ByteTrack | DeepSORT |
| Pose estimation | MediaPipe Pose (CPU-friendly) | ViTPose-S (more accurate) |
| Re-ID embedding | OSNet-x0.25 | MobileNetV3 + ArcFace |
| Strike classifier | X3D-S fine-tuned | VideoMAE-S fine-tuned |
| Stance | Rule on foot keypoints | Lightweight CNN on cropped lower body |

---

## Files being actively developed

- `src/scraper.py` — done; next touch-point is verifying selectors against a real event URL
- `src/preprocess.py` — needs millisecond timestamp fix (Phase 1, step 4)
- `src/detect.py` — next major work item (Phase 2, step 6)
- `src/reid.py` — blocked on detect.py (Phase 2, step 7)
