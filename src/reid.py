"""Fighter re-identification utilities.

This module provides a deterministic appearance baseline that works with the
tracked detections from `src.detect`. It is intentionally modest: color histogram
profiles are enough to exercise the pipeline and catch data-shape problems before
replacing this with OSNet or another learned embedding model.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from src.detect import BBox, FrameInput, PersonDetection

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None


FIGHTER_A = "A"
FIGHTER_B = "B"


def _require_cv2():
    if cv2 is None:
        raise ImportError("opencv-python is required for appearance embeddings")
    return cv2


@dataclass
class AppearanceProfile:
    fighter: str
    embedding: np.ndarray
    samples: int = 1

    def update(self, embedding: np.ndarray, weight: float = 1.0) -> None:
        total = self.samples + weight
        self.embedding = ((self.embedding * self.samples) + (embedding * weight)) / total
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm
        self.samples = int(round(total))


@dataclass
class FighterAssignment:
    frame_index: int
    timestamp_ms: float
    track_id: int
    fighter: str
    confidence: float


def _split_frame_input(frame_input: FrameInput, frame_index: int, fps: float) -> tuple[float, np.ndarray]:
    if isinstance(frame_input, tuple):
        timestamp_ms, frame = frame_input
        return float(timestamp_ms), frame
    return (frame_index / fps) * 1000.0, frame_input


def _frame_lookup(frames: Sequence[FrameInput], fps: float) -> dict[int, tuple[float, np.ndarray]]:
    return {
        frame_index: _split_frame_input(frame_input, frame_index, fps)
        for frame_index, frame_input in enumerate(frames)
    }


def crop_bbox(frame: np.ndarray, bbox: BBox, padding: int = 4) -> np.ndarray:
    padded = BBox(
        bbox.x1 - padding,
        bbox.y1 - padding,
        bbox.x2 + padding,
        bbox.y2 + padding,
    ).clip(frame.shape)
    x1, y1, x2, y2 = (int(round(v)) for v in padded.as_xyxy())
    return frame[y1:y2, x1:x2]


def color_histogram_embedding(
    frame: np.ndarray,
    bbox: BBox,
    *,
    bins: tuple[int, int, int] = (8, 8, 8),
) -> np.ndarray:
    """Return a normalized HSV color histogram for a detected fighter crop."""
    cv = _require_cv2()
    crop = crop_bbox(frame, bbox)
    if crop.size == 0:
        return np.zeros(int(np.prod(bins)), dtype=np.float32)

    hsv = cv.cvtColor(crop, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    embedding = hist.astype(np.float32).reshape(-1)
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def detections_by_track(detections: Iterable[PersonDetection]) -> dict[int, list[PersonDetection]]:
    tracks: dict[int, list[PersonDetection]] = defaultdict(list)
    for detection in detections:
        if detection.track_id is None:
            raise ValueError("All detections must have track_id before re-identification")
        tracks[detection.track_id].append(detection)
    return dict(tracks)


def _track_embedding(
    frames_by_index: Mapping[int, tuple[float, np.ndarray]],
    detections: Sequence[PersonDetection],
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    for detection in detections:
        frame_entry = frames_by_index.get(detection.frame_index)
        if frame_entry is None:
            continue
        _, frame = frame_entry
        embeddings.append(color_histogram_embedding(frame, detection.bbox))

    if not embeddings:
        return np.zeros(8 * 8 * 8, dtype=np.float32)

    mean = np.mean(np.vstack(embeddings), axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm > 0 else mean


def build_bootstrap_profiles(
    frames: Sequence[FrameInput],
    detections: Sequence[PersonDetection],
    *,
    fps: float = 25.0,
    bootstrap_frames: int = 75,
) -> dict[str, AppearanceProfile]:
    """Create Fighter A/B profiles from the first visible two tracks.

    The initial mapping is screen-position based: leftmost track becomes A and
    rightmost track becomes B. This is a bootstrap convention for development,
    not a final source of truth for UFC red/blue-corner identity.
    """
    frames_by_index = _frame_lookup(frames, fps)
    early = [
        detection
        for detection in detections
        if detection.track_id is not None and detection.frame_index < bootstrap_frames
    ]
    tracks = detections_by_track(early)
    if len(tracks) < 2:
        raise ValueError("Need at least two tracked fighters in bootstrap frames")

    ranked_tracks = sorted(
        tracks.items(),
        key=lambda item: (
            -len(item[1]),
            np.mean([d.bbox.center[0] for d in item[1]]),
        ),
    )
    top_two = ranked_tracks[:2]
    top_two.sort(key=lambda item: np.mean([d.bbox.center[0] for d in item[1]]))

    profiles: dict[str, AppearanceProfile] = {}
    for fighter, (_, track_detections) in zip((FIGHTER_A, FIGHTER_B), top_two):
        embedding = _track_embedding(frames_by_index, track_detections)
        profiles[fighter] = AppearanceProfile(
            fighter=fighter,
            embedding=embedding,
            samples=len(track_detections),
        )
    return profiles


def assign_tracks_to_profiles(
    frames: Sequence[FrameInput],
    detections: Sequence[PersonDetection],
    profiles: Mapping[str, AppearanceProfile],
    *,
    fps: float = 25.0,
) -> dict[int, tuple[str, float]]:
    """Assign each track ID to the closest fighter profile."""
    frames_by_index = _frame_lookup(frames, fps)
    assignments: dict[int, tuple[str, float]] = {}

    for track_id, track_detections in detections_by_track(detections).items():
        embedding = _track_embedding(frames_by_index, track_detections)
        scores = {
            fighter: cosine_similarity(embedding, profile.embedding)
            for fighter, profile in profiles.items()
        }
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_fighter, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        confidence = float(np.clip((best_score - second_score + 1.0) / 2.0, 0.0, 1.0))
        assignments[track_id] = (best_fighter, confidence)

    return assignments


def assign_fighters(
    frames: Sequence[FrameInput],
    detections: Sequence[PersonDetection],
    profiles: Mapping[str, AppearanceProfile] | None = None,
    *,
    fps: float = 25.0,
    bootstrap_frames: int = 75,
) -> list[FighterAssignment]:
    """Assign every tracked detection to Fighter A or Fighter B."""
    if profiles is None:
        profiles = build_bootstrap_profiles(
            frames,
            detections,
            fps=fps,
            bootstrap_frames=bootstrap_frames,
        )

    track_assignments = assign_tracks_to_profiles(frames, detections, profiles, fps=fps)
    assignments: list[FighterAssignment] = []

    for detection in detections:
        if detection.track_id is None:
            raise ValueError("All detections must have track_id before re-identification")
        fighter, confidence = track_assignments.get(detection.track_id, ("unknown", 0.0))
        assignments.append(
            FighterAssignment(
                frame_index=detection.frame_index,
                timestamp_ms=detection.timestamp_ms,
                track_id=detection.track_id,
                fighter=fighter,
                confidence=confidence,
            )
        )

    return assignments


def majority_fighter_by_track(assignments: Iterable[FighterAssignment]) -> dict[int, str]:
    """Collapse per-frame assignments to one fighter label per track."""
    votes: dict[int, Counter[str]] = defaultdict(Counter)
    for assignment in assignments:
        votes[assignment.track_id][assignment.fighter] += 1
    return {
        track_id: counter.most_common(1)[0][0]
        for track_id, counter in votes.items()
        if counter
    }
