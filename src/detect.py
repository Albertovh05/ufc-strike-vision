"""Strike detection primitives for broadcast fight video.

The heavy model choices in the handoff (YOLO/ByteTrack/pose model) are deliberately
kept behind small interfaces here. That lets the pipeline be tested with injected
detectors today, while still allowing a real YOLO or pose backend to be plugged in
later without changing downstream code.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - depends on local environment
    cv2 = None


FrameInput = np.ndarray | tuple[float, np.ndarray]
Detector = Callable[[np.ndarray], Iterable[object]]


def _require_cv2():
    if cv2 is None:
        raise ImportError("opencv-python is required for this detection helper")
    return cv2


@dataclass
class BBox:
    """Pixel-space bounding box in xyxy format."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.width / 2.0, self.y1 + self.height / 2.0)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def clip(self, frame_shape: tuple[int, ...]) -> "BBox":
        h, w = frame_shape[:2]
        return BBox(
            x1=float(np.clip(self.x1, 0, w - 1)),
            y1=float(np.clip(self.y1, 0, h - 1)),
            x2=float(np.clip(self.x2, 0, w - 1)),
            y2=float(np.clip(self.y2, 0, h - 1)),
        )


@dataclass
class PersonDetection:
    frame_index: int
    timestamp_ms: float
    bbox: BBox
    confidence: float
    track_id: int | None = None


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float = 1.0


@dataclass
class PoseFrame:
    frame_index: int
    timestamp_ms: float
    track_id: int
    keypoints: dict[str, Keypoint]


@dataclass
class CandidateStrike:
    """A high-velocity limb movement to pass to a clip classifier."""

    frame_index: int
    timestamp_ms: float
    track_id: int
    keypoint_name: str
    velocity_px_s: float
    window_start: int
    window_end: int
    confidence: float


def _split_frame_input(frame_input: FrameInput, frame_index: int, fps: float) -> tuple[float, np.ndarray]:
    if isinstance(frame_input, tuple):
        timestamp_ms, frame = frame_input
        return float(timestamp_ms), frame
    return (frame_index / fps) * 1000.0, frame_input


def _coerce_bbox(raw_bbox: object) -> BBox:
    if isinstance(raw_bbox, BBox):
        return raw_bbox
    values = list(raw_bbox)  # type: ignore[arg-type]
    if len(values) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {len(values)}")
    return BBox(*(float(v) for v in values))


def _coerce_detection(raw: object) -> tuple[BBox, float, int | None]:
    """Accept common detector output shapes and normalize to this module's schema."""
    if isinstance(raw, PersonDetection):
        return raw.bbox, raw.confidence, None

    if isinstance(raw, dict):
        bbox = _coerce_bbox(raw["bbox"])
        confidence = float(raw.get("confidence", raw.get("score", 1.0)))
        class_id = raw.get("class_id", raw.get("cls"))
        return bbox, confidence, None if class_id is None else int(class_id)

    values = list(raw)  # type: ignore[arg-type]
    if len(values) < 4:
        raise ValueError("Detection tuples must contain at least x1, y1, x2, y2")
    bbox = BBox(*(float(v) for v in values[:4]))
    confidence = float(values[4]) if len(values) >= 5 else 1.0
    class_id = int(values[5]) if len(values) >= 6 and values[5] is not None else None
    return bbox, confidence, class_id


def _ultralytics_detector(model_name: str = "yolov8n.pt") -> Detector:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for the default YOLO detector. Install it or "
            "pass a custom detector callable to detect_people()."
        ) from exc

    model = YOLO(model_name)

    def detect(frame: np.ndarray) -> Iterable[tuple[float, float, float, float, float, int]]:
        result = model(frame, verbose=False)[0]
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            yield (x1, y1, x2, y2, confidence, class_id)

    return detect


def detect_people(
    frames: Sequence[FrameInput],
    detector: Detector | None = None,
    *,
    fps: float = 25.0,
    conf_threshold: float = 0.25,
    person_class_id: int = 0,
) -> list[PersonDetection]:
    """Detect people in a frame sequence.

    `detector` should return one detection per object as either a dict with
    `bbox`/`confidence`/`class_id`, a tuple `(x1, y1, x2, y2, score, class_id)`,
    or a `PersonDetection`. When omitted, an optional Ultralytics YOLO detector is
    used if the package is installed.
    """
    detector = detector or _ultralytics_detector()
    detections: list[PersonDetection] = []

    for frame_index, frame_input in enumerate(frames):
        timestamp_ms, frame = _split_frame_input(frame_input, frame_index, fps)
        for raw in detector(frame):
            bbox, confidence, class_id = _coerce_detection(raw)
            if confidence < conf_threshold:
                continue
            if class_id is not None and class_id != person_class_id:
                continue
            detections.append(
                PersonDetection(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    bbox=bbox.clip(frame.shape),
                    confidence=confidence,
                )
            )

    return detections


def iou(a: BBox, b: BBox) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - intersection
    return 0.0 if union <= 0 else intersection / union


def group_by_frame(detections: Iterable[PersonDetection]) -> dict[int, list[PersonDetection]]:
    grouped: dict[int, list[PersonDetection]] = defaultdict(list)
    for detection in detections:
        grouped[detection.frame_index].append(detection)
    return dict(grouped)


def track_detections(
    detections: Sequence[PersonDetection],
    *,
    iou_threshold: float = 0.25,
    max_missing: int = 12,
) -> list[PersonDetection]:
    """Assign simple IoU-based track IDs.

    This is a lightweight baseline, not a replacement for ByteTrack. It gives
    downstream code stable IDs on easy clips and keeps the module functional until
    a full tracker is added.
    """
    next_track_id = 0
    active: dict[int, tuple[BBox, int]] = {}
    tracked: list[PersonDetection] = []
    detections_by_frame = group_by_frame(detections)

    for frame_index in sorted(detections_by_frame):
        frame_detections = sorted(
            detections_by_frame[frame_index],
            key=lambda d: d.confidence,
            reverse=True,
        )
        assigned_tracks: set[int] = set()

        for detection in frame_detections:
            best_track_id: int | None = None
            best_iou = 0.0
            for track_id, (last_bbox, last_seen) in active.items():
                if track_id in assigned_tracks or frame_index - last_seen > max_missing:
                    continue
                score = iou(detection.bbox, last_bbox)
                if score > best_iou:
                    best_iou = score
                    best_track_id = track_id

            if best_track_id is None or best_iou < iou_threshold:
                best_track_id = next_track_id
                next_track_id += 1

            detection.track_id = best_track_id
            active[best_track_id] = (detection.bbox, frame_index)
            assigned_tracks.add(best_track_id)
            tracked.append(detection)

        active = {
            track_id: state
            for track_id, state in active.items()
            if frame_index - state[1] <= max_missing
        }

    return tracked


def detect_and_track(
    frames: Sequence[FrameInput],
    detector: Detector | None = None,
    *,
    fps: float = 25.0,
    conf_threshold: float = 0.25,
) -> list[PersonDetection]:
    detections = detect_people(frames, detector, fps=fps, conf_threshold=conf_threshold)
    return track_detections(detections)


def crop_detection(frame: np.ndarray, detection: PersonDetection, padding: int = 0) -> np.ndarray:
    bbox = detection.bbox
    padded = BBox(
        bbox.x1 - padding,
        bbox.y1 - padding,
        bbox.x2 + padding,
        bbox.y2 + padding,
    ).clip(frame.shape)
    x1, y1, x2, y2 = (int(round(v)) for v in padded.as_xyxy())
    return frame[y1:y2, x1:x2]


def detect_velocity_spikes(
    poses: Sequence[PoseFrame],
    *,
    keypoint_names: Sequence[str] = ("left_wrist", "right_wrist", "left_ankle", "right_ankle"),
    threshold_px_s: float = 900.0,
    window_frames: int = 8,
    min_keypoint_confidence: float = 0.4,
    cooldown_frames: int = 5,
    fallback_fps: float = 25.0,
) -> list[CandidateStrike]:
    """Find candidate strikes from fast wrist/ankle motion."""
    by_track_keypoint: dict[tuple[int, str], list[PoseFrame]] = defaultdict(list)
    for pose in poses:
        for name in keypoint_names:
            keypoint = pose.keypoints.get(name)
            if keypoint and keypoint.confidence >= min_keypoint_confidence:
                by_track_keypoint[(pose.track_id, name)].append(pose)

    candidates: list[CandidateStrike] = []
    last_emit: dict[tuple[int, str], int] = {}

    for (track_id, name), track_poses in by_track_keypoint.items():
        track_poses.sort(key=lambda p: (p.frame_index, p.timestamp_ms))
        previous: PoseFrame | None = None
        for pose in track_poses:
            if previous is None:
                previous = pose
                continue

            prev_kp = previous.keypoints[name]
            kp = pose.keypoints[name]
            dt_s = (pose.timestamp_ms - previous.timestamp_ms) / 1000.0
            if dt_s <= 0:
                dt_s = max(1, pose.frame_index - previous.frame_index) / fallback_fps

            distance = float(np.hypot(kp.x - prev_kp.x, kp.y - prev_kp.y))
            velocity = distance / dt_s
            cooldown_key = (track_id, name)
            enough_gap = pose.frame_index - last_emit.get(cooldown_key, -10_000) >= cooldown_frames

            if velocity >= threshold_px_s and enough_gap:
                candidates.append(
                    CandidateStrike(
                        frame_index=pose.frame_index,
                        timestamp_ms=pose.timestamp_ms,
                        track_id=track_id,
                        keypoint_name=name,
                        velocity_px_s=velocity,
                        window_start=max(0, pose.frame_index - window_frames),
                        window_end=pose.frame_index + window_frames,
                        confidence=float(min(1.0, velocity / (threshold_px_s * 2.0))),
                    )
                )
                last_emit[cooldown_key] = pose.frame_index

            previous = pose

    return sorted(candidates, key=lambda c: (c.frame_index, c.track_id, c.keypoint_name))


def classify_combat_state(frame_detections: Sequence[PersonDetection]) -> str:
    """Heuristic combat-state classifier: distance, clinch, or ground."""
    if len(frame_detections) < 2:
        return "distance"

    fighters = sorted(frame_detections, key=lambda d: d.confidence, reverse=True)[:2]
    a, b = fighters[0].bbox, fighters[1].bbox
    overlap = iou(a, b)
    center_dx = abs(a.center[0] - b.center[0])
    avg_width = max(1.0, (a.width + b.width) / 2.0)
    avg_aspect = ((a.height / max(1.0, a.width)) + (b.height / max(1.0, b.width))) / 2.0

    if avg_aspect < 1.1:
        return "ground"
    if overlap > 0.08 or center_dx < avg_width * 0.75:
        return "clinch"
    return "distance"


def draw_detections(frame: np.ndarray, detections: Sequence[PersonDetection]) -> np.ndarray:
    """Return a copy of `frame` with boxes and track IDs drawn for debugging."""
    cv = _require_cv2()
    out = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in detection.bbox.as_xyxy())
        cv.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"id={detection.track_id}" if detection.track_id is not None else "person"
        cv.putText(out, label, (x1, max(0, y1 - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out
