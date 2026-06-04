"""Broadcast preprocessing pipeline — prepare raw video frames for detection."""

import json
from pathlib import Path

import cv2
import numpy as np

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_fight_metadata(fight_id: str) -> dict:
    path = RAW_DIR / f"{fight_id}.json"
    return json.loads(path.read_text())


def resize_frame(frame: np.ndarray, width: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_LINEAR)


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32) / 255.0


def extract_frames(video_path: str | Path, fps_target: float = 25.0) -> list[np.ndarray]:
    """Extract frames from a video file at a target sample rate."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps_target
    step = max(1, round(src_fps / fps_target))
    frames: list[np.ndarray] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            frames.append(resize_frame(frame))
        idx += 1

    cap.release()
    return frames


def save_frames(frames: list[np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(out_dir / f"frame_{i:06d}.jpg"), frame)


def process_video(video_path: str | Path, fight_id: str) -> None:
    out_dir = PROCESSED_DIR / fight_id
    frames = extract_frames(video_path)
    save_frames(frames, out_dir)
    print(f"Processed {len(frames)} frames -> {out_dir}")
