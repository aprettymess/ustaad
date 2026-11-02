"""Frame sampling and clip writing (OpenCV; ffmpeg used for browser-friendly H.264 if present)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .schemas import FrameRef


def ingest(src: str | Path, dst: Path, max_height: int = 1080) -> Path:
    """Normalize any phone video (HEVC, rotation metadata, odd fps) to H.264 MP4 that OpenCV and browsers both read.
    ffmpeg applies rotation automatically. Falls back to a plain copy if ffmpeg is missing."""
    dst = Path(dst)
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-vf", f"scale=-2:'min({max_height},ih)'", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k",
                        "-movflags", "+faststart", str(dst)], check=True)
    else:
        shutil.copy(src, dst)
    return dst


def probe(video: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frames"] / info["fps"] if info["fps"] else 0
    cap.release()
    return info


def _resize(frame: np.ndarray, max_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return frame
    return cv2.resize(frame, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)


def sample_frames(video: str | Path, out_dir: Path, fps: float, max_side: int) -> list[FrameRef]:
    """Write sampled frames as <idx>.jpg (the naming SAM 3 expects for JPEG folders)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(src_fps / fps, 1.0)
    refs: list[FrameRef] = []
    i, next_take = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i >= next_take:
            idx = len(refs)
            cv2.imwrite(str(out_dir / f"{idx:05d}.jpg"), _resize(frame, max_side), [cv2.IMWRITE_JPEG_QUALITY, 92])
            refs.append(FrameRef(idx=idx, t=round(i / src_fps, 3)))
            next_take += step
        i += 1
    cap.release()
    if not refs:
        raise ValueError(f"no frames read from {video}")
    return refs


def load_frame(frames_dir: Path, idx: int) -> np.ndarray:
    img = cv2.imread(str(frames_dir / f"{idx:05d}.jpg"))
    if img is None:
        raise FileNotFoundError(frames_dir / f"{idx:05d}.jpg")
    return img


def iter_video(video: str | Path):
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield i, frame
        i += 1
    cap.release()


class ClipWriter:
    """Writes mp4v with OpenCV, then (if ffmpeg exists) transcodes to H.264 so browsers can play it."""

    def __init__(self, path: Path, fps: float, size: tuple[int, int]):
        self.path = Path(path)
        self.tmp = self.path.with_suffix(".raw.mp4")
        self.w = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    def write(self, frame: np.ndarray) -> None:
        self.w.write(frame)

    def close(self) -> Path:
        self.w.release()
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.tmp), "-c:v", "libx264",
                            "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)],
                           check=True)
            self.tmp.unlink(missing_ok=True)
        else:
            self.tmp.replace(self.path)
        return self.path


def write_clip(frames: list[np.ndarray], path: Path, fps: float) -> Path:
    h, w = frames[0].shape[:2]
    cw = ClipWriter(path, fps, (w, h))
    for f in frames:
        cw.write(f)
    return cw.close()


def side_by_side(a: np.ndarray, b: np.ndarray, label_a: str = "MASTER", label_b: str = "YOU") -> np.ndarray:
    h = min(a.shape[0], b.shape[0], 540)
    a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
    b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
    out = np.concatenate([a, np.full((h, 8, 3), 255, np.uint8), b], axis=1)
    for x, text in ((10, label_a), (a.shape[1] + 18, label_b)):
        cv2.putText(out, text, (x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (x, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out
