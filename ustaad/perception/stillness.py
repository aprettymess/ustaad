"""Find 'hands-off' checkpoints: runs of sampled frames with low motion (and, if tracked, no hands)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..schemas import FrameRef, StillMoment, TrackedObject


def motion_energy(frames_dir: Path, n: int, size: int = 160) -> np.ndarray:
    """Mean abs grayscale difference to the previous sampled frame (energy[0] = 0)."""
    prev, out = None, np.zeros(n)
    for i in range(n):
        g = cv2.cvtColor(cv2.imread(str(frames_dir / f"{i:05d}.jpg")), cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(cv2.resize(g, (size, int(size * g.shape[0] / g.shape[1]))), (5, 5), 0)
        if prev is not None:
            out[i] = float(np.abs(g.astype(np.int16) - prev).mean())
        prev = g.astype(np.int16)
    return out


def find_still_moments(energy: np.ndarray, refs: list[FrameRef], motion_thresh: float, min_len: int,
                       tracks: dict[int, list[TrackedObject]] | None = None, hand_area_max: float = 1.0
                       ) -> list[StillMoment]:
    n = len(refs)
    still = energy < motion_thresh
    still[0] = True
    if tracks is not None:
        for i in range(n):
            hand = sum(o.area for o in tracks.get(i, []) if o.role == "hand")
            if hand > hand_area_max:
                still[i] = False
    runs, i = [], 0
    while i < n:
        if still[i]:
            j = i
            while j + 1 < n and still[j + 1]:
                j += 1
            if j - i + 1 >= min_len or i == 0 or j == n - 1:
                runs.append((i, j))
            i = j + 1
        else:
            i += 1
    # first and last frame are always checkpoints (start and end state)
    picks = [i for i, _ in runs[:1]] + [(a + b) // 2 for a, b in runs[1:-1]] + ([runs[-1][1]] if len(runs) > 1 else [])
    if not picks or picks[0] != 0:
        picks = [0] + picks
    if picks[-1] != n - 1:
        picks.append(n - 1)
    picks = sorted(set(picks))
    return [StillMoment(k=k, frame=refs[i]) for k, i in enumerate(picks)]
