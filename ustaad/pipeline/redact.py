"""Render privacy redaction onto the full-fps source video (and onto sampled frames for the auditor)."""
from __future__ import annotations

import bisect
from pathlib import Path

import cv2
import numpy as np

from ..perception.masks import MaskStore
from ..schemas import FrameRef
from ..video import ClipWriter, iter_video, probe

BoxFix = dict  # {"t0": float, "t1": float, "box": [x,y,w,h]}


def apply_style(img: np.ndarray, mask: np.ndarray, style: str, block: int) -> np.ndarray:
    if not mask.any():
        return img
    if style == "fill":
        out = img.copy()
        out[mask] = (40, 40, 40)
        return out
    if style == "blur":
        k = max(31, (min(img.shape[:2]) // 12) | 1)
        hidden = cv2.GaussianBlur(img, (k, k), 0)
    else:  # pixelate
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
        hidden = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out = img.copy()
    out[mask] = hidden[mask]
    return out


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def privacy_mask(store: MaskStore, idx: int, keys: set[str], shape: tuple[int, int], dilate_px: int,
                 neighbours: list[int] | None = None) -> np.ndarray:
    m = np.zeros(shape, bool)
    for j in [idx] + (neighbours or []):
        u = store.union(j, keys=keys)
        if u is None:
            continue
        if u.shape != shape:
            u = cv2.resize(u.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        m |= u
    return _dilate(m, dilate_px)


def box_mask(fixes: list[BoxFix], t: float, shape: tuple[int, int], pad: float = 0.02) -> np.ndarray:
    m = np.zeros(shape, bool)
    H, W = shape
    for f in fixes:
        if f["t0"] <= t <= f["t1"]:
            x, y, w, h = f["box"]
            x0, y0 = int(max(0, x - pad) * W), int(max(0, y - pad) * H)
            x1, y1 = int(min(1, x + w + pad) * W), int(min(1, y + h + pad) * H)
            m[y0:y1, x0:x1] = True
    return m


def render_video(src: Path, out: Path, refs: list[FrameRef], store: MaskStore, keys: set[str],
                 fixes: list[BoxFix], cfg, overlay_out: Path | None = None) -> int:
    """Returns number of output frames that contain any redaction."""
    info = probe(src)
    times = [r.t for r in refs]
    writer = ClipWriter(out, info["fps"], (info["width"], info["height"]))
    ov = ClipWriter(overlay_out, info["fps"], (info["width"], info["height"])) if overlay_out else None
    dil = int(cfg.dilate_px * info["width"] / 960)  # dilate is specified at 960px
    redacted = 0
    for i, frame in iter_video(src):
        t = i / info["fps"]
        k = max(0, bisect.bisect_right(times, t) - 1)
        nb = [k + 1] if cfg.temporal_union and k + 1 < len(refs) else []
        shape = frame.shape[:2]
        m = privacy_mask(store, k, keys, shape, dil, nb) | box_mask(fixes, t, shape)
        if m.any():
            redacted += 1
        writer.write(apply_style(frame, m, cfg.style, max(4, int(cfg.pixel_block * info["width"] / 960))))
        if ov:
            tint = frame.copy()
            tint[m] = (0.45 * frame[m] + 0.55 * np.array([255, 0, 255])).astype(np.uint8)
            ov.write(tint)
    writer.close()
    if ov:
        ov.close()
    return redacted


def render_sampled(frames_dir: Path, out_dir: Path, refs: list[FrameRef], store: MaskStore, keys: set[str],
                   fixes: list[BoxFix], cfg, only: list[int] | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in refs:
        if only is not None and r.idx not in only:
            continue
        img = cv2.imread(str(frames_dir / f"{r.idx:05d}.jpg"))
        nb = [r.idx + 1] if cfg.temporal_union and r.idx + 1 < len(refs) else []
        m = privacy_mask(store, r.idx, keys, img.shape[:2], cfg.dilate_px, nb) | box_mask(fixes, r.t, img.shape[:2])
        cv2.imwrite(str(out_dir / f"{r.idx:05d}.jpg"), apply_style(img, m, cfg.style, cfg.pixel_block))
